use crate::internal::*;
use crate::ops::cnn::PaddingSpec;
use crate::ops::nn::{DataFormat, DataShape};
use ndarray::prelude::*;

use super::PatchAxis;

use std::fmt::Debug;
use std::ops::Range;

use tract_itertools::{izip, Itertools};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PatchSpec {
    pub input_shape: TVec<usize>,
    pub input_inner_stride: usize,
    pub output_inner_stride: usize,
    pub kernel_shape: TVec<usize>,
    pub strides: TVec<usize>,
    pub dilations: TVec<usize>,
    pub padding: PaddingSpec,
}

impl Debug for PatchSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "input: {} kernel: {} strides: {} dil: {} pad: {:?}",
            self.input_shape.iter().join(","),
            self.kernel_shape.iter().join(","),
            self.strides.iter().join(","),
            self.dilations.iter().join(","),
            self.padding
        )
    }
}

impl PatchSpec {
    pub fn for_full_shape(
        data_format: DataFormat,
        input_full_shape: &[usize],
    ) -> TractResult<PatchSpec> {
        let shape = data_format.shape(input_full_shape.into())?;
        Ok(Self::for_data_shape(shape))
    }

    pub fn for_data_shape(data_shape: DataShape) -> PatchSpec {
        let input_shape: TVec<usize> = data_shape.hw_dims().into();
        PatchSpec {
            kernel_shape: tvec!(1; input_shape.len()),
            input_inner_stride: *data_shape.w_stride(),
            output_inner_stride: 1,
            strides: tvec!(1; input_shape.len()),
            dilations: tvec!(1; input_shape.len()),
            padding: PaddingSpec::Valid,
            input_shape,
        }
    }

    pub fn with_kernel_shape(self, kernel_shape: TVec<usize>) -> PatchSpec {
        PatchSpec { kernel_shape, ..self }
    }

    pub fn with_dilations(self, dilations: TVec<usize>) -> PatchSpec {
        PatchSpec { dilations, ..self }
    }

    pub fn with_strides(self, strides: TVec<usize>) -> PatchSpec {
        PatchSpec { strides, ..self }
    }

    pub fn with_padding(self, padding: PaddingSpec) -> PatchSpec {
        PatchSpec { padding, ..self }
    }

    pub fn with_output_inner_stride(self, output_inner_stride: usize) -> PatchSpec {
        PatchSpec { output_inner_stride, ..self }
    }

    pub fn into_patch(self) -> Patch {
        let dims = self.padding.compute(
            &self.input_shape,
            &self.kernel_shape,
            &self.dilations,
            &self.strides,
        );
        let output: TVec<usize> = dims.iter().map(|d| d.convoluted).collect();
        let pad_before: TVec<usize> = dims.iter().map(|d| d.pad_before).collect();
        let pad_after: TVec<usize> = dims.iter().map(|d| d.pad_after).collect();

        let data_field: Vec<isize> = ::ndarray::indices(&*self.kernel_shape)
            .into_iter()
            .flat_map(|coords| {
                #[allow(clippy::unnecessary_to_owned)] // I think this one is a clippy bug.
                coords
                    .slice()
                    .to_vec()
                    .into_iter()
                    .enumerate()
                    .map(|(ix, c)| (c * self.dilations[ix]) as isize - pad_before[ix] as isize)
            })
            .collect();
        let data_field = Array2::from_shape_vec(
            (self.kernel_shape.iter().cloned().product(), self.kernel_shape.len()),
            data_field,
        )
        .unwrap();
        let data_field_min_max: TVec<_> = data_field
            .columns()
            .into_iter()
            .map(|col| (col.iter().min().cloned().unwrap(), col.iter().max().cloned().unwrap()))
            .collect();

        fn strides(shape: &[usize], inner: usize) -> TVec<isize> {
            let mut strides: TVec<isize> = tvec![inner as isize];
            for dim in shape.iter().skip(1).rev() {
                let previous = *strides.last().unwrap();
                strides.push(*dim as isize * previous);
            }
            strides.reverse();
            strides
        }

        let input_storage_strides = strides(&self.input_shape, self.input_inner_stride);
        let output_storage_strides = strides(&output, self.output_inner_stride);

        let standard_layout_data_field: Vec<isize> = data_field
            .outer_iter()
            .map(|coords| izip!(coords, &input_storage_strides).map(|(a, b)| a * b).sum::<isize>())
            .collect();

        // regions[axis][range+mask]
        let regions: TVec<TVec<_>> = dims
            .iter()
            .enumerate()
            .map(|(ix, d)| {
                PatchAxis {
                    input_dim: self.input_shape[ix],
                    kernel_dim: self.kernel_shape[ix],
                    pad_before: d.pad_before,
                    pad_after: d.pad_after,
                    output_dim: d.convoluted,
                    stride: self.strides[ix],
                    dilation: self.dilations[ix],
                }
                .regions()
            })
            .collect::<TVec<_>>();

        let zone_strides = strides(&regions.iter().map(|d| d.len()).collect::<TVec<_>>(), 1);

        let zones: Vec<Zone> = regions
            .iter()
            .multi_cartesian_product()
            .map(|regions| Zone {
                input_zone_offset: 0,
                output_ranges: regions.iter().map(|reg| reg.range.clone()).collect(),
                output_shape: regions.iter().map(|reg| reg.range.end - reg.range.start).collect(),
                output_zone_offset: izip!(&regions, &output_storage_strides)
                    .map(|(reg, &stride)| reg.range.start as isize * stride)
                    .sum::<isize>(),
                valid: regions.iter().all(|reg| reg.mask.is_none()),
                values_offsets: izip!(
                    0..,
                    ndarray::indices(&*self.kernel_shape),
                    &standard_layout_data_field
                )
                .filter(|(_ix, coords, _offset)| {
                    izip!(coords.slice(), &regions)
                        .all(|(&x, axis)| !axis.mask.as_ref().map(|mask| mask[x]).unwrap_or(false))
                })
                .map(|(ix, _coords, &window_offset)| (ix, window_offset))
                .collect(),
            })
            .collect();

        let valid_zone = zones.iter().position(|z| z.valid);

        let mut valid_output_zone = tvec!();
        let mut invalid_output_zones = tvec!();
        for ix in 0..self.input_shape.len() {
            let min_max = data_field_min_max[ix];
            let min = (-min_max.0 as usize).divceil(self.strides[ix]);
            let max =
                (self.input_shape[ix].saturating_sub(min_max.1 as usize)).divceil(self.strides[ix]);
            if min != 0 {
                let mut invalid = valid_output_zone.clone();
                invalid.push(0..min);
                while invalid.len() < output.len() {
                    invalid.push(0..output[invalid.len()])
                }
                invalid_output_zones.push(invalid);
            }
            if max < output[ix] {
                let mut invalid = valid_output_zone.clone();
                invalid.push(max..output[ix]);
                while invalid.len() < output.len() {
                    invalid.push(0..output[invalid.len()])
                }
                invalid_output_zones.push(invalid);
            }
            valid_output_zone.push(min..max)
        }

        let op_strides_times_input_storage_strides =
            izip!(&self.strides, &input_storage_strides).map(|(a, b)| (*a as isize * b)).collect();

        Patch {
            spec: self,
            padded: pad_before.iter().any(|&p| p != 0) || pad_after.iter().any(|&p| p != 0),
            pad_before,
            pad_after,
            output_shape: output,
            data_field,
            data_field_min_max,
            standard_layout_data_field,
            input_storage_strides,
            output_storage_strides,
            op_strides_times_input_storage_strides,
            valid_output_zone,
            invalid_output_zones,
            zones,
            valid_zone_id: valid_zone,
            zone_strides,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Patch {
    pub spec: PatchSpec,
    pub pad_before: TVec<usize>,
    pub pad_after: TVec<usize>,
    pub padded: bool,
    pub output_shape: TVec<usize>,
    pub data_field: Array2<isize>,
    pub data_field_min_max: TVec<(isize, isize)>,
    pub standard_layout_data_field: Vec<isize>,
    pub op_strides_times_input_storage_strides: TVec<isize>,
    pub valid_output_zone: TVec<Range<usize>>,
    pub invalid_output_zones: TVec<TVec<Range<usize>>>,
    pub zones: Vec<Zone>,
    pub valid_zone_id: Option<usize>,
    pub zone_strides: TVec<isize>,
    pub input_storage_strides: TVec<isize>,
    pub output_storage_strides: TVec<isize>,
}

impl Debug for Patch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.spec)
    }
}

impl Patch {
    #[inline]
    pub fn rank(&self) -> usize {
        self.spec.input_shape.len()
    }

    unsafe fn is_valid(&self, coords: &[usize]) -> bool {
        for ix in 0..self.rank() {
            let c = *coords.get_unchecked(ix) as isize;
            let strides = *self.spec.strides.get_unchecked(ix) as isize;
            let pos = c * strides;
            let min_max = self.data_field_min_max.get_unchecked(ix);
            if pos + min_max.0 < 0
                || pos + min_max.1 >= *self.spec.input_shape.get_unchecked(ix) as isize
            {
                return false;
            }
        }
        true
    }

    pub fn valid_zone(&self) -> Option<&Zone> {
        self.valid_zone_id.map(|id| &self.zones[id])
    }

    #[inline]
    pub fn visit_output(&self, mut acceptor: impl FnMut(&Scanner)) {
        if self.zones.len() == 0 {
            return;
        }
        let mut scanner = Scanner::new(self);
        while !scanner.done() {
            acceptor(&scanner);
            scanner.next();
        }
    }

    pub fn centers_offsets(&self) -> Vec<isize> {
        if self.zones.len() == 0 {
            return vec![];
        }
        let mut scanner = Scanner::new(self);
        let len = self.output_shape.iter().cloned().product();
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(scanner.input_center_offset);
            scanner.next()
        }
        v
    }

    pub fn at<'p>(&'p self, coords: &[usize]) -> PatchIterator<'p> {
        self.at_hint(coords, None)
    }

    pub fn at_hint<'p>(&'p self, coords: &[usize], hint: Option<bool>) -> PatchIterator<'p> {
        unsafe {
            assert_eq!(coords.len(), self.spec.kernel_shape.len());
            let mut center = 0;
            for i in 0..self.op_strides_times_input_storage_strides.len() {
                center += *self.op_strides_times_input_storage_strides.get_unchecked(i)
                    * *coords.get_unchecked(i) as isize;
            }
            let valid = hint.unwrap_or_else(|| !self.padded || self.is_valid(coords));
            if valid {
                PatchIterator::Fast(FastPatchIterator { patch: self, center, item: 0 })
            } else {
                let mut input_patch_center: TVec<_> = coords.into();
                input_patch_center
                    .iter_mut()
                    .zip(self.spec.strides.iter())
                    .for_each(|(a, &b)| *a *= b);
                PatchIterator::Safe(SafePatchIterator {
                    patch: self,
                    item: 0,
                    input_patch_center,
                    center,
                })
            }
        }
    }

    pub fn global_offset_for(&self, coords: &[usize], patch_index: usize) -> usize {
        assert_eq!(coords.len(), self.spec.kernel_shape.len());
        let center = izip!(coords, &self.op_strides_times_input_storage_strides)
            .map(|(a, b)| *a as isize * *b)
            .sum::<isize>();
        (center + self.standard_layout_data_field[patch_index]) as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Zone {
    pub valid: bool,
    pub input_zone_offset: isize,
    pub output_zone_offset: isize,
    pub output_ranges: Box<[Range<usize>]>,
    pub output_shape: Box<[usize]>,
    /// (index in kernel, offset from center in image)
    pub values_offsets: Box<[(usize, isize)]>,
}

impl Zone {
    pub fn contains_output(&self, coords: &[usize]) -> bool {
        self.output_ranges.iter().zip(coords).all(|(range, &x)| x >= range.start && x < range.end)
    }

    #[inline]
    pub fn visit_output(&self, patch: &Patch, mut acceptor: impl FnMut(&ZoneScanner)) {
        let mut scanner = ZoneScanner::new(self, patch);
        while !scanner.done() {
            acceptor(&scanner);
            scanner.next();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZoneScanner<'p> {
    pub patch: &'p Patch,
    pub zone: &'p Zone,
    pub output_offset: isize,
    pub output_coords: Box<[usize]>,
    pub input_center_offset: isize,
    pub inner_loop_axis: usize,
    pub inner_loop_len: usize,
    pub inner_loop_output_range: Range<usize>,
    pub inner_loop_output_stride: isize,
    pub inner_loop_input_full_stride: isize,
    pub done: bool,
}

impl<'p> ZoneScanner<'p> {
    pub fn new(zone: &'p Zone, patch: &'p Patch) -> ZoneScanner<'p> {
        let inner_loop_axis =
            zone.output_shape.iter().enumerate().max_by_key(|(_, dim)| *dim).unwrap().0;
        let inner_loop_output_range = zone.output_ranges[inner_loop_axis].clone();
        let inner_loop_output_stride = patch.output_storage_strides[inner_loop_axis];
        let inner_loop_input_full_stride =
            patch.op_strides_times_input_storage_strides[inner_loop_axis];
        let mut scan = ZoneScanner {
            patch,
            zone,
            output_offset: 0,
            input_center_offset: 0,
            inner_loop_axis,
            inner_loop_len: inner_loop_output_range.len(),
            inner_loop_output_range,
            inner_loop_output_stride,
            inner_loop_input_full_stride,
            output_coords: zone.output_ranges.iter().map(|r| r.start).collect(),
            done: false,
        };
        scan.refresh_dependent();
        scan
    }

    #[inline]
    pub fn valid_offsets_ker_in(&self) -> impl Iterator<Item = (usize, isize)> + '_ {
        self.zone.values_offsets.iter().map(move |pair| (pair.0, pair.1 + self.input_center_offset))
    }

    pub unsafe fn next_non_inner_axis(&mut self) {
        let rank = self.patch.rank();
        let inner_loop_axis = self.inner_loop_axis;
        for axis in (0..rank).rev() {
            if axis == inner_loop_axis {
                continue;
            }
            *self.output_coords.get_unchecked_mut(axis) += 1;
            if *self.output_coords.get_unchecked_mut(axis)
                < self.zone.output_ranges.get_unchecked(axis).end
            {
                self.refresh_dependent();
                return;
            }
            *self.output_coords.get_unchecked_mut(axis) =
                self.zone.output_ranges.get_unchecked(axis).start;
        }
        self.done = true;
    }

    pub unsafe fn reset(&mut self) {
        self.output_offset = 0;
        self.input_center_offset = 0;
        for ix in 0..self.output_coords.len() {
            *self.output_coords.get_unchecked_mut(ix) =
                self.zone.output_ranges.get_unchecked(ix).start;
        }
        self.done = false;
        self.refresh_dependent()
    }

    #[inline(never)]
    fn refresh_dependent(&mut self) {
        self.input_center_offset = self
            .patch
            .op_strides_times_input_storage_strides
            .iter()
            .zip(self.output_coords.iter())
            .map(|(a, b)| *a * *b as isize)
            .sum();
        self.output_offset = self
            .patch
            .output_storage_strides
            .iter()
            .zip(self.output_coords.iter())
            .map(|(a, b)| a * *b as isize)
            .sum();
    }

    #[inline]
    pub fn next(&mut self) {
        let inner_loop_axis = self.inner_loop_axis;
        unsafe {
            *self.output_coords.get_unchecked_mut(inner_loop_axis) += 1;
            if *self.output_coords.get_unchecked(inner_loop_axis) < self.inner_loop_output_range.end
            {
                self.input_center_offset += self.inner_loop_input_full_stride;
                self.output_offset += self.inner_loop_output_stride;
            } else {
                *self.output_coords.get_unchecked_mut(inner_loop_axis) =
                    self.inner_loop_output_range.start;
                self.next_non_inner_axis();
            }
        }
    }

    pub fn done(&self) -> bool {
        self.done
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scanner<'p> {
    pub patch: &'p Patch,
    pub zone_id: usize,
    pub zone_coords: TVec<usize>,
    pub zone: &'p Zone,
    pub output_offset: isize,
    pub output_coords: TVec<usize>,
    pub input_coords: TVec<usize>,
    pub input_center_offset: isize,
    done: bool,
}

impl<'p> Scanner<'p> {
    fn new(patch: &'p Patch) -> Scanner<'p> {
        let rank = patch.rank();
        let zone = &patch.zones[0];
        Scanner {
            patch,
            zone_coords: tvec!(0; rank),
            zone,
            zone_id: 0,
            output_offset: 0,
            input_center_offset: 0,
            input_coords: tvec!(0; rank),
            output_coords: tvec!(0; rank),
            done: false,
        }
    }

    #[inline]
    pub fn valid_count(&self) -> usize {
        self.zone.values_offsets.len()
    }

    #[inline]
    pub fn valid_offsets(&self) -> impl Iterator<Item = isize> + '_ {
        self.zone.values_offsets.iter().map(move |pair| pair.1 + self.input_center_offset)
    }

    #[inline]
    pub fn valid_offsets_ker_in(&self) -> impl Iterator<Item = (usize, isize)> + '_ {
        self.zone.values_offsets.iter().map(move |pair| (pair.0, pair.1 + self.input_center_offset))
    }

    #[inline]
    pub fn next(&mut self) {
        let rank = self.patch.rank();
        let inner_dim = rank - 1;
        unsafe {
            *self.output_coords.get_unchecked_mut(inner_dim) += 1;
            *self.input_coords.get_unchecked_mut(inner_dim) +=
                *self.patch.spec.strides.get_unchecked(inner_dim);
            self.output_offset += self.patch.spec.output_inner_stride as isize;
            self.input_center_offset +=
                self.patch.op_strides_times_input_storage_strides.get_unchecked(inner_dim);
            if *self.output_coords.get_unchecked(inner_dim)
                < self.zone.output_ranges.get_unchecked(inner_dim).end
            {
                return;
            }
            if self.output_coords.get_unchecked(inner_dim)
                < self.patch.output_shape.get_unchecked(inner_dim)
            {
                self.zone_id += 1;
                *self.zone_coords.get_unchecked_mut(inner_dim) += 1;
                self.zone = self.patch.zones.get_unchecked(self.zone_id);
            } else {
                for axis in (0..rank - 1).rev() {
                    *self.output_coords.get_unchecked_mut(axis + 1) = 0;
                    *self.input_coords.get_unchecked_mut(axis + 1) = 0;
                    *self.output_coords.get_unchecked_mut(axis) += 1;
                    *self.input_coords.get_unchecked_mut(axis) +=
                        self.patch.spec.strides.get_unchecked(axis);
                    *self.zone_coords.get_unchecked_mut(axis + 1) = 0;
                    if *self.output_coords.get_unchecked(axis)
                        == self.zone.output_ranges.get_unchecked(axis).end
                    {
                        *self.zone_coords.get_unchecked_mut(axis) += 1;
                    }
                    if *self.output_coords.get_unchecked(axis)
                        < *self.patch.output_shape.get_unchecked(axis)
                    {
                        break;
                    }
                }
                if self.output_coords.get_unchecked(0) == self.patch.output_shape.get_unchecked(0) {
                    self.done = true;
                    return;
                }
                self.zone_id = 0;
                self.input_center_offset = 0;
                for i in 0..rank {
                    self.zone_id += *self.zone_coords.get_unchecked(i)
                        * *self.patch.zone_strides.get_unchecked(i) as usize;
                    self.input_center_offset += *self.input_coords.get_unchecked(i) as isize
                        * *self.patch.input_storage_strides.get_unchecked(i);
                }
                self.zone = self.patch.zones.get_unchecked(self.zone_id);
            }
        }
    }

    pub fn done(&self) -> bool {
        self.done
    }
}

#[derive(Debug)]
pub enum PatchIterator<'p> {
    Fast(FastPatchIterator<'p>),
    Safe(SafePatchIterator<'p>),
}

impl<'p> Iterator for PatchIterator<'p> {
    type Item = Option<isize>;
    #[inline(always)]
    fn next(&mut self) -> Option<Option<isize>> {
        match self {
            PatchIterator::Fast(ref mut it) => it.next(),
            PatchIterator::Safe(ref mut it) => it.next(),
        }
    }
}

#[derive(Debug)]
pub struct FastPatchIterator<'p> {
    patch: &'p Patch,
    center: isize,
    item: usize,
}

impl<'p> Iterator for FastPatchIterator<'p> {
    type Item = Option<isize>;
    #[inline(always)]
    fn next(&mut self) -> Option<Option<isize>> {
        if self.item == self.patch.standard_layout_data_field.len() {
            return None;
        }
        unsafe {
            let position =
                self.center + self.patch.standard_layout_data_field.get_unchecked(self.item);
            self.item += 1;
            Some(Some(position))
        }
    }
}

#[derive(Debug)]
pub struct SafePatchIterator<'p> {
    patch: &'p Patch,
    item: usize,
    input_patch_center: TVec<usize>,
    center: isize,
}

impl<'p> Iterator for SafePatchIterator<'p> {
    type Item = Option<isize>;
    fn next(&mut self) -> Option<Option<isize>> {
        unsafe {
            if self.item == self.patch.standard_layout_data_field.len() {
                return None;
            }
            let input_shape = &self.patch.spec.input_shape;
            let img_offset = self.patch.data_field.as_ptr().add(self.item * input_shape.len());

            for ix in 0..input_shape.len() {
                let pos = *self.input_patch_center.get_unchecked(ix) as isize + *img_offset.add(ix);
                if pos < 0 || pos as usize >= *input_shape.get_unchecked(ix) {
                    self.item += 1;
                    return Some(None);
                }
            }
            let position =
                self.center + self.patch.standard_layout_data_field.get_unchecked(self.item);
            self.item += 1;
            Some(Some(position))
        }
    }
}

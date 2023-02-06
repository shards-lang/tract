use crate::internal::*;

use tract_itertools::Itertools;
use std::ops::Range;

#[derive(Clone, Debug, new, PartialEq, Eq)]
pub struct Region {
    pub range: Range<usize>,
    pub mask: Option<TVec<bool>>,
}

#[derive(Clone, Debug, new, PartialEq, Eq)]
pub struct PatchAxis {
    pub input_dim: usize,
    pub kernel_dim: usize,
    pub pad_before: usize,
    pub pad_after: usize,
    pub output_dim: usize,
    pub stride: usize,
    pub dilation: usize,
}

impl PatchAxis {
    fn valid_range(&self) -> Option<Range<usize>> {
        let field = (self.kernel_dim - 1) * self.dilation + 1;
        if field > self.input_dim {
            return None;
        }
        let min = self.pad_before.divceil(self.stride);
        let max = (self.input_dim + self.pad_before).saturating_sub(field) / self.stride;
        if max >= min {
            Some(min..(max + 1))
        } else {
            None
        }
    }

    fn invalid_at_left(&self, pos: usize) -> usize {
        let center_pos = pos * self.stride;
        self.pad_before.saturating_sub(center_pos).divceil(self.dilation)
    }

    fn invalid_at_right(&self, pos: usize) -> usize {
        let center_pos = pos * self.stride;
        let last_valid = self.input_dim + self.pad_before;
        let valid = (last_valid - center_pos).divceil(self.dilation);
        self.kernel_dim.saturating_sub(valid)
    }

    fn make_invalid_regions(&self, range: Range<usize>) -> TVec<Region> {
        range
            .map(move |ix| (ix, (self.invalid_at_left(ix), self.invalid_at_right(ix))))
            .group_by(|&pair| pair.1)
            .into_iter()
            .map(move |(invalid, pairs)| {
                let (min, max) = pairs.map(|p| p.0).minmax().into_option().unwrap();
                let mut mask = tvec!(false; self.kernel_dim);
                for i in 0..invalid.0 {
                    mask[i] = true;
                }
                for i in 0..invalid.1 {
                    mask[self.kernel_dim - 1 - i] = true;
                }
                Region::new(min..max + 1, Some(mask))
            })
            .collect()
    }

    pub fn regions(&self) -> TVec<Region> {
        let mut regions = tvec!();
        if let Some(valid_range) = self.valid_range() {
            if valid_range.start > 0 {
                regions.extend(self.make_invalid_regions(0..valid_range.start));
            }
            if valid_range.start != valid_range.end {
                regions.push(Region::new(valid_range.clone(), None));
            }
            if valid_range.end < self.output_dim {
                regions.extend(self.make_invalid_regions(valid_range.end..self.output_dim));
            }
        } else {
            regions.extend(self.make_invalid_regions(0..self.output_dim));
        }
        regions
    }
}


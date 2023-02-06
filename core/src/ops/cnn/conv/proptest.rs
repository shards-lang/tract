use super::KernelFormat;
use crate::ops::cnn::*;
use crate::ops::nn::*;
use crate::setup_test_logger;
use tract_itertools::izip;
use tract_ndarray::prelude::*;

#[derive(Debug)]
struct ConvProblem {
    shape_in: DataShape,
    kernel_format: KernelFormat,
    group: usize,
    data: ArrayD<f32>,
    kernel: ArrayD<f32>,
    bias: Option<ArrayD<f32>>,
    pad: PaddingSpec,
    strides: TVec<usize>,
}

impl ConvProblem {
    fn geo_ker(&self) -> &[usize] {
        &self.kernel.shape()[self.kernel_format.h_axis()..][..self.shape_in.hw_rank()]
    }

    fn reference(&self) -> ArrayD<f32> {
        setup_test_logger();
        assert_eq!(self.data.shape(), &*self.shape_in.shape, "inconsistent shapes in test");
        let n = *self.shape_in.n().unwrap_or(&1);
        let ci_per_g = self.shape_in.c() / self.group;
        let co_per_g = match self.kernel_format {
            KernelFormat::OIHW => self.kernel.shape()[0] / self.group,
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.ndim() - 1],
        };
        let (shape_out, left_pads): (TVec<_>, TVec<_>) = if self.pad == PaddingSpec::Valid {
            izip!(self.shape_in.hw_dims(), self.geo_ker(), &self.strides)
                .map(|(i, k, s)| {
                    let out = (*i + 1).saturating_sub(*k).divceil(*s);
                    (out, 0)
                })
                .unzip()
        } else {
            izip!(self.shape_in.hw_dims(), self.geo_ker(), &self.strides)
                .map(|(input, k, stride)| {
                    let out = input.divceil(*stride);
                    let pad = ((out - 1) * stride + k).saturating_sub(*input);
                    (out, pad / 2)
                })
                .unzip()
        };
        let shape_out = self
            .shape_in
            .fmt
            .from_n_c_hw(self.shape_in.n().cloned().unwrap_or(1), co_per_g * self.group, shape_out)
            .unwrap();
        let mut out = ArrayD::zeros(&*shape_out.shape);
        for n in 0..n {
            for g in 0..self.group {
                for geo_out in tract_ndarray::indices(shape_out.hw_dims()) {
                    let mut output_coords: TVec<usize> = geo_out.slice().into();
                    if self.shape_in.fmt.has_n() {
                        output_coords.insert(0, n);
                    }
                    output_coords.insert(shape_out.c_axis(), 0);
                    for geo_ker in tract_ndarray::indices(self.geo_ker()) {
                        let input_coords: TVec<isize> =
                            izip!(geo_out.slice(), geo_ker.slice(), &left_pads, &self.strides)
                                .map(|(out, ker, pad, stride)| {
                                    *out as isize * *stride as isize + *ker as isize - *pad as isize
                                })
                                .collect();
                        if izip!(&input_coords, self.shape_in.hw_dims())
                            .any(|(c, i)| *c < 0 || *c >= *i as isize)
                        {
                            continue;
                        }
                        let mut input_coords: TVec<usize> =
                            input_coords.into_iter().map(|d| d as usize).collect();
                        if self.shape_in.fmt.has_n() {
                            input_coords.insert(0, n);
                        }
                        input_coords.insert(self.shape_in.c_axis(), 0);
                        for ci in 0..ci_per_g {
                            input_coords[self.shape_in.c_axis()] = ci + g * ci_per_g;
                            let i = self.data[&*input_coords];
                            for co in 0..co_per_g {
                                output_coords[shape_out.c_axis()] = co + g * co_per_g;
                                let mut kernel_coords: TVec<usize> = geo_ker.slice().into();
                                match self.kernel_format {
                                    KernelFormat::OIHW => {
                                        kernel_coords.insert(0, ci);
                                        kernel_coords.insert(0, co + g * co_per_g);
                                    }
                                    KernelFormat::HWIO => {
                                        kernel_coords.push(ci + g * ci_per_g);
                                        kernel_coords.push(co);
                                    }
                                }
                                let k = self.kernel[&*kernel_coords];
                                out[&*output_coords] += k * i;
                            }
                        }
                    }
                }
            }
        }
        if let Some(bias) = &self.bias {
            let mut shape = vec![1; out.ndim()];
            shape[shape_out.c_axis()] = bias.len();
            out += &bias.clone().into_shape(shape).unwrap();
        }
        out
    }

    fn tract(&self) -> anyhow::Result<ArrayD<f32>> {
        setup_test_logger();
        assert_eq!(self.data.shape(), &*self.shape_in.shape, "inconsistent shapes in test");
        let mut model = TypedModel::default();
        let wire = model.add_source("input", f32::fact(&self.shape_in.shape))?;
        let co = match self.kernel_format {
            KernelFormat::OIHW => self.kernel.shape()[0],
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.ndim() - 1] * self.group,
        };
        let op = ConvUnary::new(
            PoolSpec::new(
                self.shape_in.fmt,
                self.geo_ker().into(),
                self.pad.clone(),
                None,
                Some(self.strides.clone()),
                Some(co),
            ),
            self.kernel_format,
            self.kernel.clone().into_arc_tensor(),
            self.group,
            self.bias.clone().map(|a| a.into_arc_tensor()),
//            None,
        );
        let wire = model.wire_node("conv", op, &[wire])?[0];
        model.set_output_outlets(&[wire])?;
//dbg!(&model);
        let decluttered = model.into_decluttered()?;
//        let optimized = model.into_optimized()?;
//dbg!(&decluttered);
        let optimized = decluttered.into_optimized()?;
//dbg!(&optimized);
//.into_runnable()?; //.run(tvec![self.data.clone().into_tvalue()])?;
        // output.remove(0).into_tensor().into_array::<f32>()
	Ok(arr2(&[[0f32, 0f32]]).into_dyn())
    }
}

#[test]
fn crasher_monterey() -> anyhow::Result<()> {
    let pb = ConvProblem {
        shape_in: DataFormat::HWC.from_n_c_hw(1, 1, [1])?,
        kernel_format: KernelFormat::OIHW,
        group: 1,
        data: tract_ndarray::arr2(&[[0.0f32]]).into_dyn(),
        kernel: tract_ndarray::arr3(&[[[0.0f32]], [[0.0f32]]]).into_dyn(),
        bias: None,
        pad: PaddingSpec::Valid,
        strides: tvec!(1),
    };
    assert_eq!(pb.tract().unwrap(), pb.reference());
    Ok(())
}

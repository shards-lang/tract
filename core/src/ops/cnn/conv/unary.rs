use ndarray::*;
use tract_data::itertools::izip;

use crate::internal::*;
use crate::model::*;
use crate::ops;
use crate::ops::matmul::MatMulAxes;

use super::im2col::Im2Col;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry, PoolSpec};
use crate::ops::matmul::lir_unary::{
    ConcreteMatMulGeometry, LirMatMulUnary, MatMulGeometry, ProtoFusedSpec, SymbolicMatMulGeometry,
};
//use crate::ops::matmul::MatMulQParams;
use crate::ops::nn::{BaseDataShape, DataFormat, DataShape};

use tract_linalg::frame::Packer;
use tract_linalg::mmm::MatMatMul;

use std::iter::Sum;

#[derive(Debug, Clone, new, Hash)]
pub struct ConvUnary {
    pub pool_spec: PoolSpec,
    pub kernel_fmt: KernelFormat,
    pub kernel: Arc<Tensor>,

    pub group: usize,

    pub bias: Option<Arc<Tensor>>,

//    pub q_params: Option<(DatumType, MatMulQParams)>,
}

impl_dyn_hash!(ConvUnary);

impl ConvUnary {
    fn input_channels(&self) -> usize {
        match self.kernel_fmt {
            KernelFormat::OIHW => self.kernel.shape()[1] * self.group,
            KernelFormat::HWIO => self.kernel.shape()[self.kernel.shape().len() - 2],
        }
    }

    fn output_channels(&self) -> usize {
        let kshape = self.kernel.shape();
        match self.kernel_fmt {
            KernelFormat::OIHW => kshape[0],
            KernelFormat::HWIO => kshape[kshape.len() - 1] * self.group,
        }
    }

    pub fn kernel_as_group_o_ihw(&self) -> TractResult<Arc<Tensor>> {
        self.kernel_fmt.kernel_as_group_o_ihw(
            &self.kernel,
            self.group,
            self.input_channels(),
            self.output_channels(),
        )
    }

    fn kernel_as_packed_as(
        &self,
        packer: &Packer,
        k: usize,
        m: usize,
    ) -> TractResult<ArrayD<Arc<Tensor>>> {
        let kernel = self.kernel_as_group_o_ihw()?;
        unsafe {
            let mut packed_as = Array1::from(
                (0..self.group)
                    .map(|g| {
                        let packed = Tensor::uninitialized_aligned_dt(
                            kernel.datum_type(),
                            &[packer.len(k, m)],
                            packer.alignment(),
                        )?;
                        packer.pack(
                            &mut TensorView::at_prefix(&packed, &[])?,
                            &kernel.view_at_prefix(&[g])?,
                            1,
                            0,
                        );
                        Ok(packed.into_arc_tensor())
                    })
                    .collect::<TractResult<Vec<_>>>()?,
            )
            .into_dyn();
            if self.group == 1 {
                packed_as.index_axis_inplace(Axis(0), 0);
            }
            if self.pool_spec.data_format.has_n() {
                packed_as.insert_axis_inplace(Axis(0));
            }
            packed_as.insert_axis_inplace(Axis(packed_as.ndim()));
            packed_as.insert_axis_inplace(Axis(packed_as.ndim()));
            Ok(packed_as)
        }
    }

/*
    fn kernel_offset_u8_as_i8(
        &self,
        inputs: &mut [OutletId],
        model: &mut TypedModel,
    ) -> TractResult<Option<Self>> {
        if let DatumType::U8 = self.kernel.datum_type().unquantized() {
            let new_op = Self {
                kernel: self.kernel.offset_u8_as_i8(),
                q_params: self
                    .q_params
                    .as_ref()
                    .map(|(dt, qp)| -> TractResult<_> {
                        let a0 = match &qp.a0 {
                            QParamKind::Attr(_) | QParamKind::FromQType => {
                                qp.a0.offset_u8_as_i8(model, &[])?
                            }
                            QParamKind::FromInput(i) => {
                                match model.outlet_fact(inputs[*i])?.datum_type.unquantized() {
                                    DatumType::U8 => {
                                        inputs[*i] = model.wire_node(
                                            format!(
                                                "{}.offset_{}_as_i8",
                                                model.node(inputs[*i].node).name,
                                                "a0"
                                            ),
                                            ops::quant::offset_u8_as_i8(),
                                            &[inputs[*i]],
                                        )?[0];
                                    }
                                    DatumType::I32 => {
                                        let cst = model.add_const(
                                            format!(
                                                "{}.offset_{}_as_i8.cst",
                                                &model.node(inputs[*i].node).name,
                                                "a0"
                                            ),
                                            rctensor0(-128i32),
                                        )?;
                                        inputs[*i] = model.wire_node(
                                            format!(
                                                "{}.offset_{}_as_i8",
                                                model.node(inputs[*i].node).name,
                                                "a0"
                                            ),
                                            ops::math::add(),
                                            &[inputs[*i], cst],
                                        )?[0];
                                    }
                                    _ => (),
                                }
                                QParamKind::FromInput(*i)
                            }
                        };
                        Ok((*dt, MatMulQParams { a0, ..qp.clone() }))
                    })
                    .transpose()?,
                ..self.clone()
            };
            Ok(Some(new_op))
        } else {
            Ok(None)
        }
    }
*/

    fn bias_as_non_linear<T>(&self) -> TractResult<ArrayD<Vec<ProtoFusedSpec>>>
    where
        T: Datum + Copy,
    {
        let mut ops = Array1::from_elem(self.group, vec![]);

        if let Some(bias) = &self.bias {
            let bias = bias.cast_to::<T>()?;
            let bias = bias.as_slice::<T>()?;
            ops.iter_mut().zip(bias.chunks(self.output_channels() / self.group)).for_each(
                |(ops, bias)| {
                    ops.push(ProtoFusedSpec::BinPerRow(
                        rctensor1(bias).into(),
                        tract_linalg::mmm::BinOp::Add,
                    ));
                },
            )
        }
        let mut ops = ops.into_dyn();

        if self.group == 1 {
            ops.index_axis_inplace(Axis(0), 0);
        }
        if self.pool_spec.data_format.has_n() {
            ops.insert_axis_inplace(Axis(0));
        }
        Ok(ops)
    }

/*
    pub unsafe fn wire_as_quant_im2col(
        &self,
        model: &mut TypedModel,
        name: &str,
        b_dt: DatumType,
        wires: &[OutletId],
    ) -> TractResult<OutletId> {
        use crate::ops::matmul::mir_quant as qmm;

        let c_dt = self.q_params.as_ref().unwrap().0;

        let params = self.q_params.as_ref().unwrap().1.as_outlet_ids(
            model,
            name,
            wires,
            self.kernel.datum_type(),
            b_dt,
            c_dt,
        )?;

        let a0 = params[0];
        let a_scale = params[1];
        let mut b0 = params[2];
        let b_scale = params[3];
        let c0 = params[4];
        let c_scale = params[5];

        let b = wire_offset_u8_as_i8(model, name, wires[0], "b", &mut b0, "b0")?;
        let b_fact = model.outlet_fact(b)?.clone();
        let (_, m, k, n, mmm) = self.compute_geo(&b_fact)?;
        let output_shape = self.pool_spec.output_shape(&b_fact.shape)?;

        let abc_scale = qmm::combine_scales(model, name, a_scale, b_scale, c_scale)?;

        let im2col = model.wire_node(
            format!("{name}.im2col"),
            Im2Col::new(self.pool_spec.clone(), self.group, k, &b_fact.shape, mmm.clone())?,
            &[b, b0],
        )?[0];

        let a = self.kernel_as_group_o_ihw()?.into_tensor();
        let a = a.cast_to_dt(i32::datum_type())?;
        let a = a.to_array_view::<i32>()?;
        let mut sum_a = a.sum_axis(Axis(a.ndim() - 1));
        if self.group == 1 {
            sum_a.index_axis_inplace(Axis(0), 0);
        }

        if self.pool_spec.data_format.has_n() {
            sum_a.insert_axis_inplace(Axis(0));
        }
        let sum_a = model.add_const(format!("{name}.sum_a"), sum_a)?;

        let mut sum_b = model.wire_node(
            format!("{name}.sum_b"),
            super::QSumB { n: n.clone(), r: mmm.b_pack().panel_width(), k },
            &[im2col],
        )?[0];

        if self.group > 1 && self.pool_spec.data_format.c_is_last() {
            let has_n = self.pool_spec.data_format.has_n() as usize;
            sum_b = model.wire_node(
                format!("{name}.transpose_sum_b"),
                AxisOp::Move(has_n, 1 + has_n),
                &[sum_b],
            )?[0];
        }

        let b_dt = model.outlet_fact(b)?.datum_type;
        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&output_shape)?;
        let mut geometry = MatMulGeometry::from(SymbolicMatMulGeometry {
            b_datum_type: b_dt,
            m: m.to_dim(),
            k: k.to_dim(),
            n: n.clone(),
            mmm: mmm.clone(),
        });
        if n.to_usize().is_ok() {
            geometry = geometry.optimize_if(Some(&SymbolValues::default()))?;
        }
        let wire = self.wire_lir_matmatmul(
            model,
            name,
            im2col,
            mmm,
            i32::datum_type(),
            mmm_output_shape.clone().into(),
            m,
            k,
            geometry,
            c_axis,
            h_axis,
        )?;
        let has_n = self.pool_spec.data_format.has_n() as usize;
        let has_group = (self.group > 1) as usize;
        let (m_axis, n_axis) = if self.pool_spec.data_format.c_is_last() {
            (1 + has_group + has_n, has_n)
        } else {
            (has_group + has_n, 1 + has_n + has_group)
        };
        let wire = qmm::compensate_zero_points(
            model,
            name,
            wire,
            k.to_dim(),
            a0,
            b0,
            sum_a,
            sum_b,
            m_axis,
            n_axis,
        )?;

        let mut wire = qmm::requant(model, name, wire, c_dt, abc_scale, c0)?;
        if self.group > 1 {
            wire = model.wire_node(
                format!("{name}.reshape_group"),
                AxisOp::Reshape(
                    c_axis - 1,
                    mmm_output_shape[c_axis - 1..][..2].iter().map(|d| d.to_dim()).collect(),
                    tvec!((m * self.group).to_dim()),
                ),
                &[wire],
            )?[0];
        }
        let wire = Self::wire_geo_reshape(model, name, wire, &output_shape)?;
        Ok(wire)
    }
*/

    pub unsafe fn wire_as_im2col_pair(
        &self,
        model: &mut TypedModel,
        name: &str,
        mut wire: OutletId,
    ) -> TractResult<OutletId> {
        let b_fact = model.outlet_fact(wire)?.clone();
        let b_dt = b_fact.datum_type;
        let c_dt = crate::ops::matmul::output_type(b_fact.datum_type);

        let output_shape = self.pool_spec.output_shape(&b_fact.shape)?;
        let (_, m, k, n, mmm) = self.compute_geo(model.outlet_fact(wire)?)?;
        let padding = model.add_const(format!("{name}.b0"), Tensor::zero_dt(b_dt, &[])?)?;

        wire = model.wire_node(
            format!("{name}.im2col"),
            Im2Col::new(self.pool_spec.clone(), self.group, k, &b_fact.shape, mmm.clone())?,
            &[wire, padding],
        )?[0];

        let (mmm_output_shape, c_axis, h_axis) = self.mmm_output_shape(&output_shape)?;
        let mut geometry = MatMulGeometry::from(SymbolicMatMulGeometry {
            b_datum_type: b_dt,
            m: m.to_dim(),
            k: k.to_dim(),
            n: n.clone(),
            mmm: mmm.clone(),
        });
        if n.to_usize().is_ok() {
            geometry = geometry.optimize_if(Some(&SymbolValues::default()))?;
        }
        let mut wire = self.wire_lir_matmatmul(
            model,
            name,
            wire,
            mmm,
            c_dt,
            mmm_output_shape.clone().into(),
            m.to_usize().unwrap(),
            k.to_usize().unwrap(),
            geometry,
            c_axis,
            h_axis,
        )?;

        Ok(wire)
    }

    fn mmm_output_shape<D: DimLike>(
        &self,
        output_shape: &BaseDataShape<D, TVec<D>>,
    ) -> TractResult<(TVec<D>, usize, usize)> {
        let geo_collapsed_out: D = output_shape.hw_dims().iter().cloned().product();
        let shape: BaseDataShape<D, TVec<D>> = output_shape.fmt.from_n_c_hw(
            output_shape.n().cloned().unwrap_or_else(|| 1.into()),
            output_shape.c().clone(),
            tvec!(geo_collapsed_out),
        )?;
        let mut mmm_output_shape: TVec<D> = shape.shape.clone();
        let mut c_axis = shape.c_axis();
        let mut h_axis = shape.h_axis();
        if self.group > 1 {
            mmm_output_shape[shape.c_axis()] =
                mmm_output_shape[shape.c_axis()].clone() / self.group;
            mmm_output_shape.insert(shape.c_axis(), self.group.into());
            if self.group > 1 {
                if h_axis > c_axis {
                    h_axis += 1;
                }
                c_axis += 1;
            }
        }
        Ok((mmm_output_shape, c_axis, h_axis))
    }

    #[allow(clippy::type_complexity)]
    fn compute_geo(
        &self,
        input_fact: &TypedFact,
    ) -> TractResult<(PoolGeometry, usize, usize, TDim, Box<dyn MatMatMul>)> {
        let a_dt = self.kernel.datum_type();
        let b_dt = input_fact.datum_type;
        let c_dt = crate::ops::matmul::output_type(b_dt);

        let geo = self.pool_spec.compute_geo(&input_fact.shape)?;

        trace!("output channels: {:?}", self.output_channels());
        let m = self.output_channels() / self.group;
        let k = self.kernel.len() / self.output_channels();
        let n: TDim =
            self.pool_spec.output_shape(&input_fact.shape)?.hw_dims().iter().cloned().product();

        let mmm = tract_linalg::ops()
            .mmm(a_dt, b_dt, c_dt, Some(m), Some(k), n.to_usize().ok())
            .with_context(|| format!("No multiplier for {a_dt:?}x{b_dt:?} to {c_dt:?}",))?;

        Ok((geo, m, k, n, mmm))
    }

    #[allow(clippy::too_many_arguments)]
    fn wire_lir_matmatmul(
        &self,
        model: &mut TypedModel,
        name: &str,
        wire: OutletId,
        mmm: Box<dyn MatMatMul>,
        c_datum_type: DatumType,
        mmm_output_shape: ShapeFact,
        m: usize,
        k: usize,
        geometry: MatMulGeometry,
        c_m_axis: usize,
        c_n_axis: usize,
    ) -> TractResult<OutletId> {
        let kernels = self.kernel_as_packed_as(&mmm.a_pack(), k, m)?;
        let shape = kernels.shape();
        let mut fused_ops = dispatch_copy!(Self::bias_as_non_linear(mmm.internal_type())(self))?;
        for fo in &mut fused_ops {
            fo.push(ProtoFusedSpec::Store);
        }
        let mut iter = kernels.iter().cloned().zip(fused_ops.iter().cloned());
        let micro_ops = ArrayD::from_shape_fn(shape, |_| iter.next().unwrap());

        let wire = model.wire_node(
            format!("{name}.matmatmul"),
            LirMatMulUnary {
                c_fact: c_datum_type.fact(mmm_output_shape.clone()),
                micro_ops,
                c_m_axis,
                c_n_axis,
                c_final_shape: mmm_output_shape,
//                reshape_post: vec![],
                geometry,
                mmm,
            },
            &[wire],
        )?[0];
        Ok(wire)
    }
/*
    pub fn to_depth_wise<T>(&self, input: &TypedFact) -> TractResult<Box<dyn TypedOp>>
    where
        T: Datum + Clone + ::ndarray::LinalgScalar + PartialEq + Sum,
    {
        let input_shape = input.shape.as_concrete().unwrap();
        let ConcretePoolGeometry { input_shape, patch, output_shape } =
            self.pool_spec.compute_geo(&input.shape)?.to_concrete(input_shape)?.into_owned();
        let bias = if let Some(b) = &self.bias {
            b.clone()
        } else {
            Tensor::zero::<T>(&[*input_shape.c()])?.into_arc_tensor()
        };
        let op = DepthWise::new(
            patch,
            input_shape,
            output_shape,
            self.kernel_as_group_o_ihw().context("in kernel_as_group_o_ihw")?,
            bias,
        );
        Ok(Box::new(op))
    }
*/

/*
    fn declutter_stride_slice_to_downsample(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let spatial_rank = self.kernel.rank() - 2;
        if let Some(axis) = (0..spatial_rank).find(|&ax| {
            self.pool_spec.stride(ax) > 1
                && (self.pool_spec.kernel_shape[ax] == 1
                    || (self.pool_spec.padding.valid_dim(ax, self.pool_spec.stride(ax) == 1)
                        && self.pool_spec.dilation(ax) % self.pool_spec.stride(ax) == 0))
        }) {
            let downsample_factor = self.pool_spec.stride(axis);
            let mut new_op = self.clone();
            if new_op.pool_spec.dilation(axis) > 1 {
                new_op.pool_spec.dilations.as_mut().unwrap()[axis] /= downsample_factor;
            }
            new_op.pool_spec.strides.as_mut().unwrap()[axis] /= downsample_factor;
            let mut patch = TypedModelPatch::default();
            let tap = patch.tap_model(model, node.inputs[0])?;
            let shape = self
                .pool_spec
                .data_format
                .shape(input_fact.shape.iter().collect::<TVec<TDim>>())?;
            let down = patch.wire_node(
                format!("{}.downsample.{}", node.name, axis),
                crate::ops::Downsample::new(axis + shape.h_axis(), downsample_factor as isize, 0),
                &[tap],
            )?;
            let id = patch.wire_node(&*node.name, new_op, &down)?[0];
            patch.shunt_outside(model, OutletId::new(node.id, 0), id)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }
*/

    fn declutter_as_matmul(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops::matmul::*;
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let full_input_shape = input_fact.shape.to_tvec();
        let input_shape = self.pool_spec.data_format.shape(&full_input_shape)?;
        if input_shape.hw_rank() == 1
            && self.group == 1
            && self.pool_spec.stride(0) == 1
            && self.kernel.len() == self.input_channels() * self.output_channels()
        {
            let ci = self.input_channels();
            let co = self.output_channels();
            let ker = self.kernel.clone().into_tensor();
            let (a_shape, a_trans) = if self.kernel_fmt == KernelFormat::HWIO {
                ([ci, co], true)
            } else {
                ([co, ci], false)
            };
            let a = ker
                .into_shape(&a_shape)?
                .broadcast_into_rank(full_input_shape.len())?
                .into_arc_tensor();
            let trans_data = self.pool_spec.data_format == DataFormat::HWC
                || self.pool_spec.data_format == DataFormat::NHWC;
            let mut patch = TypedModelPatch::new("declutter_as_matmul");
            let a = patch.add_const(format!("{}.filters", &node.name), a)?;
            let mut inputs = node
                .inputs
                .iter()
                .map(|i| patch.tap_model(model, *i))
                .collect::<TractResult<TVec<_>>>()?;
            inputs.insert(0, a);
            let axes = MatMulAxes::default_for_rank(full_input_shape.len())
                .transposing(a_trans, trans_data, trans_data);
            // in Q case, the bias has to be injected inside the QMatMul (as it
            // must be added before requantization)
/*
            let wire = if let Some(q_params) = &self.q_params {
                let mut params = q_params.1.clone();
                params.insert_input(0); // kernel as input
                params.insert_input(2); // bias as input
                let bias = self.bias.clone().unwrap_or_else(|| rctensor0(0i32));
                anyhow::ensure!(bias.rank() == 0 || bias.rank() == 1);
                let bias = patch.add_const(format!("{}.bias", &node.name), bias)?;
                inputs.insert(2, bias);
                let op = QMatMul { axes, output_type: q_params.0, params: q_params.1.clone() };
                patch.wire_node(&*node.name, op, &inputs)?[0]
            } else {
*/
                let op = MatMul { axes };
                let mut wire = patch.wire_node(format!("{}.matmul", node.name), op, &inputs)?[0];
/*
                if let Some(b) = self.bias.as_ref().filter(|_| self.q_params.is_none()) {
                    anyhow::ensure!(b.rank() == 0 || b.rank() == 1);
                    let mut bias_shape = tvec!(1; input_shape.rank());
                    bias_shape[input_shape.c_axis()] = co;
                    let b = b.clone().into_tensor().into_shape(&bias_shape)?;
                    let b =
                        patch.add_const(format!("{}.bias.cst", node.name), b.into_arc_tensor())?;
/*
                    wire = patch.wire_node(
                        format!("{}.bias", node.name),
                        crate::ops::math::add(),
                        &[wire, b],
                    )?[0];
                }
*/
*/
/*
                wire
            };
*/
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

/*
    fn declutter_precursor_padding(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.pool_spec.padding != PaddingSpec::Valid
            && !matches!(self.pool_spec.padding, PaddingSpec::Explicit(_, _, _))
        {
            return Ok(None);
        }
        let prec = model.node(node.inputs[0].node);
        let pad = if let Some(pad) = prec.op_as::<Pad>() { pad } else { return Ok(None) };
        let value = if let PadMode::Constant(c) = &pad.mode {
            c
        } else {
            return Ok(None);
        };
        let shape = self.pool_spec.data_format.shape(&model.outlet_fact(node.inputs[0])?.shape)?;
        if value.cast_to_scalar::<i64>()? != 0
            || (self.pool_spec.data_format.has_n() && pad.pads[0] != (0, 0))
            || pad.pads[shape.c_axis()] != (0, 0)
        {
            return Ok(None);
        }
        let mut before: TVec<usize> = pad.pads[shape.hw_axes()].iter().map(|pair| pair.0).collect();
        let mut after: TVec<usize> = pad.pads[shape.hw_axes()].iter().map(|pair| pair.1).collect();
        if let PaddingSpec::Explicit(bef, aft, false) = &self.pool_spec.padding {
            izip!(&mut before, bef).for_each(|(pad, cv)| *pad += cv);
            izip!(&mut after, aft).for_each(|(pad, cv)| *pad += cv);
        }
        let padding = PaddingSpec::Explicit(before, after, false);
        let mut new = self.clone();
        new.pool_spec.padding = padding;
        let mut patch = TypedModelPatch::default();
        let wire = patch.tap_model(model, prec.inputs[0])?;
        let wire = patch.wire_node(&node.name, new, &[wire])?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        Ok(Some(patch))
    }
*/
}

impl Op for ConvUnary {
    fn name(&self) -> Cow<str> {
        "ConvUnary".into()
    }

    op_as_typed_op!();
}

impl EvalOp for ConvUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
panic!()
/*
        let mut model = TypedModel::default();

        let mut wires: TVec<OutletId> = inputs
            .iter()
            .enumerate()
            .map(|(ix, v)| {
                model.add_source(format!("source.{ix}"), v.datum_type().fact(v.shape()))
            })
            .collect::<TractResult<_>>()?;
        let wire = unsafe {
                self.wire_as_im2col_pair(&mut model, "im2col-adhoc", wires[0])?
        };
        model.set_output_outlets(&[wire])?;
        model.into_runnable()?.run(inputs)
*/
    }
}

impl TypedOp for ConvUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = self.pool_spec.output_facts(inputs)?.remove(0);
        Ok(tvec!(fact))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
	self.declutter_as_matmul(model, node)
    }

    as_op!();
}


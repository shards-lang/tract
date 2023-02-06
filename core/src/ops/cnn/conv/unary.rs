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
                let op = MatMul { axes };
                let mut wire = patch.wire_node(format!("{}.matmul", node.name), op, &inputs)?[0];
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }
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


use super::lir_unary::{LirMatMulUnary, ProtoFusedSpec};
use super::*;
use crate::internal::*;
// use crate::ops::array::TypedConcat;
use tract_ndarray::prelude::*;

/// The pseudo Unary matrix multiplier. A is constant, B is the input
#[derive(Debug, Clone, new, Hash)]
pub struct MatMulUnary {
    pub a: Arc<Tensor>,
    pub axes: MatMulAxes,
}

impl_dyn_hash!(MatMulUnary);

impl Op for MatMulUnary {
    fn name(&self) -> Cow<str> {
        "MatMulUnary".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MatMulUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        panic!()
    }
}

impl TypedOp for MatMulUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs[0].rank() == self.a.rank(),
            "Inconsistent matmul between input {:?} and attribute {:?} (rank mismatch)",
            inputs[0],
            self.a
        );
        let (_m, _k, _n, c_shape) = compute_shape(
            &self.a.shape().iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            &inputs[0].shape,
            self.axes,
        )?;
        let c_dt = output_type(inputs[0].datum_type);
        Ok(tvec!(c_dt.fact(c_shape)))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let b = args_1!(model.node_input_facts(node.id)?);
        if let Some(b_shape) = b.shape.as_concrete() {
            let patch = self.new_mat_mul_unary_finite(model, node)?;
	    Ok(None)
        } else {
            Ok(None)
        }
    }

    as_op!();
}

impl MatMulUnary {
    fn new_mat_mul_unary_finite(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<TypedModelPatch> {
        let mut patch = TypedModelPatch::default();
        let mut wire = patch.tap_model(model, node.inputs[0])?;

        let packed_as = Array::from_shape_fn(vec![1, 1], |_| {
            let pa = Tensor::zero_aligned::<f32>(&[64], 32).unwrap();
            (pa.into_arc_tensor(), vec![ ProtoFusedSpec::Store, ])
        });

        wire = patch.wire_node(format!("{}.pack", &*node.name), super::MatMatMulPack { }, &[wire],)?[0];

       let op = LirMatMulUnary { micro_ops: packed_as, };
	wire = patch.wire_node( format!("{}.matmatmul", &*node.name), op, &[wire])?[0];

        Ok(patch)
    }

}


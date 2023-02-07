use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatMatMulPack {
}

impl DynHash for MatMatMulPack {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(self, hasher)
    }
}

impl Op for MatMatMulPack {
    fn name(&self) -> Cow<str> {
        "MatMatMulPack".into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        other.downcast_ref::<Self>().map(|other| other == self).unwrap_or(false)
    }

    op_as_typed_op!();
}

impl EvalOp for MatMatMulPack {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
panic!()
    }
}

impl TypedOp for MatMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(self.output_shape(&inputs[0].shape))))
    }

    as_op!();
}

impl MatMatMulPack {
    fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
tvec!(1.into())
    }
}

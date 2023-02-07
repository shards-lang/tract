use crate::internal::*;
use ndarray::*;
use tract_itertools::Itertools;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OutputStoreSpec {
    View {
        m_axis: usize,
    },
    Strides {
        col_byte_stride: isize,
        mr: usize,
        nr: usize,
        m: usize,
        n: usize,
    },
}

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub enum ProtoFusedSpec {
    BinScalar(AttrOrInput, BinOp),
    BinPerRow(AttrOrInput, BinOp),
    BinPerCol(AttrOrInput, BinOp),
    AddRowColProducts(AttrOrInput, AttrOrInput),
    AddUnicast(OutputStoreSpec, AttrOrInput),
    Store,
}

#[derive(Clone, Debug, Hash)]
pub struct LirMatMulUnary {
    pub c_fact: TypedFact,
    pub c_m_axis: usize,
    pub c_n_axis: usize,
    pub micro_ops: ArrayD<(Arc<Tensor>, Vec<ProtoFusedSpec>)>,
    pub c_final_shape: ShapeFact,
}

impl DynHash for LirMatMulUnary {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(self, hasher)
    }
}

impl Op for LirMatMulUnary {
    fn name(&self) -> Cow<str> {
        "LirMatMulUnary".into()
    }

    op_as_typed_op!();
}

impl EvalOp for LirMatMulUnary {
    fn is_stateless(&self) -> bool {
true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
panic!()
}
}

impl TypedOp for LirMatMulUnary {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = self.c_fact.clone();
        fact.shape = self.c_final_shape.clone();
        Ok(tvec!(fact))
    }

    as_op!();
}


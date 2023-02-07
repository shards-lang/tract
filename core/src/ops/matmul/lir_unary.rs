use crate::internal::*;
use ndarray::*;
use tract_itertools::Itertools;

use tract_linalg::mmm::{
    BinOp, FusedSpec, InputStoreSpec, MatMatMul, OutputStore, OutputStoreSpec, ScratchSpace,
};
use tract_linalg::Scaler;

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub enum ProtoFusedSpec {
    BinScalar(AttrOrInput, BinOp),
    BinPerRow(AttrOrInput, BinOp),
    BinPerCol(AttrOrInput, BinOp),
    AddRowColProducts(AttrOrInput, AttrOrInput),
    AddUnicast(OutputStoreSpec, AttrOrInput),
    Scaler(Scaler),
    Store,
}

impl ProtoFusedSpec {
    pub fn name(&self) -> String {
        use ProtoFusedSpec::*;
        match self {
            BinScalar(_, op) => format!("scalar {op:?}"),
            BinPerRow(_, op) => format!("row {op:?}"),
            BinPerCol(_, op) => format!("col {op:?}"),
            AddRowColProducts(_, _) => "add row*col product".to_string(),
            AddUnicast(_, _) => "add to matrix".to_string(),
            Scaler(s) => format!("scale by {}", 1f32 * *s),
            Store => "Store".to_string(),
        }
    }

    pub fn resolve<'t>(
        &'t self,
        inputs: &'t [TValue],
        prefix: &[usize],
        output: OutputStore,
    ) -> FusedSpec<'t> {
        match self {
            ProtoFusedSpec::BinScalar(v, op) => FusedSpec::BinScalar(v.tensor(inputs), *op),
            ProtoFusedSpec::BinPerRow(v, op) => FusedSpec::BinPerRow(v.tensor(inputs), *op),
            ProtoFusedSpec::BinPerCol(v, op) => FusedSpec::BinPerCol(v.tensor(inputs), *op),
            ProtoFusedSpec::AddRowColProducts(row, col) => {
                FusedSpec::AddRowColProducts(row.tensor(inputs), col.tensor(inputs))
            }
            ProtoFusedSpec::AddUnicast(store, v) => unsafe {
                let mut view = v.tensor(inputs).view();
                for (ix, &dim) in prefix.iter().enumerate() {
                    view.offset_axis_unchecked(ix, dim as isize);
                }
                FusedSpec::AddUnicast(store.wrap(&view))
            },
            ProtoFusedSpec::Scaler(scaler) => scaler.as_fused_spec(),
            ProtoFusedSpec::Store => FusedSpec::Store(output),
        }
    }
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
        ensure!(
            self.micro_ops.ndim() == self.c_fact.rank(),
            "Constant A array rank and C rank should be the same. (resp {} and {})",
            self.micro_ops.ndim(),
            self.c_fact.rank()
        );
        let mut fact = self.c_fact.clone();
        fact.shape = self.c_final_shape.clone();
        Ok(tvec!(fact))
    }

    as_op!();
}


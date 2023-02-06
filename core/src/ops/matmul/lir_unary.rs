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
pub struct ConcreteMatMulGeometry {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub b_storage: InputStoreSpec,
}

#[derive(Clone, Debug, Hash)]
pub struct SymbolicMatMulGeometry {
    pub m: TDim,
    pub k: TDim,
    pub n: TDim,
    pub mmm: Box<dyn MatMatMul>,
    pub b_datum_type: DatumType,
}

impl ResolveTo<ConcreteMatMulGeometry> for SymbolicMatMulGeometry {
    type Param = SymbolValues;
    fn resolve(&self, param: &Self::Param) -> TractResult<ConcreteMatMulGeometry> {
        let m = self.m.eval(param).to_usize()?;
        let k = self.k.eval(param).to_usize()?;
        let n = self.n.eval(param).to_usize()?;
        let b_storage = unsafe { self.mmm.b_packed(self.b_datum_type.size_of(), k) };
        Ok(ConcreteMatMulGeometry { m, k, n, b_storage })
    }
}

pub type MatMulGeometry = GeometryBound<SymbolicMatMulGeometry, ConcreteMatMulGeometry>;

impl MatMulGeometry {
    fn k(&self) -> Cow<TDim> {
        match self {
            Self::Symbolic(it) => Cow::Borrowed(&it.k),
            Self::Concrete(it) => Cow::Owned(it.k.to_dim()),
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
    pub geometry: MatMulGeometry,
    pub mmm: Box<dyn MatMatMul>,
//    pub reshape_post: Vec<AxisOp>,
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

#[derive(Clone, Debug)]
struct State;
trivial_op_state_freeeze!(State);

impl OpState for State {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<LirMatMulUnary>().unwrap();
        let shape = op.c_fact.shape.eval_to_usize(&session.resolved_symbols)?;
        let final_shape = op.c_final_shape.eval_to_usize(&session.resolved_symbols)?;
        unsafe {
            let geometry = op.geometry.to_concrete(&session.resolved_symbols)?;
            if session
                .cached_mmm_scratch_space
                .as_deref()
                .map(|scratch| op.mmm.can_use_scratch_space(scratch))
                == Some(false)
            {
                session.cached_mmm_scratch_space = None
            }
            let scratch = session
                .cached_mmm_scratch_space
                .get_or_insert_with(|| op.mmm.allocate_scratch_space());
            eval(
                op,
                &geometry,
                scratch.as_mut(),
                &inputs,
                &shape,
                op.c_m_axis,
                op.c_n_axis,
                &final_shape,
            )
        }
    }
}

impl EvalOp for LirMatMulUnary {
    fn is_stateless(&self) -> bool {
        self.geometry.is_concrete()
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(State)))
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let geometry = self.geometry.to_concrete(&SymbolValues::default())?;
        let mut scratch = unsafe { self.mmm.allocate_scratch_space() };
        eval(
            self,
            &geometry,
            scratch.as_mut(),
            &inputs,
            self.c_fact.shape.as_concrete().unwrap(),
            self.c_m_axis,
            self.c_n_axis,
            self.c_final_shape.as_concrete().unwrap(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn eval(
    op: &LirMatMulUnary,
    geometry: &ConcreteMatMulGeometry,
    scratch: &mut dyn ScratchSpace,
    inputs: &[TValue],
    c_shape: &[usize],
    c_m_axis: usize,
    c_n_axis: usize,
    c_final_shape: &[usize],
) -> TractResult<TVec<TValue>> {
    unsafe {
        debug_assert!(op.micro_ops.len() > 0);
        let size_of_a = (*op.micro_ops.as_ptr()).0.datum_type().size_of();
        let mut c = Tensor::zero_dt(op.c_fact.datum_type, c_shape)?;
        c.set_shape_unchecked(c_final_shape);
        Ok(tvec!(c.into_tvalue()))
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


#![allow(clippy::len_zero)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::redundant_closure_call)]
#[macro_use]
pub extern crate downcast_rs;
#[macro_use]
extern crate educe;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
#[macro_use]
pub mod ops {
    use downcast_rs::Downcast;
    use std::fmt;
    #[macro_use]
    pub mod macros {
        #[macro_export]
        macro_rules! as_op {
            () => {
                fn as_op(&self) -> &dyn Op {
                    self
                }
                fn as_op_mut(&mut self) -> &mut dyn Op {
                    self
                }
            };
        }
        #[macro_export]
        macro_rules! op_as_typed_op {
            () => {
                fn as_typed(&self) -> Option<&dyn TypedOp> {
                    Some(self)
                }
            };
        }
        #[macro_export]
        macro_rules! args_1 {
            ($ inputs : expr) => {{
                if $inputs.len() != 1 {
                    $crate::internal::bail!("Expected 1 arg, got {:?}", $inputs)
                }
                let result = $inputs.pop().unwrap();
                ::std::mem::drop($inputs);
                result
            }};
        }
    }
    pub mod dummy {
        use crate::internal::*;
        #[derive(Debug, Clone, Default, Hash)]
        pub struct Dummy;
        impl Op for Dummy {
            op_as_typed_op!();
        }
        impl_dyn_hash!(Dummy);
        impl EvalOp for Dummy {
            fn is_stateless(&self) -> bool {
                unimplemented!()
            }
        }
        impl TypedOp for Dummy {
            as_op!();
            fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
                unimplemented!()
            }
        }
    }
    pub mod konst {
        use crate::internal::*;
        #[derive(Debug, Clone, Hash)]
        pub struct Const(pub Arc<Tensor>);
        impl_dyn_hash!(Const);
        impl Op for Const {
            op_as_typed_op!();
        }
        impl EvalOp for Const {
            fn is_stateless(&self) -> bool {
                true
            }
        }
        impl TypedOp for Const {
            as_op!();
            fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
                Ok(tvec!(self.0.as_ref().into()))
            }
        }
    }
    pub mod matmul {
        pub mod lir_unary {
            use crate::internal::*;
            use ndarray::*;
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
                pub micro_ops: ArrayD<(Arc<Tensor>, Vec<ProtoFusedSpec>)>,
            }
            impl DynHash for LirMatMulUnary {
                fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
                    unimplemented!()
                }
            }
            impl Op for LirMatMulUnary {
                op_as_typed_op!();
            }
            impl EvalOp for LirMatMulUnary {
                fn is_stateless(&self) -> bool {
                    unimplemented!()
                }
            }
            impl TypedOp for LirMatMulUnary {
                fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
                    Ok(tvec!(f32::fact([1, 2])))
                }
                as_op!();
            }
        }
        pub mod mir {
            use crate::ops::matmul::*;
            #[derive(Debug, Clone, Default, Hash)]
            pub struct MatMul {
                pub axes: MatMulAxes,
            }
            impl_dyn_hash!(MatMul);
            impl Op for MatMul {
                op_as_typed_op!();
            }
            impl EvalOp for MatMul {
                fn is_stateless(&self) -> bool {
                    true
                }
            }
            impl TypedOp for MatMul {
                fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
                    if inputs[0].rank() != inputs[1].rank() {
                        unimplemented!()
                    }
                    let (_m, _k, _n, c_shape) =
                        compute_shape(&inputs[0].shape, &inputs[1].shape, self.axes)?;
                    Ok(tvec!(output_type(inputs[0].datum_type).fact(c_shape)))
                }
                fn declutter(
                    &self,
                    model: &TypedModel,
                    node: &TypedNode,
                ) -> TractResult<Option<TypedModelPatch>> {
                    let a_fact = model.outlet_fact(node.inputs[0])?;
                    let b_fact = model.outlet_fact(node.inputs[1])?;
                    let konst_ix = if a_fact.konst.is_some() {
                        0
                    } else if b_fact.konst.is_some() {
                        unimplemented!()
                    } else {
                        unimplemented!()
                    };
                    let var_ix = 1 - konst_ix;
                    let flip = konst_ix == 1;
                    let axes = if flip { unimplemented!() } else { self.axes };
                    let konst = model
                        .outlet_fact(node.inputs[konst_ix])?
                        .konst
                        .clone()
                        .unwrap();
                    TypedModelPatch::replace_single_op(
                        model,
                        node,
                        &node.inputs[var_ix..][..1],
                        MatMulUnary { a: konst, axes },
                    )
                    .map(Some)
                }
                as_op!();
            }
        }
        pub mod mir_unary {
            use super::lir_unary::{LirMatMulUnary, ProtoFusedSpec};
            use super::*;
            use tract_ndarray::prelude::*;
            #[derive(Debug, Clone, Hash)]
            pub struct MatMulUnary {
                pub a: Arc<Tensor>,
                pub axes: MatMulAxes,
            }
            impl_dyn_hash!(MatMulUnary);
            impl Op for MatMulUnary {
                op_as_typed_op!();
            }
            impl EvalOp for MatMulUnary {
                fn is_stateless(&self) -> bool {
                    true
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
                        &self
                            .a
                            .shape()
                            .iter()
                            .map(|d| d.to_dim())
                            .collect::<TVec<_>>(),
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
                        unimplemented!()
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
                        (pa.into_arc_tensor(), vec![ProtoFusedSpec::Store])
                    });
                    wire = patch.wire_node(
                        format!("{}.pack", &*node.name),
                        super::MatMatMulPack {},
                        &[wire],
                    )?[0];
                    let op = LirMatMulUnary {
                        micro_ops: packed_as,
                    };
                    wire = patch.wire_node(format!("{}.matmatmul", &*node.name), op, &[wire])?[0];
                    Ok(patch)
                }
            }
        }
        pub mod pack {
            use crate::internal::*;
            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            pub struct MatMatMulPack {}
            impl DynHash for MatMatMulPack {
                fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
                    unimplemented!()
                }
            }
            impl Op for MatMatMulPack {
                op_as_typed_op!();
            }
            impl EvalOp for MatMatMulPack {
                fn is_stateless(&self) -> bool {
                    unimplemented!()
                }
            }
            impl TypedOp for MatMatMulPack {
                fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
                    Ok(tvec!(inputs[0]
                        .datum_type
                        .fact(self.output_shape(&inputs[0].shape))))
                }
                as_op!();
            }
            impl MatMatMulPack {
                fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
                    tvec!(1.into())
                }
            }
        }
        pub use self::mir::MatMul;
        pub use self::mir_unary::MatMulUnary;
        use self::pack::MatMatMulPack;
        use crate::internal::*;
        #[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
        pub struct MatMulAxes {
            pub a_m: usize,
            pub a_k: usize,
            pub b_k: usize,
            pub b_n: usize,
            pub c_m: usize,
            pub c_n: usize,
        }
        impl Default for MatMulAxes {
            fn default() -> Self {
                unimplemented!()
            }
        }
        impl MatMulAxes {
            pub fn default_for_rank(rank: usize) -> Self {
                Self::default_for_ranks(rank, rank, rank)
            }
            pub fn default_for_ranks(a: usize, b: usize, c: usize) -> Self {
                MatMulAxes {
                    a_m: a - 2,
                    a_k: a - 1,
                    b_k: b - 2,
                    b_n: b - 1,
                    c_m: c - 2,
                    c_n: c - 1,
                }
            }
            pub fn transposing_b(self) -> Self {
                MatMulAxes {
                    b_n: self.b_k,
                    b_k: self.b_n,
                    ..self
                }
            }
            pub fn transposing_c(self) -> Self {
                MatMulAxes {
                    c_n: self.c_m,
                    c_m: self.c_n,
                    ..self
                }
            }
            pub fn transposing(self, a: bool, b: bool, c: bool) -> Self {
                let mut it = self;
                if a {
                    unimplemented!()
                }
                if b {
                    it = it.transposing_b();
                }
                if c {
                    it = it.transposing_c();
                }
                it
            }
        }
        pub fn compute_shape<D: DimLike>(
            ashape: &[D],
            bshape: &[D],
            axes: MatMulAxes,
        ) -> TractResult<(D, D, D, TVec<D>)> {
            let a_shape_bc: TVec<D> = ashape
                .iter()
                .enumerate()
                .filter_map(|(ix, dim)| {
                    if ix != axes.a_m && ix != axes.a_k {
                        unimplemented!()
                    } else {
                        None
                    }
                })
                .collect();
            let b_shape_bc = bshape
                .iter()
                .enumerate()
                .filter_map(|(ix, dim)| {
                    if ix != axes.b_k && ix != axes.b_n {
                        unimplemented!()
                    } else {
                        None
                    }
                })
                .collect();
            let mut c_shape = crate::broadcast::multi_broadcast(&[a_shape_bc, b_shape_bc]).unwrap();
            let (m, ka) = (ashape[axes.a_m].clone(), ashape[axes.a_k].clone());
            let (kb, n) = (bshape[axes.b_k].clone(), bshape[axes.b_n].clone());
            if ka != kb {
                unimplemented!()
            }
            if axes.c_m < axes.c_n {
                unimplemented!()
            } else {
                c_shape.insert(axes.c_n, n.clone());
                c_shape.insert(axes.c_m, m.clone());
            }
            Ok((m, ka, n, c_shape))
        }
        pub fn output_type(input: DatumType) -> DatumType {
            if input.is_float() {
                input
            } else {
                unimplemented!()
            }
        }
    }
    pub mod source {
        use crate::internal::*;
        #[derive(Debug, Clone, Hash)]
        pub struct TypedSource {
            pub fact: TypedFact,
        }
        impl_dyn_hash!(TypedSource);
        impl Op for TypedSource {
            op_as_typed_op!();
        }
        impl EvalOp for TypedSource {
            fn is_stateless(&self) -> bool {
                false
            }
        }
        impl TypedOp for TypedSource {
            fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
                Ok(tvec!(self.fact.clone()))
            }
            as_op!();
        }
    }
    use crate::internal::*;
    use crate::optim::OptimizerSession;
    pub trait EvalOp {
        #[allow(unused_variables)]
        fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
            unimplemented!()
        }
        fn is_stateless(&self) -> bool;
    }
    pub trait Op:
        fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + EvalOp + DynHash
    {
        fn same_as(&self, _other: &dyn Op) -> bool {
            false
        }
        fn as_typed(&self) -> Option<&dyn TypedOp>;
    }
    pub trait TypedOp:
        Op + fmt::Debug + dyn_clone::DynClone + Send + Sync + 'static + Downcast + EvalOp + DynHash
    {
        fn as_op(&self) -> &dyn Op;
        fn as_op_mut(&mut self) -> &mut dyn Op;
        fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>>;
        #[allow(unused_variables)]
        fn declutter_with_session(
            &self,
            session: &mut OptimizerSession,
            model: &TypedModel,
            node: &TypedNode,
        ) -> TractResult<Option<TypedModelPatch>> {
            self.declutter(model, node)
        }
        #[allow(unused_variables)]
        fn declutter(
            &self,
            model: &TypedModel,
            node: &TypedNode,
        ) -> TractResult<Option<TypedModelPatch>> {
            Ok(None)
        }
        #[allow(unused_variables)]
        fn codegen(
            &self,
            model: &TypedModel,
            node: &TypedNode,
        ) -> TractResult<Option<TypedModelPatch>> {
            Ok(None)
        }
        #[allow(unused_variables)]
        fn nested_model_multipliers(&self, inputs: &[&TypedFact]) -> Vec<(Cow<str>, f64)> {
            unimplemented!()
        }
    }
    impl_downcast!(Op);
    dyn_clone::clone_trait_object!(TypedOp);
    impl<O: TypedOp> From<O> for Box<dyn TypedOp> {
        fn from(it: O) -> Box<dyn TypedOp> {
            Box::new(it)
        }
    }
    impl<'a> From<&'a Box<dyn TypedOp>> for Box<dyn TypedOp> {
        fn from(it: &'a Box<dyn TypedOp>) -> Box<dyn TypedOp> {
            it.clone()
        }
    }
    impl AsRef<dyn Op> for Box<dyn TypedOp> {
        fn as_ref(&self) -> &dyn Op {
            self.as_op()
        }
    }
    impl AsMut<dyn Op> for Box<dyn TypedOp> {
        fn as_mut(&mut self) -> &mut dyn Op {
            unimplemented!()
        }
    }
    impl std::fmt::Display for Box<dyn TypedOp> {
        fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
            write!(fmt, "foo")
        }
    }
    #[derive(Debug, Clone, Hash, PartialEq, Eq)]
    pub enum AttrOrInput {
        Attr(Arc<Tensor>),
        Input(usize),
    }
}
mod broadcast {
    use tract_data::internal::*;
    pub fn multi_broadcast<D>(shapes: &[impl AsRef<[D]>]) -> Option<TVec<D>>
    where
        D: DimLike,
    {
        let one = D::one();
        let len = shapes.iter().map(|shape| shape.as_ref().len()).max()?;
        let mut shape: TVec<D> = tvec!();
        for i in 0..len {
            unimplemented!()
        }
        shape.reverse();
        Some(shape)
    }
}
mod hash {
    use crate::ops::*;
    use std::hash::{Hash, Hasher};
    impl Hash for Box<dyn TypedOp> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            unimplemented!()
        }
    }
}
pub mod model {
    mod fact {
        use crate::internal::*;
        use downcast_rs::Downcast;
        use std::fmt;
        #[derive(Clone, PartialEq, Eq, Hash)]
        pub struct ShapeFact {
            dims: TVec<TDim>,
            concrete: Option<TVec<usize>>,
        }
        impl ShapeFact {
            #[inline]
            pub fn rank(&self) -> usize {
                self.dims.len()
            }
            fn compute_concrete(&mut self) {
                assert!(self
                    .dims
                    .iter()
                    .all(|d| d.to_isize().map(|d| d >= 0).unwrap_or(true)));
                self.concrete = self
                    .dims
                    .iter()
                    .map(|d| d.to_usize())
                    .collect::<TractResult<TVec<_>>>()
                    .ok()
            }
            #[inline]
            pub fn as_concrete(&self) -> Option<&[usize]> {
                self.concrete.as_deref()
            }
            pub fn from_dims<D: ToDim, T: IntoIterator<Item = D>>(it: T) -> ShapeFact {
                let mut dims = ShapeFact {
                    dims: it.into_iter().map(|d| d.to_dim()).collect(),
                    concrete: None,
                };
                dims.compute_concrete();
                dims
            }
            pub fn compatible_with(&self, _other: &ShapeFact) -> bool {
                if self.rank() == _other.rank() {
                    self.dims
                        .iter()
                        .zip(_other.dims.iter())
                        .all(|(dim, other_dim)| dim.compatible_with(other_dim))
                } else {
                    unimplemented!()
                }
            }
        }
        impl std::ops::Deref for ShapeFact {
            type Target = [TDim];
            fn deref(&self) -> &[TDim] {
                &self.dims
            }
        }
        impl<D: ToDim, T: IntoIterator<Item = D>> From<T> for ShapeFact {
            fn from(it: T) -> ShapeFact {
                ShapeFact::from_dims(it)
            }
        }
        pub trait Fact:
            std::fmt::Debug + Downcast + dyn_clone::DynClone + Send + Sync + 'static
        {
            fn to_typed_fact(&self) -> TractResult<Cow<TypedFact>>;
            fn matches(&self, t: &Tensor, symbols: Option<&SymbolValues>) -> TractResult<bool> {
                unimplemented!()
            }
            fn same_as(&self, _other: &dyn Fact) -> bool;
            fn compatible_with(&self, _other: &dyn Fact) -> bool;
            fn datum_type(&self) -> Option<DatumType>;
        }
        impl_downcast!(Fact);
        #[derive(Clone, PartialEq, Eq, Hash)]
        pub struct TypedFact {
            pub datum_type: DatumType,
            pub shape: ShapeFact,
            pub konst: Option<Arc<Tensor>>,
            pub uniform: Option<Arc<Tensor>>,
        }
        impl TypedFact {
            pub fn shape<T, S>(shape: S) -> TypedFact
            where
                T: Datum,
                S: Into<ShapeFact>,
            {
                Self::dt_shape(T::datum_type(), shape)
            }
            pub fn dt_shape<S>(datum_type: DatumType, shape: S) -> TypedFact
            where
                S: Into<ShapeFact>,
            {
                TypedFact {
                    datum_type,
                    shape: shape.into(),
                    konst: None,
                    uniform: None,
                }
            }
            pub fn rank(&self) -> usize {
                if cfg!(debug_assertions) {
                    self.consistent().unwrap();
                }
                self.shape.rank()
            }
            pub fn consistent(&self) -> TractResult<()> {
                if let Some(k) = &self.konst {
                    if !self.matches(k.as_ref(), None)? {
                        unimplemented!()
                    }
                }
                if let Some(u) = &self.uniform {
                    if self.datum_type != u.datum_type() {
                        unimplemented!()
                    }
                }
                if let (Some(u), Some(k)) = (self.uniform.as_deref(), self.konst.as_deref()) {
                    if let Some(k) = k.as_uniform() {
                        if &k != u {
                            unimplemented!()
                        }
                    } else {
                        unimplemented!()
                    }
                }
                Ok(())
            }
        }
        impl Fact for TypedFact {
            fn to_typed_fact(&self) -> TractResult<Cow<TypedFact>> {
                unimplemented!()
            }
            fn matches(&self, t: &Tensor, symbols: Option<&SymbolValues>) -> TractResult<bool> {
                if self.datum_type != t.datum_type() || self.shape.len() != t.rank() {
                    unimplemented!()
                }
                for i in 0..t.rank() {
                    if let Ok(dim) = self.shape[i]
                        .eval(symbols.unwrap_or(&SymbolValues::default()))
                        .to_usize()
                    {
                        if dim != t.shape()[i] {
                            unimplemented!()
                        }
                    }
                }
                Ok(true)
            }
            fn same_as(&self, other: &dyn Fact) -> bool {
                unimplemented!()
            }
            fn compatible_with(&self, other: &dyn Fact) -> bool {
                if cfg!(debug_assertions) {
                    self.consistent().unwrap()
                }
                if let Some(other) = other.downcast_ref::<Self>() {
                    if cfg!(debug_assertions) {
                        other.consistent().unwrap()
                    }
                    self.datum_type == other.datum_type && self.shape.compatible_with(&other.shape)
                } else {
                    unimplemented!()
                }
            }
            fn datum_type(&self) -> Option<DatumType> {
                unimplemented!()
            }
        }
        impl From<Tensor> for TypedFact {
            fn from(t: Tensor) -> TypedFact {
                TypedFact::from(t.into_arc_tensor())
            }
        }
        impl<'t> From<&'t Tensor> for TypedFact {
            fn from(t: &'t Tensor) -> TypedFact {
                TypedFact::from(t.clone())
            }
        }
        impl From<Arc<Tensor>> for TypedFact {
            fn from(t: Arc<Tensor>) -> TypedFact {
                TypedFact {
                    datum_type: t.datum_type(),
                    shape: ShapeFact::from_dims(t.shape().iter().map(TDim::from)),
                    uniform: t.as_uniform().map(Arc::new),
                    konst: Some(t),
                }
            }
        }
        impl<'a> From<&'a TypedFact> for TypedFact {
            fn from(fact: &TypedFact) -> TypedFact {
                fact.clone()
            }
        }
        impl fmt::Debug for TypedFact {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                unimplemented!()
            }
        }
        pub trait DatumExt {
            fn scalar_fact() -> TypedFact;
            fn fact<S>(shape: S) -> TypedFact
            where
                S: Into<ShapeFact>;
        }
        impl<T: Datum> DatumExt for T {
            #[allow(clippy::needless_borrow)]
            fn scalar_fact() -> TypedFact {
                unimplemented!()
            }
            fn fact<S>(shape: S) -> TypedFact
            where
                S: Into<ShapeFact>,
            {
                TypedFact::shape::<Self, _>(shape)
            }
        }
        pub trait DatumTypeExt {
            fn scalar_fact(&self) -> TypedFact;
            fn fact<S>(&self, shape: S) -> TypedFact
            where
                S: Into<ShapeFact>;
        }
        impl DatumTypeExt for DatumType {
            #[allow(clippy::needless_borrow)]
            fn scalar_fact(&self) -> TypedFact {
                unimplemented!()
            }
            fn fact<S>(&self, shape: S) -> TypedFact
            where
                S: Into<ShapeFact>,
            {
                TypedFact::dt_shape(*self, shape)
            }
        }
    }
    mod graph {
        use crate::internal::*;
        use std::fmt;
        pub trait SpecialOps<F, O> {
            fn create_dummy(&self) -> O;
            fn create_source(&self, fact: F) -> O;
            fn is_source(op: &O) -> bool;
            fn wire_node(
                &mut self,
                name: impl Into<String>,
                op: impl Into<O>,
                inputs: &[OutletId],
            ) -> TractResult<TVec<OutletId>>;
        }
        #[derive(Clone, Debug, Educe)]
        #[educe(Hash)]
        pub struct Graph<F, O>
        where
            F: Fact + Hash + Clone + 'static,
            O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
        {
            pub nodes: Vec<Node<F, O>>,
            pub inputs: Vec<OutletId>,
            pub outputs: Vec<OutletId>,
            #[educe(Hash(method = "hash_outlet_labels"))]
            pub outlet_labels: HashMap<OutletId, String>,
            #[educe(Hash(method = "hash_properties"))]
            pub properties: HashMap<String, Arc<Tensor>>,
            pub symbol_table: SymbolTable,
        }
        fn hash_outlet_labels<H: std::hash::Hasher>(it: &HashMap<OutletId, String>, state: &mut H) {
            unimplemented!()
        }
        fn hash_properties<H: std::hash::Hasher>(it: &HashMap<String, Arc<Tensor>>, state: &mut H) {
            unimplemented!()
        }
        impl<F, O> Default for Graph<F, O>
        where
            F: Fact + Hash + Clone + 'static,
            O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
        {
            fn default() -> Graph<F, O> {
                Graph {
                    nodes: vec![],
                    inputs: vec![],
                    outputs: vec![],
                    outlet_labels: HashMap::new(),
                    properties: HashMap::new(),
                    symbol_table: Default::default(),
                }
            }
        }
        impl<F, O> Graph<F, O>
        where
            F: Fact + Hash + Clone + 'static,
            O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
            Graph<F, O>: SpecialOps<F, O>,
        {
            pub fn add_source(
                &mut self,
                name: impl Into<String>,
                fact: F,
            ) -> TractResult<OutletId> {
                let source = self.create_source(fact.clone());
                let id = self.add_node(name, source, tvec!(fact))?;
                let id = OutletId::new(id, 0);
                self.inputs.push(id);
                Ok(id)
            }
        }
        impl<F, O> Graph<F, O>
        where
            F: Fact + Hash + Clone + 'static,
            O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
        {
            pub fn add_node(
                &mut self,
                name: impl Into<String>,
                op: impl Into<O>,
                output_facts: TVec<F>,
            ) -> TractResult<usize> {
                let op = op.into();
                let name = name.into();
                let id = self.nodes.len();
                let outputs = output_facts
                    .into_iter()
                    .map(|fact| Outlet {
                        fact,
                        successors: tvec!(),
                    })
                    .collect();
                let node = Node {
                    id,
                    name,
                    op,
                    inputs: vec![],
                    outputs,
                };
                self.nodes.push(node);
                Ok(id)
            }
            pub fn add_edge(&mut self, outlet: OutletId, inlet: InletId) -> TractResult<()> {
                if let Some(previous) = self.nodes[inlet.node].inputs.get(inlet.slot).cloned() {
                    unimplemented!()
                }
                {
                    let prec = &mut self.nodes[outlet.node];
                    prec.outputs[outlet.slot].successors.push(inlet);
                }
                let succ = &mut self.nodes[inlet.node];
                #[allow(clippy::comparison_chain)]
                if inlet.slot == succ.inputs.len() {
                    succ.inputs.push(outlet);
                } else if inlet.slot < succ.inputs.len() {
                    unimplemented!()
                } else {
                    unimplemented!()
                }
                Ok(())
            }
            pub fn input_outlets(&self) -> TractResult<&[OutletId]> {
                Ok(&self.inputs)
            }
            pub fn set_input_outlets(&mut self, inputs: &[OutletId]) -> TractResult<()> {
                self.inputs = inputs.to_vec();
                Ok(())
            }
            pub fn output_outlets(&self) -> TractResult<&[OutletId]> {
                Ok(&self.outputs)
            }
            pub fn set_output_outlets(&mut self, outputs: &[OutletId]) -> TractResult<()> {
                self.outputs = outputs.to_vec();
                Ok(())
            }
            pub fn node(&self, id: usize) -> &Node<F, O> {
                &self.nodes[id]
            }
            pub fn node_mut(&mut self, id: usize) -> &mut Node<F, O> {
                &mut self.nodes[id]
            }
            pub fn nodes(&self) -> &[Node<F, O>] {
                &self.nodes
            }
            pub fn node_input_facts(&self, node_id: usize) -> TractResult<TVec<&F>> {
                self.nodes[node_id]
                    .inputs
                    .iter()
                    .map(|o| self.outlet_fact(*o))
                    .collect()
            }
            pub fn outlet_fact(&self, outlet: OutletId) -> TractResult<&F> {
                let outlets = &self.nodes[outlet.node].outputs;
                Ok(outlets
                    .get(outlet.slot)
                    .map(|o| &o.fact).unwrap())
            }
            pub fn outlet_label(&self, outlet: OutletId) -> Option<&str> {
                self.outlet_labels.get(&outlet).map(|s| &**s)
            }
            pub fn eval_order(&self) -> TractResult<Vec<usize>> {
                eval_order(self)
            }
            #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
            #[inline]
            pub fn check_edges(&self) -> TractResult<()> {
                for node_id in self.eval_order()? {
                    let node = &self.nodes[node_id];
                    for (ix, input) in node.inputs.iter().enumerate() {
                        let prec = &self.nodes[input.node];
                        if !prec.outputs[input.slot]
                            .successors
                            .contains(&InletId { node: node.id, slot: ix })
                        {
                            unimplemented!()
                        }
                    }
                    for (ix, output) in node.outputs.iter().enumerate() {
                        for succ in &output.successors {
                            if self.nodes[succ.node].inputs[succ.slot] != OutletId::new(node.id, ix)
                            {
                                unimplemented!()
                            }
                        }
                    }
                }
                Ok(())
            }
            pub fn outlet_successors(&self, outlet: OutletId) -> &[InletId] {
                &self.nodes[outlet.node].outputs[outlet.slot].successors
            }
        }
        impl<F: Fact + Clone + 'static, O> Graph<F, O>
        where
            F: Fact + Clone + 'static + From<std::sync::Arc<Tensor>> + Hash,
            O: fmt::Debug
                + fmt::Display
                + From<crate::ops::konst::Const>
                + AsRef<dyn Op>
                + AsMut<dyn Op>
                + Clone
                + Hash
                + 'static,
        {
            pub fn add_const(
                &mut self,
                name: impl Into<String>,
                v: impl IntoArcTensor,
            ) -> TractResult<OutletId> {
                let v = v.into_arc_tensor();
                let fact = F::from(v.clone());
                let name = name.into();
                self.add_node(name, crate::ops::konst::Const(v), tvec!(fact))
                    .map(|id| id.into())
            }
        }
        impl<F, O> Graph<F, O>
        where
            F: Fact + Clone + 'static + std::hash::Hash + for<'a> std::convert::From<&'a F>,
            O: std::fmt::Display
                + std::fmt::Debug
                + Clone
                + AsRef<dyn Op>
                + AsMut<dyn Op>
                + Clone
                + 'static
                + std::hash::Hash
                + for<'a> std::convert::From<&'a O>,
            Graph<F, O>: SpecialOps<F, O>,
        {
            #[cfg(debug_assertions)]
            pub fn check_compact(&self) -> TractResult<()> {
                let order = self.eval_order()?;
                let useless_sources = self
                    .input_outlets()?
                    .iter()
                    .filter(|io| {
                        self.outlet_successors(**io).len() == 0
                            && !self.output_outlets().unwrap().contains(io)
                    })
                    .count();
                if order.len() + useless_sources != self.nodes.len() {
                    unimplemented!()
                }
                if (0..order.len()).any(|ix| order[ix] != ix) {
                    unimplemented!()
                }
                let mut seen = std::collections::HashSet::new();
                for (ix, n) in self.nodes.iter().enumerate() {
                    if ix != n.id {
                        unimplemented!()
                    }
                    if seen.contains(&n.name) {
                        unimplemented!()
                    }
                    seen.insert(&n.name);
                }
                Ok(())
            }
            pub fn compact(&mut self) -> TractResult<()> {
                use crate::model::translator::Translate;
                let mut result = crate::model::translator::IntoTranslator.translate_model(self)?;
                std::mem::swap(self, &mut result);
                Ok(())
            }
        }
    }
    mod node {
        use crate::internal::*;
        use std::fmt;
        use std::fmt::{Debug, Display};
        #[derive(Debug, Clone, Educe)]
        #[educe(Hash)]
        pub struct Node<F: Fact + Hash, O: Hash> {
            pub id: usize,
            pub name: String,
            pub inputs: Vec<OutletId>,
            #[cfg_attr(feature = "serialize", serde(skip))]
            pub op: O,
            pub outputs: TVec<Outlet<F>>,
        }
        impl<F: Fact + Hash, O: Hash + std::fmt::Display> fmt::Display for Node<F, O> {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                write!(fmt, "#{} \"{}\" {}", self.id, self.name, self.op)
            }
        }
        impl<F, NodeOp> Node<F, NodeOp>
        where
            F: Fact + Hash,
            NodeOp: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + AsMut<dyn Op> + Hash,
        {
            pub fn op(&self) -> &dyn Op {
                self.op.as_ref()
            }
            pub fn op_as<O: Op>(&self) -> Option<&O> {
                self.op().downcast_ref::<O>()
            }
            pub fn op_is<O: Op>(&self) -> bool {
                self.op_as::<O>().is_some()
            }
        }
        #[derive(Clone, Default, Educe)]
        #[educe(Hash)]
        pub struct Outlet<F: Fact + Hash> {
            pub fact: F,
            pub successors: TVec<InletId>,
        }
        impl<F: Fact + Hash> fmt::Debug for Outlet<F> {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                unimplemented!()
            }
        }
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct OutletId {
            pub node: usize,
            pub slot: usize,
        }
	impl OutletId {
	    pub fn new(node: usize, slot: usize) -> OutletId {
                OutletId { node, slot }
            }
	}
        impl fmt::Debug for OutletId {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                unimplemented!()
            }
        }
        impl From<usize> for OutletId {
            fn from(node: usize) -> OutletId {
                OutletId::new(node, 0)
            }
        }
        impl From<(usize, usize)> for OutletId {
            fn from(pair: (usize, usize)) -> OutletId {
                OutletId::new(pair.0, pair.1)
            }
        }
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
        pub struct InletId {
            pub node: usize,
            pub slot: usize,
        }
    }
    pub mod order {
        use crate::internal::*;
        use std::fmt::{Debug, Display};
        pub fn eval_order<F, O>(model: &super::Graph<F, O>) -> TractResult<Vec<usize>>
        where
            F: Fact + Hash + Clone + 'static,
            O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
        {
            let inputs = model
                .input_outlets()?
                .iter()
                .map(|n| n.node)
                .collect::<Vec<usize>>();
            let targets = model
                .output_outlets()?
                .iter()
                .map(|n| n.node)
                .collect::<Vec<usize>>();
            eval_order_for_nodes(model.nodes(), &inputs, &targets, &[])
        }
        pub fn eval_order_for_nodes<F, O>(
            nodes: &[Node<F, O>],
            model_inputs: &[usize],
            model_outputs: &[usize],
            more_dependencies: &[(usize, usize)],
        ) -> TractResult<Vec<usize>>
        where
            F: Fact + Hash + Clone + 'static,
            O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
        {
            let mut done = std::collections::HashSet::new();
            let mut order: Vec<usize> = vec![];
            for &model_target in model_outputs {
                if done.contains(&model_target) {
                    unimplemented!()
                }
                let mut current_stack: Vec<(usize, usize)> = vec![(model_target, 0)];
                let mut pending = std::collections::HashSet::new();
                while let Some((current_node, current_input)) = current_stack.pop() {
                    let deps_from_inputs = nodes[current_node].inputs.len();
                    let all_deps_count = deps_from_inputs
                        + more_dependencies
                            .iter()
                            .filter(|a| a.0 == current_node)
                            .count();
                    if model_inputs.contains(&current_node) || current_input == all_deps_count {
                        order.push(current_node);
                        done.insert(current_node);
                        pending.remove(&current_node);
                    } else {
                        let precursor: usize = nodes[current_node]
                            .inputs
                            .iter()
                            .filter(|n| nodes[n.node].inputs.len() > 0)
                            .map(|n| n.node)
                            .chain(
                                more_dependencies
                                    .iter()
                                    .filter(|a| a.0 == current_node)
                                    .map(|n| n.1),
                            )
                            .chain(
                                nodes[current_node]
                                    .inputs
                                    .iter()
                                    .filter(|n| nodes[n.node].inputs.len() == 0)
                                    .map(|n| n.node),
                            )
                            .nth(current_input)
                            .unwrap();
                        if done.contains(&precursor) {
                            current_stack.push((current_node, current_input + 1));
                        } else if pending.contains(&precursor) {
                            unimplemented!()
                        } else {
                            pending.insert(precursor);
                            current_stack.push((current_node, current_input));
                            current_stack.push((precursor, 0));
                        }
                    }
                }
            }
            Ok(order)
        }
    }
    mod patch {
        use crate::internal::*;
        use std::fmt::{Debug, Display};
        use std::ops::{Deref, DerefMut};
        use tract_data::itertools::{izip, Itertools};
        #[derive(Clone, Debug)]
        pub struct ModelPatch<F, O>
        where
            F: Fact + Clone + 'static + Hash,
            O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
        {
            pub context: Vec<String>,
            pub dont_apply_twice: Option<String>,
            pub model: Graph<F, O>,
            pub inputs: HashMap<usize, usize>,
            pub incoming: HashMap<OutletId, OutletId>,
            pub shunt_outlet_by: HashMap<OutletId, OutletId>,
            pub obliterate: Vec<usize>,
        }
        impl<F, O> Default for ModelPatch<F, O>
        where
            F: Fact + Clone + 'static + Hash,
            O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
        {
            fn default() -> ModelPatch<F, O> {
                ModelPatch {
                    context: vec![],
                    dont_apply_twice: None,
                    model: Graph::default(),
                    inputs: HashMap::default(),
                    incoming: HashMap::new(),
                    shunt_outlet_by: HashMap::new(),
                    obliterate: vec![],
                }
            }
        }
        impl<F, O> Deref for ModelPatch<F, O>
        where
            F: Fact + Clone + 'static + Hash,
            O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
        {
            type Target = Graph<F, O>;
            fn deref(&self) -> &Graph<F, O> {
                &self.model
            }
        }
        impl<F, O> DerefMut for ModelPatch<F, O>
        where
            F: Fact + Clone + 'static + Hash,
            O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
        {
            fn deref_mut(&mut self) -> &mut Graph<F, O> {
                &mut self.model
            }
        }
        impl<F, O> ModelPatch<F, O>
        where
            F: Fact + Clone + 'static + Hash,
            O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
            Graph<F, O>: SpecialOps<F, O>,
        {
            pub fn push_context(&mut self, s: impl Into<String>) {
                self.context.push(s.into());
            }
            pub fn tap_model(
                &mut self,
                model: &Graph<F, O>,
                outlet: OutletId,
            ) -> TractResult<OutletId> {
                let fact = model.outlet_fact(outlet)?;
                let id = self.add_source(
                    format!("incoming-{}/{}", outlet.node, outlet.slot),
                    dyn_clone::clone(fact),
                )?;
                self.incoming.insert(id, outlet);
                Ok(id)
            }
            pub fn shunt_outside(
                &mut self,
                model: &Graph<F, O>,
                outlet: OutletId,
                by: OutletId,
            ) -> TractResult<()> {
                let original_fact = model.outlet_fact(outlet)?;
                let new_fact = self.model.outlet_fact(by)?;
                if !original_fact.compatible_with(new_fact) {
                    unimplemented!()
                }
                self.shunt_outlet_by.insert(outlet, by);
                Ok(())
            }
            pub fn obliterate(&mut self, node: usize) -> TractResult<()> {
                self.obliterate.push(node);
                Ok(())
            }
            pub fn replace_single_op<IO: Into<O>>(
                patched_model: &Graph<F, O>,
                node: &Node<F, O>,
                inputs: &[OutletId],
                new_op: IO,
            ) -> TractResult<ModelPatch<F, O>> {
                let mut patch = ModelPatch::default();
                let new_op = new_op.into();
                let inputs = inputs
                    .iter()
                    .map(|i| patch.tap_model(patched_model, *i))
                    .collect::<TractResult<TVec<_>>>()?;
                let wires = patch.wire_node(&node.name, new_op, &inputs)?;
                for (ix, o) in wires.iter().enumerate() {
                    patch.shunt_outside(patched_model, OutletId::new(node.id, ix), *o)?;
                }
                patch.obliterate(node.id)?;
                Ok(patch)
            }
            pub fn apply(self, target: &mut Graph<F, O>) -> TractResult<()> {
                let prior_target_inputs = target.input_outlets()?.len();
                let prior_target_outputs = target.output_outlets()?.len();
                let ModelPatch {
                    model: patch,
                    incoming: mut mapping,
                    shunt_outlet_by,
                    obliterate,
                    inputs: replaced_inputs,
                    ..
                } = self;
                let mut all_inputs = HashMap::new();
                let mut model_input_outlets = target.input_outlets()?.to_vec();
                for node in patch.nodes {
                    if <Graph<F, O>>::is_source(&node.op)
                        && mapping.contains_key(&OutletId::new(node.id, 0))
                    {
                        continue;
                    }
                    let Node {
                        id: patch_node_id,
                        name,
                        inputs,
                        op,
                        outputs,
                    } = node;
                    let n_outputs = outputs.len();
                    for dup in 0..target.nodes.len() {
                        if target.node(dup).op().same_as(op.as_ref())
                            && inputs.len() == target.node(dup).inputs.len()
                            && inputs
                                .iter()
                                .zip(target.node(dup).inputs.iter())
                                .all(|(patch_input, d)| mapping[patch_input] == *d)
                        {
                            unimplemented!()
                        }
                    }
                    let facts = outputs.into_iter().map(|of| of.fact).collect();
                    let added_node_id = target.add_node(name, op, facts)?;
                    for ix in 0..n_outputs {
                        mapping.insert(
                            OutletId::new(patch_node_id, ix),
                            OutletId::new(added_node_id, ix),
                        );
                    }
                    all_inputs.insert(added_node_id, inputs);
                    if <Graph<F, O>>::is_source(&target.node(added_node_id).op) {
                        unimplemented!()
                    }
                }
                debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
                debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
                for (outlet, by) in shunt_outlet_by {
                    let replace_by = mapping[&by];
                    let succs = target.nodes()[outlet.node].outputs[outlet.slot]
                        .successors
                        .clone();
                    for succ in succs {
                        unimplemented!()
                    }
                    for o in target.outputs.iter_mut() {
                        if *o == outlet {
                            *o = replace_by;
                        }
                    }
                    if let Some(label) = target.outlet_labels.remove(&outlet) {
                        unimplemented!()
                    }
                }
                if target.outputs.len() > target.outputs.iter().sorted().dedup().count() {
                    unimplemented!()
                }
                debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
                debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
                for (node, inputs) in all_inputs {
                    for (ix, input) in inputs.into_iter().enumerate() {
                        target.add_edge(mapping[&input], InletId { node, slot: ix})?;
                    }
                }
                debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
                debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
                for node in obliterate {
                    target.node_mut(node).op = target.create_dummy();
                }
                debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
                debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
                target.set_input_outlets(&model_input_outlets)?;
                Ok(())
            }
        }
    }
    pub mod translator {
        use crate::internal::*;
        use std::fmt;
        pub trait Translate<TI1, O1, TI2, O2>: fmt::Debug
        where
            TI1: Fact + Hash + Clone + 'static,
            TI2: Fact + Hash + Clone + 'static,
            O1: fmt::Display + fmt::Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
            O2: fmt::Display + fmt::Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
        {
            fn translate_node(
                &self,
                source: &Graph<TI1, O1>,
                node: &Node<TI1, O1>,
                target: &mut Graph<TI2, O2>,
                mapping: &HashMap<OutletId, OutletId>,
            ) -> TractResult<TVec<OutletId>>;
            fn translate_model(&self, source: &Graph<TI1, O1>) -> TractResult<Graph<TI2, O2>> {
                Ok(self.translate_model_with_mappings(source)?.0)
            }
            fn translate_model_with_mappings(
                &self,
                source: &Graph<TI1, O1>,
            ) -> TractResult<(Graph<TI2, O2>, HashMap<OutletId, OutletId>)> {
                let mut target = Graph::default();
                let mut mapping = HashMap::new();
                for old_id in source.eval_order()? {
                    let node = source.node(old_id);
                    trace!("Translating {} {:?}", node, self);
                    let outlets = self
                        .translate_node(source, node, &mut target, &mapping)?;
                    for (ix, outlet) in outlets.into_iter().enumerate() {
                        mapping.insert(OutletId::new(node.id, ix), outlet);
                        if let Some(label) = source.outlet_label(OutletId::new(node.id, ix)) {
                            unimplemented!()
                        }
                    }
                }
                for i in source.input_outlets()? {
                    if !mapping.contains_key(i) {
                        unimplemented!()
                    }
                }
                target.inputs = source.input_outlets()?.iter().map(|i| mapping[i]).collect();
                target.outputs = source
                    .output_outlets()?
                    .iter()
                    .map(|o| mapping[o])
                    .collect();
                target.symbol_table = source.symbol_table.clone();
                target.properties = source.properties.clone();
                Ok((target, mapping))
            }
        }
        #[derive(Debug)]
        pub struct IntoTranslator;
        impl<TI1, O1, TI2, O2, EO, ETI> Translate<TI1, O1, TI2, O2> for IntoTranslator
        where
            TractError: From<EO> + From<ETI>,
            TI1: Fact + Hash + Clone + 'static,
            TI2: Fact + Hash + for<'a> TryFrom<&'a TI1, Error = EO> + Clone + 'static,
            O1: fmt::Display
                + fmt::Debug
                + Clone
                + AsRef<dyn Op>
                + AsMut<dyn Op>
                + Clone
                + 'static
                + Hash,
            O2: fmt::Display
                + for<'a> TryFrom<&'a O1, Error = ETI>
                + fmt::Debug
                + AsRef<dyn Op>
                + AsMut<dyn Op>
                + Clone
                + Hash
                + 'static,
            Graph<TI2, O2>: SpecialOps<TI2, O2>,
        {
            fn translate_node(
                &self,
                source: &Graph<TI1, O1>,
                node: &Node<TI1, O1>,
                target: &mut Graph<TI2, O2>,
                mapping: &HashMap<OutletId, OutletId>,
            ) -> TractResult<TVec<OutletId>> {
                let node_is_input =
                    (0..node.outputs.len()).all(|o| source.inputs.contains(&(node.id, o).into()));
                if node_is_input {
                    (0..node.outputs.len())
                        .map(|i| {
                            target.add_source(
                                if node.outputs.len() > 1 {
                                    unimplemented!()
                                } else {
                                    node.name.to_string()
                                },
                                TI2::try_from(&node.outputs[i].fact)?,
                            )
                        })
                        .collect()
                } else {
                    let new_op = O2::try_from(&node.op)?;
                    let facts = node
                        .outputs
                        .iter()
                        .map(|of| Ok(TI2::try_from(&of.fact)?))
                        .collect::<TractResult<TVec<_>>>()?;
                    let new_id = target.add_node(node.name.clone(), new_op, facts)?;
                    for (ix, o) in node.inputs.iter().enumerate() {
                        target.add_edge(mapping[o], InletId { node: new_id, slot: ix })?
                    }
                    Ok(node
                        .outputs
                        .iter()
                        .enumerate()
                        .map(|(ix, _)| OutletId::new(new_id, ix))
                        .collect())
                }
            }
        }
    }
    pub mod typed {
        use crate::internal::*;
        use crate::ops;
        pub type TypedModel = Graph<TypedFact, Box<dyn TypedOp>>;
        pub type TypedNode = Node<TypedFact, Box<dyn TypedOp>>;
        pub type TypedModelPatch = ModelPatch<TypedFact, Box<dyn TypedOp>>;
        impl SpecialOps<TypedFact, Box<dyn TypedOp>> for TypedModel {
            fn is_source(op: &Box<dyn TypedOp>) -> bool {
                op.as_op()
                    .downcast_ref::<ops::source::TypedSource>()
                    .is_some()
            }
            fn create_dummy(&self) -> Box<dyn TypedOp> {
                Box::new(crate::ops::dummy::Dummy::default())
            }
            fn create_source(&self, fact: TypedFact) -> Box<dyn TypedOp> {
                Box::new(crate::ops::source::TypedSource { fact })
            }
            fn wire_node(
                &mut self,
                name: impl Into<String>,
                op: impl Into<Box<dyn TypedOp>>,
                inputs: &[OutletId],
            ) -> TractResult<TVec<OutletId>> {
                let op = op.into();
                let name = name.into();
                {
                    let output_facts = || -> TractResult<TVec<TypedFact>> {
                        let input_facts = inputs
                            .iter()
                            .map(|o| self.outlet_fact(*o))
                            .collect::<TractResult<TVec<_>>>()?;
                        let facts = op
                            .output_facts(&input_facts)?;
                        if input_facts.iter().all(|f| f.konst.is_some()) && op.is_stateless() {
                            unimplemented!()
                        }
                        Ok(facts)
                    };
                    let output_facts = output_facts()?;
                    let id = self.add_node(&name, &op, output_facts)?;
                    inputs
                        .iter()
                        .enumerate()
                        .try_for_each(|(ix, i)| self.add_edge(*i, InletId { node: id, slot:ix }))?;
                    TractResult::Ok(
                        self.node(id)
                            .outputs
                            .iter()
                            .enumerate()
                            .map(|(ix, _)| OutletId::new(id, ix))
                            .collect(),
                    )
                }
            }
        }
        impl TypedModel {
            pub fn into_optimized(mut self) -> TractResult<TypedModel> {
                self.declutter()?;
                self.optimize()?;
                Ok(self)
            }
            pub fn check_consistency(&self) -> TractResult<()> {
                Ok(())
            }
            pub fn into_decluttered(mut self) -> TractResult<TypedModel> {
                self.declutter()?;
                Ok(self)
            }
            pub fn declutter(&mut self) -> TractResult<()> {
                crate::optim::Optimizer::declutter()
                    .session()
                    .optimize(self)
            }
            pub fn optimize(&mut self) -> TractResult<()> {
                crate::optim::Optimizer::codegen().optimize(self)
            }
        }
    }
    pub use self::fact::*;
    pub use self::graph::*;
    pub use self::node::*;
    pub use self::order::eval_order;
    pub use self::patch::ModelPatch;
    pub use crate::ops::{Op, TypedOp};
    pub use typed::*;
}
pub mod optim {
    use crate::internal::*;
    use std::collections::HashSet;
    use std::fmt::Debug;
    use tract_itertools::Itertools;
    mod op_optim {
        use super::OptimizerSession;
        use crate::internal::*;
        #[derive(Clone)]
        pub struct OpOptim(
            pub &'static str,
            pub  fn(
                op: &dyn TypedOp,
                session: &mut OptimizerSession,
                model: &TypedModel,
                node: &TypedNode,
            ) -> TractResult<Option<TypedModelPatch>>,
            pub usize,
        );
        impl OpOptim {
            fn full_pass(
                &mut self,
                session: &mut OptimizerSession,
                new: &TypedModel,
            ) -> TractResult<Option<TypedModelPatch>> {
                for (ix, &id) in new.eval_order()?.iter().enumerate().skip(self.2) {
                    let node = &new.nodes()[id];
                    let patch = (self.1)(node.op.as_ref(), session, new, node)?;
                    if let Some(mut p) = patch {
                        p.push_context(format!("{self:?} {node}"));
                        self.2 = ix + p.dont_apply_twice.is_some() as usize;
                        return Ok(Some(p));
                    }
                }
                Ok(None)
            }
        }
        impl std::fmt::Debug for OpOptim {
            fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "{}", self.0)
            }
        }
        impl super::TypedPass for OpOptim {
            fn reset(&mut self) -> TractResult<()> {
                self.2 = 0;
                Ok(())
            }
            fn next(
                &mut self,
                session: &mut OptimizerSession,
                model: &TypedModel,
            ) -> TractResult<Option<TypedModelPatch>> {
                self.full_pass(session, model)
            }
        }
    }
    mod prop_const {
        use crate::internal::*;
        use crate::ops::konst::Const;
        use crate::optim::OptimizerSession;
        #[derive(Clone, Debug)]
        pub struct PropConst;
        impl super::TypedPass for PropConst {
            fn reset(&mut self) -> TractResult<()> {
                Ok(())
            }
            fn next(
                &mut self,
                _session: &mut OptimizerSession,
                model: &TypedModel,
            ) -> TractResult<Option<TypedModelPatch>> {
                let mut patch = TypedModelPatch::default();
                for n in model.eval_order()? {
                    let node = model.node(n);
                    if node.op.is_stateless() && !node.op_is::<Const>() {
                        if let Some(inputs) = model
                            .node_input_facts(n)?
                            .iter()
                            .map(|f| f.konst.clone().map(|t| t.into_tvalue()))
                            .collect()
                        {
                            match node.op.eval(inputs) {
                                Ok(res) => {
                                    unimplemented!()
                                }
                                Err(e) => {
                                    unimplemented!()
                                }
                            }
                        }
                    }
                }
                Ok(Some(patch).filter(|p| p.nodes.len() > 0))
            }
        }
    }
    use self::prop_const::PropConst;
    use op_optim::OpOptim;
    pub trait TypedPass: Debug + Send + Sync + dyn_clone::DynClone {
        fn reset(&mut self) -> TractResult<()>;
        fn next(
            &mut self,
            session: &mut OptimizerSession,
            model: &TypedModel,
        ) -> TractResult<Option<TypedModelPatch>>;
    }
    dyn_clone::clone_trait_object!(TypedPass);
    #[derive(Debug)]
    pub struct Optimizer {
        passes: Vec<Box<dyn TypedPass>>,
        steps: Option<usize>,
    }
    impl Optimizer {
        fn passes(passes: Vec<Box<dyn TypedPass>>) -> Optimizer {
            Optimizer {
                passes,
                steps: None,
            }
        }
        pub fn declutter() -> Optimizer {
            Optimizer::passes(vec![
                Box::new(PropConst),
                Box::new(OpOptim("declutter", TypedOp::declutter_with_session, 0)),
            ])
        }
        pub fn codegen() -> Optimizer {
            Optimizer::passes(vec![
                Box::new(PropConst),
                Box::new(OpOptim(
                    "codegen",
                    |op, _session, model, node| TypedOp::codegen(op, model, node),
                    0,
                )),
                Box::new(OpOptim("declutter", TypedOp::declutter_with_session, 0)),
            ])
        }
        pub fn optimize(&self, model: &mut TypedModel) -> TractResult<()> {
            self.session().optimize(model)
        }
        pub fn session(&self) -> OptimizerSession {
            OptimizerSession {
                optimizer: self,
                counter: 0,
                seen: Default::default(),
            }
        }
    }
    #[derive(Debug)]
    pub struct OptimizerSession<'o> {
        optimizer: &'o Optimizer,
        counter: usize,
        seen: HashSet<String>,
    }
    impl<'o> OptimizerSession<'o> {
        pub fn optimize(&mut self, model: &mut TypedModel) -> TractResult<()> {
            model
                .check_consistency()?;
            model
                .compact()?;
            for i in 0.. {
                let old = self.counter;
                self.run_all_passes(i, model)?;
                if old == self.counter {
                    return Ok(());
                }
                model.compact()?;
            }
            unreachable!()
        }
        pub fn run_all_passes(&mut self, i: usize, model: &mut TypedModel) -> TractResult<()> {
            let mut passes = self.optimizer.passes.clone();
            for p in passes.iter_mut() {
                self.run_one_pass_outer(i, p.as_mut(), model)?;
                model.compact()?;
                model
                    .check_consistency()?;
            }
            Ok(())
        }
        pub fn run_one_pass_outer(
            &mut self,
            i: usize,
            p: &mut dyn TypedPass,
            model: &mut TypedModel,
        ) -> TractResult<()> {
            loop {
                let old_counter = self.counter;
                self.run_one_pass_inner(i, p, model)?;
                if self.counter == old_counter {
                    return Ok(());
                }
                model
                    .compact()?;
            }
        }
        pub fn run_one_pass_inner(
            &mut self,
            i: usize,
            p: &mut dyn TypedPass,
            model: &mut TypedModel,
        ) -> TractResult<()> {
            p.reset()?;
            if let Some(steps) = self.optimizer.steps {
                unimplemented!()
            }
            while let Some(mut patch) = p.next(self, model)? {
                patch.push_context(format!("{p:?}/{i}"));
                patch
                    .model
                    .check_consistency()?;
                model
                    .check_consistency()?;
                if let Some(watchdog) = patch.dont_apply_twice.take() {
                    unimplemented!()
                }
                debug!(
                    "applying patch #{}: {}",
                    self.counter,
                    patch.context.iter().rev().join(" >> "),
                );
                patch.apply(model)?;
                model
                    .check_consistency()?;
                self.counter += 1;
                if let Some(steps) = self.optimizer.steps {
                    unimplemented!()
                }
            }
            model
                .check_consistency()?;
            Ok(())
        }
    }
}
pub mod value {
    use crate::internal::*;
    use std::rc::Rc;
    use TValue::*;
    #[derive(Clone, PartialEq, Eq)]
    pub enum TValue {
        Const(Arc<Tensor>),
        Var(Rc<Tensor>),
    }
    pub trait IntoTValue {
        fn into_tvalue(self) -> TValue;
    }
    impl IntoTValue for Arc<Tensor> {
        fn into_tvalue(self) -> TValue {
            Const(self)
        }
    }
}
pub mod prelude {
    pub use crate::value::{IntoTValue, TValue};
    pub use std::sync::Arc;
}
pub mod internal {
    pub use crate::model::*;
    pub use crate::ops::{AttrOrInput, EvalOp, Op};
    pub use crate::prelude::*;
    pub use std::borrow::Cow;
    pub use std::collections::HashMap;
    pub use std::hash::Hash;
    pub use tract_data::internal::*;
}
#[test]
fn crasher_monterey_matmul() {
    use crate::internal::*;
    use crate::ops::matmul::*;
    use tract_ndarray::prelude::*;
    let mut model = TypedModel::default();
    let wire = model.add_source("input", f32::fact(&[1usize, 1])).unwrap();
    let a = model
        .add_const("a", Tensor::zero::<f32>(&[2, 1]).unwrap().into_arc_tensor())
        .unwrap();
    let axes = MatMulAxes::default_for_rank(2).transposing(false, true, true);
    let op = MatMul { axes };
    let wire = model.wire_node("conv", op, &[a, wire]).unwrap()[0];
    model.set_output_outlets(&[wire]).unwrap();
    let decluttered = model.into_decluttered().unwrap();
    let optimized = decluttered.into_optimized().unwrap();
}

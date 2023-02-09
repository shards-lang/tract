pub type TractResult<T> = anyhow::Result<T>;
use ndarray::*;
pub struct Tensor;
use std::sync::Arc;
macro_rules! as_op {
    () => {
        fn as_any(&self) -> &dyn Any {
            self
        }
    };
}
#[derive(Clone)]
pub struct Const;
impl Op for Const {}
impl TypedOp for Const {
    as_op!();
}
#[derive(Copy, Clone)]
pub enum BinOp {
    Min,
    Max,
}
#[derive(Clone, Copy)]
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
#[derive(Clone)]
pub enum ProtoFusedSpec {
    BinScalar(AttrOrInput, BinOp),
    BinPerRow(AttrOrInput, BinOp),
    BinPerCol(AttrOrInput, BinOp),
    AddRowColProducts(AttrOrInput, AttrOrInput),
    AddUnicast(OutputStoreSpec, AttrOrInput),
    Store,
}
#[derive(Clone)]
pub struct LirMatMulUnary {
    pub micro_ops: ArrayD<(Arc<Tensor>, Vec<ProtoFusedSpec>)>,
}
impl Op for LirMatMulUnary {}
impl TypedOp for LirMatMulUnary {
    as_op!();
}
#[derive(Clone)]
pub struct MatMul {}
impl Op for MatMul {}
impl TypedOp for MatMul {
    as_op!();
}
#[derive(Clone)]
pub struct MatMulUnary {
    pub a: Arc<Tensor>,
}
impl Op for MatMulUnary {}
impl TypedOp for MatMulUnary {
    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let packed_as = Array::from_shape_fn(vec![1, 1], |_| {
            (Arc::new(Tensor), vec![ProtoFusedSpec::Store])
        });
        TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs,
            LirMatMulUnary {
                micro_ops: packed_as,
            },
        )
        .map(Some)
    }
    as_op!();
}
#[derive(Clone)]
pub struct TypedSource {}
impl Op for TypedSource {}
impl TypedOp for TypedSource {
    as_op!();
}
use std::any::Any;
pub trait Op: dyn_clone::DynClone + Send + Sync + 'static {}
pub trait TypedOp: Op + dyn_clone::DynClone + Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
    #[allow(unused_variables)]
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        unimplemented!()
    }
    #[allow(unused_variables)]
    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        unimplemented!()
    }
}
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
        unimplemented!()
    }
}
impl AsMut<dyn Op> for Box<dyn TypedOp> {
    fn as_mut(&mut self) -> &mut dyn Op {
        unimplemented!()
    }
}
#[derive(Clone)]
pub enum AttrOrInput {
    Attr(Arc<Tensor>),
    Input(usize),
}
#[derive(Clone)]
pub struct TypedFact;
pub trait SpecialOps<O> {
    fn create_dummy(&self) -> O;
    fn create_source(&self, fact: TypedFact) -> O;
    fn wire_node(&mut self, op: impl Into<O>, inputs: &[OutletId]) -> TractResult<Vec<OutletId>>;
}
#[derive(Clone)]
pub struct Graph<O>
where
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub nodes: Vec<Node<O>>,
    pub inputs: Vec<OutletId>,
    pub outputs: Vec<OutletId>,
}
impl<O> Default for Graph<O>
where
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn default() -> Graph<O> {
        Graph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
        }
    }
}
impl<O> Graph<O>
where
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Graph<O>: SpecialOps<O>,
{
    pub fn add_source(&mut self, fact: TypedFact) -> TractResult<OutletId> {
        let source = self.create_source(fact.clone());
        let id = self.add_node(source, vec![fact])?;
        let id = OutletId::new(id, 0);
        self.inputs.push(id);
        Ok(id)
    }
}
impl<O> Graph<O>
where
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub fn add_node(
        &mut self,
        op: impl Into<O>,
        output_facts: Vec<TypedFact>,
    ) -> TractResult<usize> {
        let op = op.into();
        let id = self.nodes.len();
        let outputs = output_facts
            .into_iter()
            .map(|_fact| Outlet { successors: vec![] })
            .collect();
        let node = Node {
            id,
            op,
            inputs: vec![],
            outputs,
        };
        self.nodes.push(node);
        Ok(id)
    }
    pub fn add_edge(&mut self, outlet: OutletId, inlet: InletId) -> TractResult<()> {
        {
            let prec = &mut self.nodes[outlet.node];
            prec.outputs[outlet.slot].successors.push(inlet);
        }
        let succ = &mut self.nodes[inlet.node];
        if inlet.slot == succ.inputs.len() {
            succ.inputs.push(outlet);
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
    pub fn outlet_fact(&self, outlet: OutletId) -> TractResult<&TypedFact> {
        let outlets = &self.nodes[outlet.node].outputs;
        Ok(outlets.get(outlet.slot).map(|o| &TypedFact).unwrap())
    }
}
impl<O> Graph<O>
where
    O: From<Const> + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub fn add_const(&mut self, v: Arc<Tensor>) -> TractResult<OutletId> {
        self.add_node(Const, vec![TypedFact]).map(|id| id.into())
    }
}
#[derive(Clone)]
pub struct Node<O> {
    pub id: usize,
    pub inputs: Vec<OutletId>,
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub op: O,
    pub outputs: Vec<Outlet>,
}
#[derive(Clone, Default)]
pub struct Outlet {
    pub successors: Vec<InletId>,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutletId {
    pub node: usize,
    pub slot: usize,
}
impl OutletId {
    pub fn new(node: usize, slot: usize) -> OutletId {
        OutletId { node, slot }
    }
}
impl From<usize> for OutletId {
    fn from(node: usize) -> OutletId {
        OutletId::new(node, 0)
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub struct InletId {
    pub node: usize,
    pub slot: usize,
}
use std::ops::{Deref, DerefMut};
#[derive(Clone)]
pub struct ModelPatch<O>
where
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub model: Graph<O>,
    pub inputs: HashMap<usize, usize>,
    pub incoming: HashMap<OutletId, OutletId>,
    pub shunt_outlet_by: HashMap<OutletId, OutletId>,
    pub obliterate: Vec<usize>,
}
impl<O> Default for ModelPatch<O>
where
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn default() -> ModelPatch<O> {
        ModelPatch {
            model: Graph::default(),
            inputs: HashMap::default(),
            incoming: HashMap::new(),
            shunt_outlet_by: HashMap::new(),
            obliterate: vec![],
        }
    }
}
impl<O> Deref for ModelPatch<O>
where
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    type Target = Graph<O>;
    fn deref(&self) -> &Graph<O> {
        unimplemented!()
    }
}
impl<O> DerefMut for ModelPatch<O>
where
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn deref_mut(&mut self) -> &mut Graph<O> {
        &mut self.model
    }
}
impl<O> ModelPatch<O>
where
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Graph<O>: SpecialOps<O>,
{
    pub fn tap_model(&mut self, model: &Graph<O>, outlet: OutletId) -> TractResult<OutletId> {
        let fact = model.outlet_fact(outlet)?;
        let id = self.add_source(TypedFact)?;
        self.incoming.insert(id, outlet);
        Ok(id)
    }
    pub fn shunt_outside(&mut self, outlet: OutletId, by: OutletId) -> TractResult<()> {
        self.shunt_outlet_by.insert(outlet, by);
        Ok(())
    }
    pub fn obliterate(&mut self, node: usize) -> TractResult<()> {
        self.obliterate.push(node);
        Ok(())
    }
    pub fn replace_single_op<IO: Into<O>>(
        patched_model: &Graph<O>,
        node: &Node<O>,
        inputs: &[OutletId],
        new_op: IO,
    ) -> TractResult<ModelPatch<O>> {
        let mut patch = ModelPatch::default();
        let new_op = new_op.into();
        let inputs = inputs
            .iter()
            .map(|i| patch.tap_model(patched_model, *i))
            .collect::<TractResult<Vec<_>>>()?;
        let wires = patch.wire_node(new_op, &inputs)?;
        for (ix, o) in wires.iter().enumerate() {
            patch.shunt_outside(OutletId::new(node.id, ix), *o)?;
        }
        patch.obliterate(node.id)?;
        Ok(patch)
    }
    pub fn apply(self, target: &mut Graph<O>) -> TractResult<()> {
        let prior_target_inputs = target.input_outlets()?.len();
        let prior_target_outputs = target.output_outlets()?.len();
        let ModelPatch {
            model: patch,
            incoming: mut mapping,
            shunt_outlet_by,
            obliterate,
            ..
        } = self;
        let mut all_inputs = HashMap::new();
        let model_input_outlets = target.input_outlets()?.to_vec();
        for node in patch.nodes {
            if model_input_outlets.contains(&OutletId::new(node.id, 0)) && mapping.contains_key(&OutletId::new(node.id, 0)) {
                continue;
            }
            let Node {
                id: patch_node_id,
                inputs,
                op,
                outputs,
            } = node;
            let n_outputs = outputs.len();
            let facts = outputs.into_iter().map(|of| TypedFact).collect();
            let added_node_id = target.add_node(op, facts)?;
            for ix in 0..n_outputs {
                mapping.insert(
                    OutletId::new(patch_node_id, ix),
                    OutletId::new(added_node_id, ix),
                );
            }
            all_inputs.insert(added_node_id, inputs);
        }
        for (outlet, by) in shunt_outlet_by {
            let replace_by = mapping[&by];
            for o in target.outputs.iter_mut() {
                if *o == outlet {
                    *o = replace_by;
                }
            }
        }
        for (node, inputs) in all_inputs {
            for (ix, input) in inputs.into_iter().enumerate() {
                target.add_edge(mapping[&input], InletId { node, slot: ix })?;
            }
        }
        for node in obliterate {
            target.nodes[node].op = target.create_dummy();
        }
        target.set_input_outlets(&model_input_outlets)?;
        Ok(())
    }
}
pub type TypedModel = Graph<Box<dyn TypedOp>>;
pub type TypedNode = Node<Box<dyn TypedOp>>;
pub type TypedModelPatch = ModelPatch<Box<dyn TypedOp>>;
impl SpecialOps<Box<dyn TypedOp>> for TypedModel {
    fn create_dummy(&self) -> Box<dyn TypedOp> {
        Box::new(Const)
    }
    fn create_source(&self, _fact: TypedFact) -> Box<dyn TypedOp> {
        Box::new(TypedSource {})
    }
    fn wire_node(
        &mut self,
        op: impl Into<Box<dyn TypedOp>>,
        inputs: &[OutletId],
    ) -> TractResult<Vec<OutletId>> {
        let op = op.into();
        {
            let output_facts = vec![TypedFact];
            let id = self.add_node(&op, output_facts)?;
            inputs
                .iter()
                .enumerate()
                .try_for_each(|(ix, i)| self.add_edge(*i, InletId { node: id, slot: ix }))?;
            TractResult::Ok(
                self.nodes[id]
                    .outputs
                    .iter()
                    .enumerate()
                    .map(|(ix, _)| OutletId::new(id, ix))
                    .collect(),
            )
        }
    }
}
pub use std::collections::HashMap;
#[test]
fn crasher_monterey_matmul() {
    let mut model = TypedModel::default();
    let source = model.add_source(TypedFact).unwrap();
    let a = model.add_const(Arc::new(Tensor)).unwrap();
    let mm = model.wire_node(MatMul {}, &[a, source]).unwrap()[0];
    model.set_output_outlets(&[mm]).unwrap();
    let patch = TypedModelPatch::replace_single_op(
        &model,
        &model.nodes[mm.node],
        &[source],
        MatMulUnary {
            a: Arc::new(Tensor),
        },
    )
    .unwrap();
    patch.apply(&mut model).unwrap();
    let wire = model.outputs[0];
    let patch = model.nodes[wire.node]
        .op
        .codegen(&model, &model.nodes[wire.node])
        .unwrap()
        .unwrap();
}

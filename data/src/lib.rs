type TractResult<T> = Result<T, ()>;
use ndarray::*;
use std::sync::Arc;
#[derive(Clone)]
struct Const;
#[derive(Copy, Clone)]
enum BinOp {
    Min,
    Max,
}
#[derive(Clone, Copy)]
enum OutputStoreSpec {
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
enum ProtoFusedSpec {
    BinScalar(AttrOrInput, BinOp),
    BinPerRow(AttrOrInput, BinOp),
    BinPerCol(AttrOrInput, BinOp),
    AddRowColProducts(AttrOrInput, AttrOrInput),
    AddUnicast(OutputStoreSpec, AttrOrInput),
    Store,
}
#[derive(Clone)]
struct MatMulUnary {}
impl TypedOp for MatMulUnary {}
#[derive(Clone)]
struct TypedSource {}
impl TypedOp for TypedSource {}
trait TypedOp: dyn_clone::DynClone + Send + Sync + 'static {}
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
#[derive(Clone)]
enum AttrOrInput {
    Attr(Arc<()>),
    Input(usize),
}
trait SpecialOps<O> {
    fn create_source(&self) -> O;
    fn wire_node(&mut self, op: impl Into<O>, inputs: &[OutletId]) -> TractResult<Vec<OutletId>>;
}
#[derive(Clone)]
struct Graph<O>
where
    O: AsRef<dyn TypedOp> + AsMut<dyn TypedOp> + Clone + 'static,
{
    pub nodes: Vec<Node<O>>,
    pub inputs: Vec<OutletId>,
    pub outputs: Vec<OutletId>,
}
impl<O> Default for Graph<O>
where
    O: AsRef<dyn TypedOp> + AsMut<dyn TypedOp> + Clone + 'static,
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
    O: AsRef<dyn TypedOp> + AsMut<dyn TypedOp> + Clone + 'static,
    Graph<O>: SpecialOps<O>,
{
    pub fn add_source(&mut self) -> TractResult<OutletId> {
        let source = self.create_source();
        let id = self.add_node(source)?;
        let id = OutletId::new(id, 0);
        self.inputs.push(id);
        Ok(id)
    }
}
impl<O> Graph<O>
where
    O: AsRef<dyn TypedOp> + AsMut<dyn TypedOp> + Clone + 'static,
{
    pub fn add_node(&mut self, op: impl Into<O>) -> TractResult<usize> {
        let op = op.into();
        let id = self.nodes.len();
        let outputs = vec![Outlet { successors: vec![] }];
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
}
#[derive(Clone)]
struct Node<O> {
    pub id: usize,
    pub inputs: Vec<OutletId>,
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub op: O,
    pub outputs: Vec<Outlet>,
}
#[derive(Clone, Default)]
struct Outlet {
    pub successors: Vec<InletId>,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OutletId {
    pub node: usize,
    pub slot: usize,
}
impl OutletId {
    pub fn new(node: usize, slot: usize) -> OutletId {
        OutletId { node, slot }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
struct InletId {
    pub node: usize,
    pub slot: usize,
}
use std::ops::{Deref, DerefMut};
#[derive(Clone)]
struct ModelPatch<O>
where
    O: AsRef<dyn TypedOp> + AsMut<dyn TypedOp> + Clone + 'static,
{
    pub model: Graph<O>,
    pub inputs: HashMap<usize, usize>,
    pub incoming: HashMap<OutletId, OutletId>,
    pub shunt_outlet_by: HashMap<OutletId, OutletId>,
}
impl<O> Default for ModelPatch<O>
where
    O: AsRef<dyn TypedOp> + AsMut<dyn TypedOp> + Clone + 'static,
{
    fn default() -> ModelPatch<O> {
        ModelPatch {
            model: Graph::default(),
            inputs: HashMap::default(),
            incoming: HashMap::new(),
            shunt_outlet_by: HashMap::new(),
        }
    }
}
impl<O> Deref for ModelPatch<O>
where
    O: AsRef<dyn TypedOp> + AsMut<dyn TypedOp> + Clone + 'static,
{
    type Target = Graph<O>;
    fn deref(&self) -> &Graph<O> {
        unimplemented!()
    }
}
impl<O> DerefMut for ModelPatch<O>
where
    O: AsRef<dyn TypedOp> + AsMut<dyn TypedOp> + Clone + 'static,
{
    fn deref_mut(&mut self) -> &mut Graph<O> {
        &mut self.model
    }
}
impl<O> ModelPatch<O>
where
    O: AsRef<dyn TypedOp> + AsMut<dyn TypedOp> + Clone + 'static,
    Graph<O>: SpecialOps<O>,
{
    pub fn tap_model(&mut self, model: &Graph<O>, outlet: OutletId) -> TractResult<OutletId> {
        let id = self.add_source()?;
        self.incoming.insert(id, outlet);
        Ok(id)
    }
    pub fn shunt_outside(&mut self, outlet: OutletId, by: OutletId) -> TractResult<()> {
        self.shunt_outlet_by.insert(outlet, by);
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
        Ok(patch)
    }
    pub fn apply(self, target: &mut Graph<O>) -> TractResult<()> {
        let ModelPatch {
            model: patch,
            incoming: mut mapping,
            shunt_outlet_by,
            ..
        } = self;
        let mut all_inputs = HashMap::new();
        let model_input_outlets = target.inputs.clone();
        for node in patch.nodes {
            if target.inputs.contains(&OutletId::new(node.id, 0))
                && mapping.contains_key(&OutletId::new(node.id, 0))
            {
                continue;
            }
            let Node {
                id: patch_node_id,
                inputs,
                op,
                outputs,
            } = node;
            let n_outputs = outputs.len();
            let added_node_id = target.add_node(op)?;
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
        target.inputs = model_input_outlets;
        Ok(())
    }
}
type TypedModel = Graph<Box<dyn TypedOp>>;
type TypedModelPatch = ModelPatch<Box<dyn TypedOp>>;
impl SpecialOps<Box<dyn TypedOp>> for TypedModel {
    fn create_source(&self) -> Box<dyn TypedOp> {
        Box::new(TypedSource {})
    }
    fn wire_node(
        &mut self,
        op: impl Into<Box<dyn TypedOp>>,
        inputs: &[OutletId],
    ) -> TractResult<Vec<OutletId>> {
        let op = op.into();
        {
            let id = self.add_node(&op)?;
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
use std::collections::HashMap;
#[test]
fn crasher_monterey_matmul() {
    let mut model = TypedModel::default();
    let source = model.add_source().unwrap();
    let mm = model.wire_node(MatMulUnary {}, &[source]).unwrap()[0];
    model.outputs = vec![mm];
    let patch = TypedModelPatch::replace_single_op(
        &model,
        &model.nodes[mm.node],
        &[source],
        MatMulUnary {},
    )
    .unwrap();
    patch.apply(&mut model).unwrap();
    let wire = model.outputs[0];
    let packed_as =
        Array::from_shape_fn(vec![1, 1], |_| (Arc::new(()), vec![ProtoFusedSpec::Store]));
    packed_as.clone();
}

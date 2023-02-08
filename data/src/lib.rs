pub type TractError = anyhow::Error;
pub type TractResult<T> = anyhow::Result<T>;
use ndarray::*;
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}
impl Tensor {
    pub fn zero(shape: &[usize]) -> Tensor {
        let data = vec![0.0f32; shape.iter().cloned().product::<usize>()];
        Tensor {
            shape: shape.into(),
            data,
        }
    }
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn into_arc_tensor(self) -> Arc<Tensor> {
        Arc::new(self)
    }
}
use std::sync::Arc;
macro_rules! as_op {
    () => {
        fn as_op(&self) -> &dyn Op {
            self
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    };
}
#[derive(Clone, Default)]
pub struct Dummy;
impl Op for Dummy {}
impl TypedOp for Dummy {
    as_op!();
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<Vec<TypedFact>> {
        unimplemented!()
    }
}
#[derive(Clone)]
pub struct Const(pub Arc<Tensor>);
impl Op for Const {}
impl TypedOp for Const {
    as_op!();
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<Vec<TypedFact>> {
        unimplemented!()
    }
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
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<Vec<TypedFact>> {
        Ok(vec![fact([1, 2])])
    }
    as_op!();
}
#[derive(Clone)]
pub struct MatMul {}
impl Op for MatMul {}
impl TypedOp for MatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TypedFact>> {
        let (_m, _k, _n, c_shape) = compute_shape(&inputs[0].shape, &inputs[1].shape)?;
        Ok(vec![fact(c_shape)])
    }
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let konst = model.outlet_fact(node.inputs[0])?.konst.clone().unwrap();
        TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs[1..2],
            MatMulUnary { a: konst },
        )
        .map(Some)
    }
    as_op!();
}
#[derive(Clone)]
pub struct MatMulUnary {
    pub a: Arc<Tensor>,
}
impl Op for MatMulUnary {}
impl TypedOp for MatMulUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TypedFact>> {
        let (_m, _k, _n, c_shape) = compute_shape(&self.a.shape(), &inputs[0].shape)?;
        Ok(vec![fact(c_shape)])
    }
    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let patch = self.new_mat_mul_unary_finite(model, node)?;
        Ok(Some(patch))
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
            let pa = Tensor::zero(&[64]);
            (pa.into_arc_tensor(), vec![ProtoFusedSpec::Store])
        });
        wire = patch.wire_node(format!("{}.pack", &*node.name), MatMatMulPack {}, &[wire])?[0];
        let op = LirMatMulUnary {
            micro_ops: packed_as,
        };
        patch.wire_node(format!("{}.matmatmul", &*node.name), op, &[wire])?[0];
        Ok(patch)
    }
}
#[derive(Clone, PartialEq, Eq)]
pub struct MatMatMulPack {}
impl Op for MatMatMulPack {}
impl TypedOp for MatMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TypedFact>> {
        Ok(vec![fact([1])])
    }
    as_op!();
}
pub fn compute_shape(
    ashape: &[usize],
    bshape: &[usize],
) -> TractResult<(usize, usize, usize, Vec<usize>)> {
    let (m, k) = (ashape[0], ashape[1]);
    let n = bshape[1];
    Ok((m, k, n, vec!(n, m)))
}
#[derive(Clone)]
pub struct TypedSource {}
impl Op for TypedSource {}
impl TypedOp for TypedSource {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<Vec<TypedFact>> {
        unimplemented!()
    }
    as_op!();
}
use std::any::Any;
pub trait Op: dyn_clone::DynClone + Send + Sync + 'static {}
pub trait TypedOp: Op + dyn_clone::DynClone + Send + Sync + 'static {
    fn as_op(&self) -> &dyn Op;
    fn as_any(&self) -> &dyn Any;
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TypedFact>>;
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
pub struct ShapeFact {
    dims: Vec<usize>,
}
impl ShapeFact {
    pub fn from_dims<T: IntoIterator<Item = usize>>(it: T) -> ShapeFact {
        let mut dims = ShapeFact {
            dims: it.into_iter().collect(),
        };
        dims
    }
}
impl std::ops::Deref for ShapeFact {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        &self.dims
    }
}
impl<T: IntoIterator<Item = usize>> From<T> for ShapeFact {
    fn from(it: T) -> ShapeFact {
        ShapeFact::from_dims(it)
    }
}
#[derive(Clone)]
pub struct TypedFact {
    pub shape: ShapeFact,
    pub konst: Option<Arc<Tensor>>,
}
impl TypedFact {
    pub fn dt_shape<S>(shape: S) -> TypedFact
    where
        S: Into<ShapeFact>,
    {
        TypedFact {
            shape: shape.into(),
            konst: None,
        }
    }
}
impl From<Arc<Tensor>> for TypedFact {
    fn from(t: Arc<Tensor>) -> TypedFact {
        TypedFact {
            shape: ShapeFact::from_dims(t.shape().to_vec()),
            konst: Some(t),
        }
    }
}
impl<'a> From<&'a TypedFact> for TypedFact {
    fn from(fact: &TypedFact) -> TypedFact {
        fact.clone()
    }
}
fn fact<S>(shape: S) -> TypedFact
where
    S: Into<ShapeFact>,
{
    TypedFact::dt_shape(shape)
}
pub trait SpecialOps<F, O> {
    fn create_dummy(&self) -> O;
    fn create_source(&self, fact: F) -> O;
    fn is_source(op: &O) -> bool;
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        inputs: &[OutletId],
    ) -> TractResult<Vec<OutletId>>;
}
#[derive(Clone)]
pub struct Graph<F, O>
where
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub nodes: Vec<Node<F, O>>,
    pub inputs: Vec<OutletId>,
    pub outputs: Vec<OutletId>,
}
impl<F, O> Default for Graph<F, O>
where
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn default() -> Graph<F, O> {
        Graph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
        }
    }
}
impl<F, O> Graph<F, O>
where
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Graph<F, O>: SpecialOps<F, O>,
{
    pub fn add_source(&mut self, name: impl Into<String>, fact: F) -> TractResult<OutletId> {
        let source = self.create_source(fact.clone());
        let id = self.add_node(name, source, vec![fact])?;
        let id = OutletId::new(id, 0);
        self.inputs.push(id);
        Ok(id)
    }
}
impl<F, O> Graph<F, O>
where
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        output_facts: Vec<F>,
    ) -> TractResult<usize> {
        let op = op.into();
        let name = name.into();
        let id = self.nodes.len();
        let outputs = output_facts
            .into_iter()
            .map(|fact| Outlet {
                fact,
                successors: vec![],
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
    pub fn outlet_fact(&self, outlet: OutletId) -> TractResult<&F> {
        let outlets = &self.nodes[outlet.node].outputs;
        Ok(outlets.get(outlet.slot).map(|o| &o.fact).unwrap())
    }
    pub fn eval_order(&self) -> TractResult<Vec<usize>> {
        eval_order(self)
    }
}
impl<F: Clone + 'static, O> Graph<F, O>
where
    F: Clone + 'static + From<std::sync::Arc<Tensor>>,
    O: From<Const> + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub fn add_const(&mut self, name: impl Into<String>, v: Arc<Tensor>) -> TractResult<OutletId> {
        let fact = F::from(v.clone());
        let name = name.into();
        self.add_node(name, Const(v), vec![fact])
            .map(|id| id.into())
    }
}
impl<F, O> Graph<F, O>
where
    F: Clone + 'static + for<'a> std::convert::From<&'a F>,
    O: Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + for<'a> std::convert::From<&'a O>,
    Graph<F, O>: SpecialOps<F, O>,
{
    pub fn compact(&mut self) -> TractResult<()> {
        let mut result = IntoTranslator.translate_model(self)?;
        std::mem::swap(self, &mut result);
        Ok(())
    }
}
#[derive(Clone)]
pub struct Node<F, O> {
    pub id: usize,
    pub name: String,
    pub inputs: Vec<OutletId>,
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub op: O,
    pub outputs: Vec<Outlet<F>>,
}
#[derive(Clone, Default)]
pub struct Outlet<F> {
    pub fact: F,
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
impl From<(usize, usize)> for OutletId {
    fn from(pair: (usize, usize)) -> OutletId {
        OutletId::new(pair.0, pair.1)
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub struct InletId {
    pub node: usize,
    pub slot: usize,
}
pub fn eval_order<F, O>(model: &Graph<F, O>) -> TractResult<Vec<usize>>
where
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
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
    eval_order_for_nodes(&model.nodes, &inputs, &targets, &[])
}
pub fn eval_order_for_nodes<F, O>(
    nodes: &[Node<F, O>],
    model_inputs: &[usize],
    model_outputs: &[usize],
    more_dependencies: &[(usize, usize)],
) -> TractResult<Vec<usize>>
where
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
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
use std::ops::{Deref, DerefMut};
#[derive(Clone)]
pub struct ModelPatch<F, O>
where
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub model: Graph<F, O>,
    pub inputs: HashMap<usize, usize>,
    pub incoming: HashMap<OutletId, OutletId>,
    pub shunt_outlet_by: HashMap<OutletId, OutletId>,
    pub obliterate: Vec<usize>,
}
impl<F, O> Default for ModelPatch<F, O>
where
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn default() -> ModelPatch<F, O> {
        ModelPatch {
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
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    type Target = Graph<F, O>;
    fn deref(&self) -> &Graph<F, O> {
        unimplemented!()
    }
}
impl<F, O> DerefMut for ModelPatch<F, O>
where
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn deref_mut(&mut self) -> &mut Graph<F, O> {
        &mut self.model
    }
}
impl<F, O> ModelPatch<F, O>
where
    F: Clone + 'static,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Graph<F, O>: SpecialOps<F, O>,
{
    pub fn tap_model(&mut self, model: &Graph<F, O>, outlet: OutletId) -> TractResult<OutletId> {
        let fact = model.outlet_fact(outlet)?;
        let id = self.add_source(
            format!("incoming-{}/{}", outlet.node, outlet.slot),
            dyn_clone::clone(fact),
        )?;
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
            .collect::<TractResult<Vec<_>>>()?;
        let wires = patch.wire_node(&node.name, new_op, &inputs)?;
        for (ix, o) in wires.iter().enumerate() {
            patch.shunt_outside(OutletId::new(node.id, ix), *o)?;
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
            ..
        } = self;
        let mut all_inputs = HashMap::new();
        let model_input_outlets = target.input_outlets()?.to_vec();
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
            let facts = outputs.into_iter().map(|of| of.fact).collect();
            let added_node_id = target.add_node(name, op, facts)?;
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
pub trait Translate<TI1, O1, TI2, O2>
where
    TI1: Clone + 'static,
    TI2: Clone + 'static,
    O1: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    O2: AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn translate_node(
        &self,
        source: &Graph<TI1, O1>,
        node: &Node<TI1, O1>,
        target: &mut Graph<TI2, O2>,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<Vec<OutletId>>;
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
            let node = &source.nodes[old_id];
            let outlets = self.translate_node(source, node, &mut target, &mapping)?;
            for (ix, outlet) in outlets.into_iter().enumerate() {
                mapping.insert(OutletId::new(node.id, ix), outlet);
            }
        }
        target.inputs = source.input_outlets()?.iter().map(|i| mapping[i]).collect();
        target.outputs = source
            .output_outlets()?
            .iter()
            .map(|o| mapping[o])
            .collect();
        Ok((target, mapping))
    }
}
pub struct IntoTranslator;
impl<TI1, O1, TI2, O2, EO, ETI> Translate<TI1, O1, TI2, O2> for IntoTranslator
where
    TractError: From<EO> + From<ETI>,
    TI1: Clone + 'static,
    TI2: for<'a> TryFrom<&'a TI1, Error = EO> + Clone + 'static,
    O1: Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    O2: for<'a> TryFrom<&'a O1, Error = ETI> + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Graph<TI2, O2>: SpecialOps<TI2, O2>,
{
    fn translate_node(
        &self,
        source: &Graph<TI1, O1>,
        node: &Node<TI1, O1>,
        target: &mut Graph<TI2, O2>,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<Vec<OutletId>> {
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
                .collect::<TractResult<Vec<_>>>()?;
            let new_id = target.add_node(node.name.clone(), new_op, facts)?;
            for (ix, o) in node.inputs.iter().enumerate() {
                target.add_edge(
                    mapping[o],
                    InletId {
                        node: new_id,
                        slot: ix,
                    },
                )?
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
pub type TypedModel = Graph<TypedFact, Box<dyn TypedOp>>;
pub type TypedNode = Node<TypedFact, Box<dyn TypedOp>>;
pub type TypedModelPatch = ModelPatch<TypedFact, Box<dyn TypedOp>>;
impl SpecialOps<TypedFact, Box<dyn TypedOp>> for TypedModel {
    fn is_source(op: &Box<dyn TypedOp>) -> bool {
        op.as_any().downcast_ref::<TypedSource>().is_some()
    }
    fn create_dummy(&self) -> Box<dyn TypedOp> {
        Box::new(Dummy::default())
    }
    fn create_source(&self, _fact: TypedFact) -> Box<dyn TypedOp> {
        Box::new(TypedSource {})
    }
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn TypedOp>>,
        inputs: &[OutletId],
    ) -> TractResult<Vec<OutletId>> {
        let op = op.into();
        let name = name.into();
        {
            let output_facts = || -> TractResult<Vec<TypedFact>> {
                let input_facts = inputs
                    .iter()
                    .map(|o| self.outlet_fact(*o))
                    .collect::<TractResult<Vec<_>>>()?;
                let facts = op.output_facts(&input_facts)?;
                Ok(facts)
            };
            let output_facts = output_facts()?;
            let id = self.add_node(&name, &op, output_facts)?;
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
    let wire = model.add_source("input", fact([1usize, 1])).unwrap();
    let a = model
        .add_const("a", Tensor::zero(&[2, 1]).into_arc_tensor())
        .unwrap();
    let wire = model.wire_node("conv", MatMul {}, &[a, wire]).unwrap()[0];
    model.set_output_outlets(&[wire]).unwrap();
    let patch = model
        .nodes[wire.node]
        .op
        .declutter(&model, &model.nodes[wire.node])
        .unwrap()
        .unwrap();
    patch.apply(&mut model).unwrap();
    model.compact().unwrap();
    let wire = model.outputs[0];
    let patch = model
        .nodes[wire.node]
        .op
        .codegen(&model, &model.nodes[wire.node])
        .unwrap()
        .unwrap();
}

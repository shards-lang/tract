use crate::internal::*;
use crate::model::*;
use crate::ops;
use crate::optim::OptimizerSession;

/// A model with completely determined types and shapes.
pub type TypedModel = Graph<TypedFact, Box<dyn TypedOp>>;
/// Node for TypedModel graph
pub type TypedNode = Node<TypedFact, Box<dyn TypedOp>>;
/// A ModelPatch for TypedModel.
pub type TypedModelPatch = ModelPatch<TypedFact, Box<dyn TypedOp>>;

impl SpecialOps<TypedFact, Box<dyn TypedOp>> for TypedModel {
    fn is_source(op: &Box<dyn TypedOp>) -> bool {
        op.as_op().downcast_ref::<ops::source::TypedSource>().is_some()
    }

    fn create_dummy(&self) -> Box<dyn TypedOp> {
        Box::new(crate::ops::dummy::Dummy::new())
    }

    fn create_source(&self, fact: TypedFact) -> Box<dyn TypedOp> {
        Box::new(crate::ops::source::TypedSource::new(fact))
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
                let facts = op.output_facts(&input_facts).context("in output_facts invocation")?;
                if input_facts.iter().all(|f| f.konst.is_some()) && op.is_stateless() {
                    let tensors = input_facts
                        .iter()
                        .map(|f| f.konst.clone().unwrap().into_tvalue())
                        .collect::<TVec<_>>();
                    if let Ok(outputs) = op.eval(tensors) {
                        return Ok(outputs.into_iter().map(|t| TypedFact::from(&*t)).collect());
                    }
                }
                Ok(facts)
            };

            let output_facts = output_facts()
                .with_context(|| format!("wiring {name} ({op:?}), determining output_facts"))?;
            let id = self.add_node(&name, &op, output_facts)?;
            inputs
                .iter()
                .enumerate()
                .try_for_each(|(ix, i)| self.add_edge(*i, InletId::new(id, ix)))?;
            TractResult::Ok(
                self.node(id)
                    .outputs
                    .iter()
                    .enumerate()
                    .map(|(ix, _)| OutletId::new(id, ix))
                    .collect(),
            )
        }
        .with_context(|| format!("Wiring node \"{name}\", {op:?}"))
    }
}

impl TypedModel {
    pub fn signature(&self) -> u64 {
        use std::hash::Hasher;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn into_optimized(mut self) -> TractResult<TypedModel> {
        self.declutter()?;
        self.optimize()?;
        Ok(self)
    }
    #[cfg(not(all(debug_assertions, feature = "paranoid_assertions")))]
    #[inline]
    pub fn check_consistency(&self) -> TractResult<()> {
        Ok(())
    }

    #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
    pub fn check_consistency(&self) -> TractResult<()> {
        self.check_edges()?;
        for node_id in &self.eval_order()? {
            let input_facts = self.node_input_facts(*node_id)?;
            let node = &self.nodes[*node_id];
            if node.id != *node_id {
                bail!("Node at position {} has id {}", node_id, node.id);
            }
            let output_facts = node.op.output_facts(&input_facts)?;
            if node.outputs.len() != output_facts.len() {
                bail!(
                    "Inconsistent model, node output count mismatch. Op says {}, node says {}. {}",
                    output_facts.len(),
                    node.outputs.len(),
                    node
                );
            }
            if node
                .outputs
                .iter()
                .map(|o| &o.fact)
                .zip(output_facts.iter())
                .any(|(a, b)| a.datum_type != b.datum_type || a.shape != b.shape)
            {
                bail!(
                            "Inconsistent model, output types mismatch. Op says: {:?}, node says: {:?}. {} with inputs {:?}. {}",
                            output_facts, node.outputs.iter().map(|o| &o.fact).collect::<Vec<_>>(), node, input_facts, node)
            }
        }
        for node in &self.nodes {
            for (ix, output) in node.outputs.iter().enumerate() {
                output.fact.consistent().with_context(|| {
                    format!("Inconsistent fact {:?}: {:?}", OutletId::new(node.id, ix), output.fact)
                })?
            }
        }
        Ok(())
    }

    pub fn into_decluttered(mut self) -> TractResult<TypedModel> {
        self.declutter()?;
        Ok(self)
    }

    /// Perform declutter passes on the network.
    pub fn declutter(&mut self) -> TractResult<()> {
        crate::optim::Optimizer::declutter().session().optimize(self)
    }

    /// Perform optimization passes on the model, using a given optimizer session.
    pub fn optimize_with_session(&mut self, session: &mut OptimizerSession) -> TractResult<()> {
        session.optimize(self)
    }

    /// Translate the graph to locally optimized operators (LIR or MIR ops).
    pub fn optimize(&mut self) -> TractResult<()> {
        crate::optim::Optimizer::codegen().optimize(self)
    }
}


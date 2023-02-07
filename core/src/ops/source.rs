use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct SourceState(pub usize);

#[derive(Debug, Clone, new, Hash)]
pub struct TypedSource {
    pub fact: TypedFact,
}

impl_dyn_hash!(TypedSource);

impl Op for TypedSource {
    fn name(&self) -> Cow<str> {
        "Source".into()
    }
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

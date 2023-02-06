use std::borrow::Borrow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::internal::*;
use crate::model::order::eval_order_for_nodes;
use crate::model::{Fact, Graph, OutletId};

#[derive(Default)]
pub struct SessionState {
    pub inputs: HashMap<usize, TValue>,
    pub resolved_symbols: SymbolValues,
    pub tensors: HashMap<String, Tensor>,
    pub cached_mmm_scratch_space: Option<Box<dyn tract_linalg::mmm::ScratchSpace>>,
}

impl Clone for SessionState {
    fn clone(&self) -> Self {
        SessionState {
            inputs: self.inputs.clone(),
            resolved_symbols: self.resolved_symbols.clone(),
            tensors: self.tensors.clone(),
            cached_mmm_scratch_space: None,
        }
    }
}

impl Debug for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SessionState({:?})", self.resolved_symbols)
    }
}


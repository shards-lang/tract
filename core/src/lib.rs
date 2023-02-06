#![allow(clippy::len_zero)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::redundant_closure_call)]

pub extern crate anyhow;
extern crate bit_set;
#[macro_use]
extern crate derive_new;
#[macro_use]
pub extern crate downcast_rs;
#[macro_use]
extern crate educe;
#[allow(unused_imports)]
#[macro_use]
extern crate log;
#[allow(unused_imports)]
#[macro_use]
pub extern crate ndarray;
#[cfg(test)]
extern crate env_logger;
pub extern crate num_traits;

pub extern crate tract_data;
pub extern crate tract_linalg;

#[macro_use]
pub mod macros;
#[macro_use]
pub mod ops;

mod broadcast;
mod hash;
mod late_bind;
pub mod model;
pub mod optim;
pub mod plan;
pub mod value;

pub use dyn_clone;

/// This prelude is meant for code using tract.
pub mod prelude {
    pub use crate::model::*;
    pub use crate::value::{IntoTValue, TValue};
    pub use std::sync::Arc;
    pub use tract_data::prelude::*;

    pub use ndarray as tract_ndarray;
    pub use num_traits as tract_num_traits;
    pub use tract_data;
    pub use tract_linalg;
}

/// This prelude is meant for code extending tract (like implementing new ops).
pub mod internal {
    pub use crate::hash::{hash_f32, hash_opt_f32, SloppyHash};
    pub use crate::late_bind::*;
    pub use crate::model::*;
    pub use crate::ops::{ AttrOrInput, EvalOp, Op, OpState };
    pub use crate::plan::SessionState;
    pub use crate::prelude::*;
    pub use anyhow::{anyhow, bail, ensure, format_err, Context as TractErrorContext};
    pub use dims;
    pub use downcast_rs as tract_downcast_rs;
    pub use std::borrow::Cow;
    pub use std::collections::HashMap;
    pub use std::hash::Hash;
    pub use std::marker::PhantomData;
    pub use tract_data::internal::*;
    pub use tract_data::{
        dispatch_copy, dispatch_datum, dispatch_datum_by_size, dispatch_floatlike, dispatch_numbers,
    };
    pub use tvec;
    pub use {args_1, args_2, args_3, args_4, args_5, args_6, args_7, args_8};
    pub use {as_op, impl_op_same_as, not_a_typed_op, op_as_typed_op};
//    pub use {bin_to_super_type, element_wise, element_wise_oop};
}

#[cfg(test)]
#[allow(dead_code)]
fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

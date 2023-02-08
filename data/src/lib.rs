#![allow(clippy::len_zero)]
#![allow(clippy::missing_safety_doc)]
pub type TractError = anyhow::Error;
pub type TractResult<T> = anyhow::Result<T>;
pub mod prelude {
    pub use crate::datum::{round_ties_to_even, Blob, Datum, DatumType, QParams};
    pub use crate::dim::{Symbol, SymbolTable, SymbolValues, TDim, ToDim};
    pub use crate::tensor::{IntoArcTensor, IntoTensor, Tensor};
    pub use crate::{TractError, TractResult};
}
pub mod internal {
    pub use crate::dim::DimLike;
    pub use crate::prelude::*;
}
mod datum {
    use crate::dim::TDim;
    use crate::tensor::litteral::*;
    use crate::tensor::Tensor;
    use num_traits::AsPrimitive;
    use std::hash::Hash;
    use std::{fmt, ops};
    #[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
    pub struct Blob(pub Vec<u8>);
    impl fmt::Display for Blob {
        fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
            unimplemented!()
        }
    }
    #[derive(Copy, Clone, PartialEq)]
    pub enum QParams {
        MinMax { min: f32, max: f32 },
        ZpScale { zero_point: i32, scale: f32 },
    }
    impl Eq for QParams {}
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
    pub enum DatumType {
        I8,
        I16,
        I32,
        I64,
        F32,
        TDim,
        Blob,
        String,
    }
    impl DatumType {
        pub fn is_unsigned(&self) -> bool {
            false
        }
        pub fn is_signed(&self) -> bool {
            matches!(
                self.unquantized(),
                DatumType::I8 | DatumType::I16 | DatumType::I32 | DatumType::I64
            )
        }
        pub fn is_float(&self) -> bool {
            matches!(self, DatumType::F32)
        }
        pub fn is_copy(&self) -> bool {
            self.is_unsigned() || self.is_signed() || self.is_float()
        }
        pub fn unquantized(&self) -> DatumType {
            match self {
                _ => *self,
            }
        }
        #[inline]
        pub fn size_of(&self) -> usize {
            std::mem::size_of::<f32>()
        }
        #[inline]
        pub fn alignment(&self) -> usize {
            match self {
                DatumType::TDim => std::mem::size_of::<usize>(),
                DatumType::String => std::mem::size_of::<usize>(),
                _ => self.size_of(),
            }
        }
    }
    pub fn round_ties_to_even(x: f32) -> f32 {
        unimplemented!()
    }
    #[inline]
    pub fn scale_by<T: Datum + AsPrimitive<f32>>(b: T, a: f32) -> T
    where
        f32: AsPrimitive<T>,
    {
        unimplemented!()
    }
    pub trait ClampCast: PartialOrd + Copy + 'static {
        #[inline(always)]
        fn clamp_cast<O>(self) -> O
        where
            Self: AsPrimitive<O> + Datum,
            O: AsPrimitive<Self> + num_traits::Bounded + Datum,
        {
            unimplemented!()
        }
    }
    pub trait Datum:
        Clone + Send + Sync + fmt::Debug + fmt::Display + Default + 'static + PartialEq
    {
        fn name() -> &'static str;
        fn datum_type() -> DatumType;
    }
    macro_rules! datum {
        ($ t : ty , $ v : ident) => {
            impl From<$t> for Tensor {
                fn from(it: $t) -> Tensor {
                    tensor0(it)
                }
            }
            impl Datum for $t {
                fn name() -> &'static str {
                    stringify!($t)
                }
                fn datum_type() -> DatumType {
                    DatumType::$v
                }
            }
        };
    }
    datum!(f32, F32);
    datum!(TDim, TDim);
    datum!(String, String);
    datum!(Blob, Blob);
}
mod dim {
    use std::fmt;
    use std::ops;
    mod sym {
        use std::fmt;
        use std::sync::{Arc, Mutex, Weak};
        #[derive(Clone, Default)]
        pub struct SymbolTable;
        #[derive(Clone)]
        pub struct Symbol(Weak<Mutex<()>>, char);
        impl PartialEq for Symbol {
            fn eq(&self, other: &Self) -> bool {
                unimplemented!()
            }
        }
        impl Eq for Symbol {}
        impl PartialOrd for Symbol {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                unimplemented!()
            }
        }
        impl Ord for Symbol {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                unimplemented!()
            }
        }
        impl std::hash::Hash for Symbol {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                unimplemented!()
            }
        }
        impl fmt::Debug for Symbol {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                unimplemented!()
            }
        }
        #[derive(Clone, Debug, Default)]
        pub struct SymbolValues(Vec<Option<i64>>);
    }
    mod tree {
        use super::sym::*;
        use num_traits::{AsPrimitive, PrimInt, Zero};
        use std::{fmt, ops};
        #[derive(Debug)]
        pub struct UndeterminedSymbol(TDim);
        impl std::fmt::Display for UndeterminedSymbol {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                unimplemented!()
            }
        }
        #[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Debug)]
        pub enum TDim {
            Sym(Symbol),
            Val(i64),
            Add(Vec<TDim>),
            Mul(Vec<TDim>),
            MulInt(i64, Box<TDim>),
            Div(Box<TDim>, u64),
        }
        use TDim::*;
        impl fmt::Display for TDim {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                unimplemented!()
            }
        }
        impl TDim {
            pub fn to_i64(&self) -> anyhow::Result<i64> {
                if let Val(v) = self {
                    Ok(*v)
                } else {
                    unimplemented!()
                }
            }
        }
        impl Zero for TDim {
            fn zero() -> Self {
                unimplemented!()
            }
            fn is_zero(&self) -> bool {
                unimplemented!()
            }
        }
        impl Default for TDim {
            fn default() -> TDim {
                unimplemented!()
            }
        }
        impl ::std::iter::Sum for TDim {
            fn sum<I: Iterator<Item = TDim>>(iter: I) -> TDim {
                unimplemented!()
            }
        }
        impl std::iter::Product for TDim {
            fn product<I: Iterator<Item = TDim>>(iter: I) -> Self {
                unimplemented!()
            }
        }
        macro_rules! from_i {
            ($ i : ty) => {
                impl From<$i> for TDim {
                    fn from(v: $i) -> TDim {
                        TDim::Val(v as _)
                    }
                }
                impl<'a> From<&'a $i> for TDim {
                    fn from(v: &'a $i) -> TDim {
                        TDim::Val(*v as _)
                    }
                }
            };
        }
        from_i!(i32);
        from_i!(usize);
        impl<I> ops::Add<I> for TDim
        where
            I: Into<TDim>,
        {
            type Output = Self;
            fn add(mut self, rhs: I) -> Self {
                unimplemented!()
            }
        }
        impl<'a> ops::Add<&'a TDim> for TDim {
            type Output = Self;
            fn add(mut self, rhs: &'a TDim) -> Self {
                unimplemented!()
            }
        }
        impl<I> ops::Sub<I> for TDim
        where
            I: Into<TDim>,
        {
            type Output = Self;
            fn sub(mut self, rhs: I) -> Self {
                unimplemented!()
            }
        }
        impl<'a> ops::Sub<&'a TDim> for TDim {
            type Output = Self;
            fn sub(mut self, rhs: &'a TDim) -> Self {
                unimplemented!()
            }
        }
        impl<I: Into<TDim>> ops::Mul<I> for TDim {
            type Output = Self;
            fn mul(mut self, rhs: I) -> Self {
                unimplemented!()
            }
        }
        impl<'a> ops::Mul<&'a TDim> for TDim {
            type Output = Self;
            fn mul(mut self, rhs: &'a TDim) -> Self {
                unimplemented!()
            }
        }
        impl<I: AsPrimitive<u64> + PrimInt> ops::Div<I> for TDim {
            type Output = Self;
            fn div(mut self, rhs: I) -> Self {
                unimplemented!()
            }
        }
        impl<I: AsPrimitive<u64> + PrimInt> ops::Rem<I> for TDim {
            type Output = Self;
            fn rem(mut self, rhs: I) -> Self {
                unimplemented!()
            }
        }
    }
    pub use self::sym::{Symbol, SymbolTable, SymbolValues};
    pub use self::tree::{TDim, UndeterminedSymbol};
    use crate::{TractError, TractResult};
    pub trait DimLike:
        Clone
        + Default
        + PartialEq
        + From<usize>
        + for<'a> std::convert::TryFrom<&'a TDim, Error = TractError>
        + ::num_traits::Zero
        + fmt::Debug
        + fmt::Display
        + std::hash::Hash
        + ops::Add<Self, Output = Self>
        + ops::Add<usize, Output = Self>
        + for<'a> ops::Add<&'a Self, Output = Self>
        + ops::Sub<Self, Output = Self>
        + ops::Sub<usize, Output = Self>
        + for<'a> ops::Sub<&'a Self, Output = Self>
        + ops::Mul<Self, Output = Self>
        + ops::Mul<usize, Output = Self>
        + for<'a> ops::Mul<&'a Self, Output = Self>
        + ops::Div<usize, Output = Self>
        + ops::Rem<usize, Output = Self>
        + Send
        + Sync
        + 'static
        + std::iter::Sum
        + std::iter::Product
        + ToDim
    {
        fn maybe_div(&self, other: &Self) -> TractResult<(Self, u64)>;
        fn divceil(&self, other: usize) -> Self {
            unimplemented!()
        }
        fn to_i64(&self) -> TractResult<i64>;
        fn to_usize(&self) -> TractResult<usize> {
            self.to_i64().map(|d| d as usize)
        }
        fn to_isize(&self) -> TractResult<isize> {
            self.to_i64().map(|d| d as isize)
        }
        fn to_i32(&self) -> TractResult<i32> {
            unimplemented!()
        }
        fn one() -> Self;
        fn eval(&self, values: &SymbolValues) -> Self;
    }
    impl DimLike for TDim {
        fn maybe_div(&self, other: &Self) -> TractResult<(Self, u64)> {
            unimplemented!()
        }
        fn to_i64(&self) -> TractResult<i64> {
            TDim::to_i64(self)
        }
        fn one() -> Self {
            unimplemented!()
        }
        fn eval(&self, values: &SymbolValues) -> Self {
            unimplemented!()
        }
    }
    impl<'a> std::convert::TryFrom<&'a TDim> for TDim {
        type Error = anyhow::Error;
        fn try_from(d: &'a TDim) -> TractResult<TDim> {
            unimplemented!()
        }
    }
    pub trait ToDim {
        fn to_dim(&self) -> TDim;
    }
    impl<I: Into<TDim> + Clone> ToDim for I {
        fn to_dim(&self) -> TDim {
            self.clone().into()
        }
    }
}
mod tensor {
    use crate::datum::{round_ties_to_even, scale_by, Blob, ClampCast, Datum, DatumType, QParams};
    use crate::dim::TDim;
    use ndarray::prelude::*;
    use std::alloc;
    use std::fmt;
    use std::mem::{align_of, size_of};
    use std::sync::Arc;
    pub mod litteral {
        use super::Tensor;
        use crate::datum::Datum;
        use ndarray::*;
        pub fn tensor0<A: Datum>(x: A) -> Tensor {
            Tensor::from(arr0(x))
        }
    }
    #[derive(Eq)]
    pub struct Tensor {
        dt: DatumType,
        shape: Vec<usize>,
        strides: Vec<isize>,
        len: usize,
        layout: alloc::Layout,
        data: *mut u8,
    }
    unsafe impl Send for Tensor {}
    unsafe impl Sync for Tensor {}
    impl Tensor {
        pub unsafe fn uninitialized<T: Datum>(shape: &[usize]) -> anyhow::Result<Tensor> {
            Self::uninitialized_dt(T::datum_type(), shape)
        }
        pub unsafe fn uninitialized_dt(dt: DatumType, shape: &[usize]) -> anyhow::Result<Tensor> {
            Self::uninitialized_aligned_dt(dt, shape, dt.alignment())
        }
        pub unsafe fn uninitialized_aligned<T: Datum>(
            shape: &[usize],
            alignment: usize,
        ) -> anyhow::Result<Tensor> {
            Self::uninitialized_aligned_dt(T::datum_type(), shape, alignment)
        }
        pub unsafe fn uninitialized_aligned_dt(
            dt: DatumType,
            shape: &[usize],
            alignment: usize,
        ) -> anyhow::Result<Tensor> {
            if dt == String::datum_type() {
                unimplemented!()
            } else if dt == Blob::datum_type() {
                unimplemented!()
            } else if dt == TDim::datum_type() {
                unimplemented!()
            }
            assert!(dt.is_copy());
            let bytes = shape.iter().cloned().product::<usize>() * dt.size_of();
            let layout = alloc::Layout::from_size_align(bytes, alignment)?;
            let data = if bytes == 0 {
                unimplemented!()
            } else {
                let ptr = alloc::alloc(layout);
                assert!(!ptr.is_null());
                ptr
            } as *mut u8;
            let mut tensor = Tensor {
                strides: vec![],
                layout,
                dt,
                shape: shape.into(),
                data,
                len: 0,
            };
            tensor.update_strides_and_len();
            #[cfg(debug_assertions)]
            if !data.is_null() {
                if dt == DatumType::F32 {
                    tensor
                        .as_slice_mut_unchecked::<f32>()
                        .iter_mut()
                        .for_each(|f| *f = std::f32::NAN);
                } else {
                    unimplemented!()
                }
            }
            Ok(tensor)
        }
        pub fn clear<T: Datum + num_traits::Zero + Clone>(&mut self) -> anyhow::Result<()> {
            self.fill_t(T::zero())
        }
        pub fn zero<T: Datum + num_traits::Zero>(shape: &[usize]) -> anyhow::Result<Tensor> {
            unsafe {
                let mut t = Tensor::uninitialized::<T>(shape)?;
                t.clear::<T>()?;
                Ok(t)
            }
        }
        pub fn fill_t<T: Datum + Clone>(&mut self, value: T) -> anyhow::Result<()> {
            self.as_slice_mut::<T>()?
                .iter_mut()
                .for_each(|item| *item = value.clone());
            Ok(())
        }
        pub fn zero_aligned<T: Datum + num_traits::Zero>(
            shape: &[usize],
            alignment: usize,
        ) -> anyhow::Result<Tensor> {
            unsafe {
                let mut tensor = Self::uninitialized_aligned::<T>(shape, alignment)?;
                tensor.clear::<T>()?;
                Ok(tensor)
            }
        }
        #[inline]
        pub fn rank(&self) -> usize {
            self.shape.len()
        }
        #[inline]
        pub fn shape(&self) -> &[usize] {
            &self.shape
        }
        #[inline]
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> usize {
            self.len
        }
        fn update_strides_and_len(&mut self) {
            self.strides.clear();
            compute_natural_stride_to(&mut self.strides, &self.shape);
            self.len = if self.rank() == 0 {
                1
            } else {
                unsafe { *self.strides.get_unchecked(0) as usize * self.shape.get_unchecked(0) }
            }
        }
        #[inline]
        pub fn datum_type(&self) -> DatumType {
            self.dt
        }
        #[inline]
        pub unsafe fn set_datum_type(&mut self, dt: DatumType) {
            self.dt = dt
        }
        fn check_for_access<D: Datum>(&self) -> anyhow::Result<()> {
            if self.datum_type().unquantized() != D::datum_type().unquantized() {
                unimplemented!()
            }
            Ok(())
        }
        pub fn as_ptr<D: Datum>(&self) -> anyhow::Result<*const D> {
            self.check_for_access::<D>()?;
            Ok(self.data as *const D)
        }
        pub fn as_ptr_mut<D: Datum>(&mut self) -> anyhow::Result<*mut D> {
            self.as_ptr::<D>().map(|p| p as *mut D)
        }
        pub fn as_slice_mut<D: Datum>(&mut self) -> anyhow::Result<&mut [D]> {
            let ptr: *mut D = self.as_ptr_mut()?;
            if ptr.is_null() {
                unimplemented!()
            } else {
                unsafe { Ok(std::slice::from_raw_parts_mut::<D>(ptr, self.len())) }
            }
        }
        pub unsafe fn as_slice_unchecked<D: Datum>(&self) -> &[D] {
            if self.data.is_null() {
                unimplemented!()
            } else {
                std::slice::from_raw_parts::<D>(self.data as *const D, self.len())
            }
        }
        pub unsafe fn as_slice_mut_unchecked<D: Datum>(&mut self) -> &mut [D] {
            if self.data.is_null() {
                unimplemented!()
            } else {
                std::slice::from_raw_parts_mut::<D>(self.data as *mut D, self.len())
            }
        }
        unsafe fn is_uniform_t<T: Datum>(&self) -> bool {
            let slice = self.as_slice_unchecked::<T>();
            slice[1..].iter().all(|x| x == &slice[0])
        }
        pub fn is_uniform(&self) -> bool {
            if self.len() <= 1 {
                unimplemented!()
            }
            unsafe { Tensor::is_uniform_t::<f32>(self) }
        }
        unsafe fn as_uniform_t<T: Datum>(&self) -> Tensor {
            let v: T = self.as_slice_unchecked::<T>()[0].clone();
            litteral::tensor0(v)
        }
        pub fn as_uniform(&self) -> Option<Tensor> {
            if self.len() >= 1 && self.is_uniform() {
                unsafe {
                    let mut t = Tensor::as_uniform_t::<f32>(self);
                    t.set_datum_type(self.datum_type());
                    Some(t)
                }
            } else {
                unimplemented!()
            }
        }
        fn from_datum<T: Datum>(it: ArrayD<T>) -> Tensor {
            if it.as_slice().is_some() {
                let layout =
                    alloc::Layout::from_size_align(it.len() * size_of::<T>(), align_of::<T>())
                        .unwrap();
                let shape = it.shape().into();
                let vec = it.into_raw_vec().into_boxed_slice();
                let data = Box::into_raw(vec) as *mut u8;
                let mut t = Tensor {
                    dt: T::datum_type(),
                    shape,
                    layout,
                    data,
                    strides: vec![],
                    len: 0,
                };
                t.update_strides_and_len();
                return t;
            }
            unsafe { unimplemented!() }
        }
    }
    impl PartialEq for Tensor {
        fn eq(&self, other: &Tensor) -> bool {
            unimplemented!()
        }
    }
    fn compute_natural_stride_to(strides: &mut Vec<isize>, shape: &[usize]) {
        match shape.len() {
            0 => (),
            1 => strides.push(1),
            2 => strides.extend_from_slice(&[shape[1] as isize, 1]),
            3 => strides.extend_from_slice(&[(shape[1] * shape[2]) as isize, shape[2] as _, 1]),
            4 => strides.extend_from_slice(&[
                (shape[1] * shape[2] * shape[3]) as isize,
                (shape[2] * shape[3]) as _,
                shape[3] as _,
                1,
            ]),
            _ => {
                unimplemented!()
            }
        }
    }
    impl<D: ::ndarray::Dimension, T: Datum> From<Array<T, D>> for Tensor {
        fn from(it: Array<T, D>) -> Tensor {
            Tensor::from_datum(it.into_dyn())
        }
    }
    pub trait IntoTensor: Sized {
        fn into_tensor(self) -> Tensor;
    }
    pub trait IntoArcTensor: Sized {
        fn into_arc_tensor(self) -> Arc<Tensor>;
    }
    impl IntoArcTensor for Tensor {
        fn into_arc_tensor(self) -> Arc<Tensor> {
            Arc::new(self)
        }
    }
    impl IntoArcTensor for Arc<Tensor> {
        fn into_arc_tensor(self) -> Arc<Tensor> {
            self
        }
    }
}

#![allow(clippy::len_zero)]
#![allow(clippy::missing_safety_doc)]
pub type TractError = anyhow::Error;
pub type TractResult<T> = anyhow::Result<T>;
pub mod prelude {
    pub use crate::datum::{Datum, DatumType};
    pub use crate::tensor::{IntoArcTensor, IntoTensor, Tensor};
    pub use crate::{TractError, TractResult};
}
pub mod internal {
    pub use crate::prelude::*;
}
mod datum {
    use crate::tensor::litteral::*;
    use crate::tensor::Tensor;
    use num_traits::AsPrimitive;
    use std::hash::Hash;
    use std::{fmt, ops};
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
    pub enum DatumType {
        F32,
    }
    impl DatumType {
        pub fn is_unsigned(&self) -> bool {
            false
        }
        pub fn is_signed(&self) -> bool {
		false
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
                _ => self.size_of(),
            }
        }
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
}
mod tensor {
    use crate::datum::{ClampCast, Datum, DatumType};
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

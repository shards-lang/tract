#![allow(clippy::len_zero)]
#![allow(clippy::missing_safety_doc)]
#[macro_use]
mod macros {
    #[macro_export]
    macro_rules ! tvec { (@ one $ x : expr) => (1usize) ; ($ elem : expr ; $ n : expr) => ({ $ crate :: TVec :: from_elem ($ elem , $ n) }) ; ($ ($ x : expr) ,*$ (,) *) => ({ let count = 0usize $ (+ tvec ! (@ one $ x)) *; # [allow (unused_mut)] let mut vec = $ crate :: TVec :: new () ; if count <= vec . inline_size () { $ (vec . push ($ x) ;) * vec } else { $ crate :: TVec :: from_vec (vec ! [$ ($ x ,) *]) } }) ; }
    #[macro_export]
    macro_rules ! dispatch_datum { ($ ($ path : ident) ::* ($ dt : expr) ($ ($ args : expr) ,*)) => { { use $ crate :: prelude :: DatumType ; match $ dt { DatumType :: Bool => $ ($ path) ::*::< bool > ($ ($ args) ,*) , DatumType :: U8 => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: U16 => $ ($ path) ::*::< u16 > ($ ($ args) ,*) , DatumType :: U32 => $ ($ path) ::*::< u32 > ($ ($ args) ,*) , DatumType :: U64 => $ ($ path) ::*::< u64 > ($ ($ args) ,*) , DatumType :: I8 => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: I16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: I32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: I64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: F16 => $ ($ path) ::*::< f16 > ($ ($ args) ,*) , DatumType :: F32 => $ ($ path) ::*::< f32 > ($ ($ args) ,*) , DatumType :: F64 => $ ($ path) ::*::< f64 > ($ ($ args) ,*) , DatumType :: Blob => $ ($ path) ::*::< Blob > ($ ($ args) ,*) , DatumType :: TDim => $ ($ path) ::*::< TDim > ($ ($ args) ,*) , DatumType :: String => $ ($ path) ::*::< String > ($ ($ args) ,*) , DatumType :: QI8 (_) => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: QU8 (_) => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: QI32 (_) => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , } } } }
    #[macro_export]
    macro_rules ! dispatch_datum_by_size { ($ ($ path : ident) ::* ($ dt : expr) ($ ($ args : expr) ,*)) => { { use $ crate :: prelude :: DatumType ; match $ dt { DatumType :: Bool => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: U8 => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: U16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: U32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: U64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: I8 => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: I16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: I32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: I64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: F16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: F32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: F64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: Blob => $ ($ path) ::*::< Blob > ($ ($ args) ,*) , DatumType :: TDim => $ ($ path) ::*::< TDim > ($ ($ args) ,*) , DatumType :: String => $ ($ path) ::*::< String > ($ ($ args) ,*) , DatumType :: QI8 (_) => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: QU8 (_) => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: QI32 (_) => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , } } } }
    #[macro_export]
    macro_rules ! dispatch_copy { ($ ($ path : ident) ::* ($ dt : expr) ($ ($ args : expr) ,*)) => { { use $ crate :: prelude :: DatumType ; match $ dt { DatumType :: Bool => $ ($ path) ::*::< bool > ($ ($ args) ,*) , DatumType :: U8 => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: U16 => $ ($ path) ::*::< u16 > ($ ($ args) ,*) , DatumType :: U32 => $ ($ path) ::*::< u32 > ($ ($ args) ,*) , DatumType :: U64 => $ ($ path) ::*::< u64 > ($ ($ args) ,*) , DatumType :: I8 => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: I16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: I32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: I64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: F16 => $ ($ path) ::*::< f16 > ($ ($ args) ,*) , DatumType :: F32 => $ ($ path) ::*::< f32 > ($ ($ args) ,*) , DatumType :: F64 => $ ($ path) ::*::< f64 > ($ ($ args) ,*) , DatumType :: QI8 (_) => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: QU8 (_) => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: QI32 (_) => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , _ => panic ! ("{:?} is not Copy" , $ dt) } } } }
    #[macro_export]
    macro_rules ! dispatch_copy_by_size { ($ ($ path : ident) ::* ($ dt : expr) ($ ($ args : expr) ,*)) => { { use $ crate :: prelude :: DatumType ; match $ dt { DatumType :: Bool => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: U8 => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: U16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: U32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: U64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: I8 => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: I16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: I32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: I64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: F16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: F32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: F64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: QI8 (_) => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: QU8 (_) => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: QI32 (_) => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , _ => panic ! ("{:?} is not Copy" , $ dt) } } } }
    #[macro_export]
    macro_rules ! dispatch_numbers { ($ ($ path : ident) ::* ($ dt : expr) ($ ($ args : expr) ,*)) => { { use $ crate :: prelude :: DatumType ; match $ dt { DatumType :: U8 => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: U16 => $ ($ path) ::*::< u16 > ($ ($ args) ,*) , DatumType :: U32 => $ ($ path) ::*::< u32 > ($ ($ args) ,*) , DatumType :: U64 => $ ($ path) ::*::< u64 > ($ ($ args) ,*) , DatumType :: I8 => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: I16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: I32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: I64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: F16 => $ ($ path) ::*::< f16 > ($ ($ args) ,*) , DatumType :: F32 => $ ($ path) ::*::< f32 > ($ ($ args) ,*) , DatumType :: F64 => $ ($ path) ::*::< f64 > ($ ($ args) ,*) , DatumType :: QI8 (_) => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: QU8 (_) => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: QI32 (_) => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , _ => $ crate :: anyhow :: bail ! ("{:?} is not a number" , $ dt) } } } }
    #[macro_export]
    macro_rules ! dispatch_zerolike { ($ ($ path : ident) ::* ($ dt : expr) ($ ($ args : expr) ,*)) => { { use $ crate :: prelude :: DatumType ; match $ dt { DatumType :: TDim => $ ($ path) ::*::< TDim > ($ ($ args) ,*) , DatumType :: U8 => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: U16 => $ ($ path) ::*::< u16 > ($ ($ args) ,*) , DatumType :: U32 => $ ($ path) ::*::< u32 > ($ ($ args) ,*) , DatumType :: U64 => $ ($ path) ::*::< u64 > ($ ($ args) ,*) , DatumType :: I8 => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: I16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: I32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: I64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: F16 => $ ($ path) ::*::< f16 > ($ ($ args) ,*) , DatumType :: F32 => $ ($ path) ::*::< f32 > ($ ($ args) ,*) , DatumType :: F64 => $ ($ path) ::*::< f64 > ($ ($ args) ,*) , DatumType :: QI8 (_) => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: QU8 (_) => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: QI32 (_) => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , _ => $ crate :: anyhow :: bail ! ("{:?} is doesn't implement num_traits::Zero" , $ dt) } } } }
    #[macro_export]
    macro_rules ! dispatch_floatlike { ($ ($ path : ident) ::* ($ dt : expr) ($ ($ args : expr) ,*)) => { { use $ crate :: prelude :: DatumType ; match $ dt { DatumType :: F16 => $ ($ path) ::*::< f16 > ($ ($ args) ,*) , DatumType :: F32 => $ ($ path) ::*::< f32 > ($ ($ args) ,*) , DatumType :: F64 => $ ($ path) ::*::< f64 > ($ ($ args) ,*) , _ => $ crate :: anyhow :: bail ! ("{:?} is not float-like" , $ dt) } } } }
    #[macro_export]
    macro_rules ! dispatch_signed { ($ ($ path : ident) ::* ($ dt : expr) ($ ($ args : expr) ,*)) => { { use $ crate :: prelude :: DatumType ; match $ dt { DatumType :: F16 => $ ($ path) ::*::< f16 > ($ ($ args) ,*) , DatumType :: F32 => $ ($ path) ::*::< f32 > ($ ($ args) ,*) , DatumType :: F64 => $ ($ path) ::*::< f64 > ($ ($ args) ,*) , DatumType :: I8 => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: I16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: I32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: I64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: TDim => $ ($ path) ::*::< TDim > ($ ($ args) ,*) , _ => $ crate :: anyhow :: bail ! ("{:?} is not signed" , $ dt) } } } }
    #[macro_export]
    macro_rules ! dispatch_hash { ($ ($ path : ident) ::* ($ dt : expr) ($ ($ args : expr) ,*)) => { { use $ crate :: prelude :: DatumType ; match $ dt { DatumType :: Bool => $ ($ path) ::*::< bool > ($ ($ args) ,*) , DatumType :: U8 => $ ($ path) ::*::< u8 > ($ ($ args) ,*) , DatumType :: U16 => $ ($ path) ::*::< u16 > ($ ($ args) ,*) , DatumType :: U32 => $ ($ path) ::*::< u32 > ($ ($ args) ,*) , DatumType :: U64 => $ ($ path) ::*::< u64 > ($ ($ args) ,*) , DatumType :: I8 => $ ($ path) ::*::< i8 > ($ ($ args) ,*) , DatumType :: I16 => $ ($ path) ::*::< i16 > ($ ($ args) ,*) , DatumType :: I32 => $ ($ path) ::*::< i32 > ($ ($ args) ,*) , DatumType :: I64 => $ ($ path) ::*::< i64 > ($ ($ args) ,*) , DatumType :: Blob => $ ($ path) ::*::< Blob > ($ ($ args) ,*) , DatumType :: TDim => $ ($ path) ::*::< TDim > ($ ($ args) ,*) , DatumType :: String => $ ($ path) ::*::< String > ($ ($ args) ,*) , DatumType :: ComplexI16 => $ ($ path) ::*::< Complex < i16 >> ($ ($ args) ,*) , DatumType :: ComplexI32 => $ ($ path) ::*::< Complex < i32 >> ($ ($ args) ,*) , DatumType :: ComplexI64 => $ ($ path) ::*::< Complex < i64 >> ($ ($ args) ,*) , _ => $ crate :: anyhow :: bail ! ("{:?} is not Hash" , $ dt) } } } }
}
pub type TVec<T> = smallvec::SmallVec<[T; 4]>;
pub type TractError = anyhow::Error;
pub type TractResult<T> = anyhow::Result<T>;
pub mod prelude {
    pub use crate::datum::{round_ties_to_even, Blob, Datum, DatumType, QParams};
    pub use crate::dim::{Symbol, SymbolTable, SymbolValues, TDim, ToDim};
    pub use crate::tensor::{natural_strides, IntoArcTensor, IntoTensor, Tensor};
    pub use crate::tvec;
    pub use crate::TVec;
    pub use crate::{TractError, TractResult};
}
pub mod internal {
    pub use crate::dim::{parse_tdim, DimLike};
    pub use crate::prelude::*;
}
pub use anyhow;
mod datum {
    use crate::dim::TDim;
    use crate::tensor::litteral::*;
    use crate::tensor::Tensor;
    use crate::TVec;
    use half::f16;
    use num_traits::AsPrimitive;
    use scan_fmt::scan_fmt;
    use std::hash::Hash;
    use std::{fmt, ops};
    #[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
    pub struct Blob(pub Vec<u8>);
    impl ops::Deref for Blob {
        type Target = [u8];
        fn deref(&self) -> &[u8] {
            unimplemented!()
        }
    }
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
    impl Ord for QParams {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            unimplemented!()
        }
    }
    impl PartialOrd for QParams {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            unimplemented!()
        }
    }
    #[allow(clippy::derive_hash_xor_eq)]
    impl Hash for QParams {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            unimplemented!()
        }
    }
    impl QParams {
        pub fn zp_scale(&self) -> (i32, f32) {
            unimplemented!()
        }
        pub fn dq(&self, i: i32) -> f32 {
            unimplemented!()
        }
    }
    impl std::fmt::Debug for QParams {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            unimplemented!()
        }
    }
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
    pub enum DatumType {
        Bool,
        U8,
        U16,
        U32,
        U64,
        I8,
        I16,
        I32,
        I64,
        F16,
        F32,
        F64,
        TDim,
        Blob,
        String,
        QI8(QParams),
        QU8(QParams),
        QI32(QParams),
    }
    impl DatumType {
        pub fn super_types(&self) -> TVec<DatumType> {
            unimplemented!()
        }
        pub fn is_unsigned(&self) -> bool {
            matches!(
                self.unquantized(),
                DatumType::U8 | DatumType::U16 | DatumType::U32 | DatumType::U64
            )
        }
        pub fn is_signed(&self) -> bool {
            matches!(
                self.unquantized(),
                DatumType::I8 | DatumType::I16 | DatumType::I32 | DatumType::I64
            )
        }
        pub fn is_float(&self) -> bool {
            matches!(self, DatumType::F16 | DatumType::F32 | DatumType::F64)
        }
        pub fn is_copy(&self) -> bool {
            *self == DatumType::Bool || self.is_unsigned() || self.is_signed() || self.is_float()
        }
        pub fn is_quantized(&self) -> bool {
            unimplemented!()
        }
        pub fn qparams(&self) -> Option<QParams> {
            unimplemented!()
        }
        #[inline(always)]
        pub fn zp_scale(&self) -> (i32, f32) {
            unimplemented!()
        }
        pub fn unquantized(&self) -> DatumType {
            match self {
                DatumType::QI8(_) => DatumType::I8,
                DatumType::QU8(_) => DatumType::U8,
                DatumType::QI32(_) => DatumType::I32,
                _ => *self,
            }
        }
        pub fn is_integer(&self) -> bool {
            unimplemented!()
        }
        #[inline]
        pub fn size_of(&self) -> usize {
            dispatch_datum!(std::mem::size_of(self)())
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
    const TOINT: f32 = 1.0f32 / std::f32::EPSILON;
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
    impl<T: PartialOrd + Copy + 'static> ClampCast for T {}
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
    datum!(bool, Bool);
    datum!(f16, F16);
    datum!(f32, F32);
    datum!(f64, F64);
    datum!(i8, I8);
    datum!(i16, I16);
    datum!(i32, I32);
    datum!(i64, I64);
    datum!(u8, U8);
    datum!(u16, U16);
    datum!(u32, U32);
    datum!(u64, U64);
    datum!(TDim, TDim);
    datum!(String, String);
    datum!(Blob, Blob);
}
mod dim {
    use num_traits::Zero;
    use std::fmt;
    use std::ops;
    mod parse {
        use super::*;
        use nom::branch::alt;
        use nom::bytes::complete::tag;
        use nom::character::complete::{alpha1, alphanumeric1, digit1};
        use nom::combinator::{all_consuming, map, map_res, recognize};
        use nom::multi::many0;
        use nom::sequence::{delimited, pair, separated_pair};
        use nom::IResult;
        pub fn parse_tdim(symbol_table: &SymbolTable, input: &str) -> TractResult<TDim> {
            unimplemented!()
        }
        fn expr<'i>(symbol_table: &SymbolTable, i: &'i str) -> IResult<&'i str, TDim> {
            unimplemented!()
        }
        macro_rules! bin {
            ($ name : ident , $ next : ident , $ op : expr , $ builder : expr) => {
                fn $name<'i>(symbol_table: &SymbolTable, input: &'i str) -> IResult<&'i str, TDim> {
                    let s = symbol_table;
                    alt((
                        map(
                            separated_pair(|i| $next(s, i), tag($op), |i| $next(s, i)),
                            $builder,
                        ),
                        |i| $next(s, i),
                    ))(input)
                }
            };
        }
        bin!(add, sub, "+", |(a, b)| a + b);
        bin!(sub, mul, "-", |(a, b)| a - b);
        bin!(mul, div, "*", |(a, b)| a * b);
        fn div<'i>(symbol_table: &SymbolTable, input: &'i str) -> IResult<&'i str, TDim> {
            unimplemented!()
        }
        fn atom<'i>(symbol_table: &SymbolTable, i: &'i str) -> IResult<&'i str, TDim> {
            unimplemented!()
        }
        fn identifier<'i>(symbol_table: &SymbolTable, i: &'i str) -> IResult<&'i str, Symbol> {
            unimplemented!()
        }
        fn numeric(i: &str) -> IResult<&str, i64> {
            unimplemented!()
        }
    }
    mod sym {
        use itertools::Itertools;
        use std::fmt;
        use std::sync::{Arc, Mutex, Weak};
        use string_interner::StringInterner;
        use string_interner::Symbol as _;
        #[derive(Clone, Default)]
        pub struct SymbolTable(Arc<Mutex<StringInterner>>);
        impl SymbolTable {
            pub fn sym(&self, name: &str) -> Symbol {
                unimplemented!()
            }
        }
        #[derive(Clone)]
        pub struct Symbol(Weak<Mutex<StringInterner>>, string_interner::DefaultSymbol);
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
        impl std::fmt::Display for Symbol {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        impl std::ops::Index<&Symbol> for SymbolValues {
            type Output = Option<i64>;
            fn index(&self, index: &Symbol) -> &Self::Output {
                unimplemented!()
            }
        }
    }
    mod tree {
        use super::sym::*;
        use itertools::Itertools;
        use num_traits::{AsPrimitive, PrimInt, Zero};
        use std::collections::HashMap;
        use std::{fmt, ops};
        #[derive(Debug)]
        pub struct UndeterminedSymbol(TDim);
        impl std::fmt::Display for UndeterminedSymbol {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                unimplemented!()
            }
        }
        impl std::error::Error for UndeterminedSymbol {}
        macro_rules ! b (($ e : expr) => { Box :: new ($ e) }) ;
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
            pub fn reduce(self) -> TDim {
                unimplemented!()
            }
            fn cost(&self) -> usize {
                unimplemented!()
            }
            fn wiggle(&self) -> Vec<TDim> {
                unimplemented!()
            }
            pub fn simplify(self) -> TDim {
                unimplemented!()
            }
            fn gcd(&self) -> u64 {
                unimplemented!()
            }
            fn div(&self, d: u64) -> TDim {
                unimplemented!()
            }
        }
        pub(super) fn reduce_ratio(mut p: i64, mut q: i64) -> (i64, u64) {
            unimplemented!()
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
        from_i!(i64);
        from_i!(u64);
        from_i!(usize);
        impl ops::Neg for TDim {
            type Output = Self;
            fn neg(self) -> Self {
                unimplemented!()
            }
        }
        impl<'a> ops::AddAssign<&'a TDim> for TDim {
            fn add_assign(&mut self, rhs: &'a TDim) {
                unimplemented!()
            }
        }
        impl<I> ops::AddAssign<I> for TDim
        where
            I: Into<TDim>,
        {
            fn add_assign(&mut self, rhs: I) {
                unimplemented!()
            }
        }
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
        #[allow(clippy::suspicious_op_assign_impl)]
        impl<'a> ops::SubAssign<&'a TDim> for TDim {
            fn sub_assign(&mut self, rhs: &'a TDim) {
                unimplemented!()
            }
        }
        impl<I> ops::SubAssign<I> for TDim
        where
            I: Into<TDim>,
        {
            fn sub_assign(&mut self, rhs: I) {
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
        impl<I: Into<TDim>> ops::MulAssign<I> for TDim {
            fn mul_assign(&mut self, rhs: I) {
                unimplemented!()
            }
        }
        impl<'a> ops::MulAssign<&'a TDim> for TDim {
            fn mul_assign(&mut self, rhs: &'a TDim) {
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
        impl<I: AsPrimitive<u64> + PrimInt> ops::DivAssign<I> for TDim {
            fn div_assign(&mut self, rhs: I) {
                unimplemented!()
            }
        }
        impl<I: AsPrimitive<u64> + PrimInt> ops::Div<I> for TDim {
            type Output = Self;
            fn div(mut self, rhs: I) -> Self {
                unimplemented!()
            }
        }
        impl<I: AsPrimitive<u64> + PrimInt> ops::RemAssign<I> for TDim {
            fn rem_assign(&mut self, rhs: I) {
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
    pub use self::parse::parse_tdim;
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
pub mod hash {
    use std::hash::{Hash, Hasher};
    struct WrappedHasher<'a>(&'a mut dyn Hasher);
    impl<'a> Hasher for WrappedHasher<'a> {
        fn finish(&self) -> u64 {
            unimplemented!()
        }
        fn write(&mut self, bytes: &[u8]) {
            unimplemented!()
        }
    }
}
mod scatter {
    use crate::prelude::*;
    use ndarray::Dimension;
    pub(crate) unsafe fn scatter_contig_data<T: Datum>(
        mut src: *const T,
        dst: *mut T,
        dst_len_and_strides: &[(usize, usize)],
    ) {
        unimplemented!()
    }
}
mod tensor {
    use crate::datum::{round_ties_to_even, scale_by, Blob, ClampCast, Datum, DatumType, QParams};
    use crate::dim::TDim;
    use crate::TVec;
    use half::f16;
    use itertools::Itertools;
    use ndarray::prelude::*;
    use std::alloc;
    use std::borrow::Cow;
    use std::fmt;
    use std::hash::Hash;
    use std::mem::{align_of, size_of};
    use std::ops::Range;
    use std::sync::Arc;
    pub mod litteral {
        use super::Tensor;
        use crate::datum::Datum;
        use ndarray::*;
        use std::sync::Arc;
        pub fn arr4<A, V, U, T>(xs: &[V]) -> Array4<A>
        where
            V: FixedInitializer<Elem = U> + Clone,
            U: FixedInitializer<Elem = T> + Clone,
            T: FixedInitializer<Elem = A> + Clone,
            A: Clone,
        {
            unimplemented!()
        }
        pub fn tensor0<A: Datum>(x: A) -> Tensor {
            Tensor::from(arr0(x))
        }
    }
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum Approximation {
        Exact,
        Close,
        Approximate,
    }
    impl Approximation {
        fn atol_and_rtol(&self, dt: &DatumType) -> (f64, f64) {
            unimplemented!()
        }
    }
    #[derive(Eq)]
    pub struct Tensor {
        dt: DatumType,
        shape: TVec<usize>,
        strides: TVec<isize>,
        len: usize,
        layout: alloc::Layout,
        data: *mut u8,
    }
    unsafe impl Send for Tensor {}
    unsafe impl Sync for Tensor {}
    impl Clone for Tensor {
        fn clone(&self) -> Tensor {
            unimplemented!()
        }
    }
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
                strides: tvec!(),
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
        pub unsafe fn from_raw_dt_align(
            dt: DatumType,
            shape: &[usize],
            content: &[u8],
            align: usize,
        ) -> anyhow::Result<Tensor> {
            unimplemented!()
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
        #[inline]
        pub fn strides(&self) -> &[isize] {
            unimplemented!()
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
        pub unsafe fn set_shape_unchecked(&mut self, shape: &[usize]) {
            unimplemented!()
        }
        pub fn set_shape(&mut self, shape: &[usize]) -> anyhow::Result<()> {
            unimplemented!()
        }
        pub fn permute_axes(self, axes: &[usize]) -> anyhow::Result<Tensor> {
            unimplemented!()
        }
        fn clip_range_bounds(
            &self,
            axis: usize,
            range: impl std::ops::RangeBounds<usize>,
        ) -> Range<usize> {
            unimplemented!()
        }
        #[inline]
        pub fn datum_type(&self) -> DatumType {
            self.dt
        }
        #[inline]
        pub unsafe fn set_datum_type(&mut self, dt: DatumType) {
            self.dt = dt
        }
        pub fn dump(&self, force_full: bool) -> anyhow::Result<String> {
            unimplemented!()
        }
        pub unsafe fn into_array_unchecked<D: Datum>(self) -> ArrayD<D> {
            unimplemented!()
        }
        fn check_for_access<D: Datum>(&self) -> anyhow::Result<()> {
            if self.datum_type().unquantized() != D::datum_type().unquantized() {
                unimplemented!()
            }
            Ok(())
        }
        pub fn to_array_view<D: Datum>(&self) -> anyhow::Result<ArrayViewD<D>> {
            unimplemented!()
        }
        pub unsafe fn to_array_view_unchecked<D: Datum>(&self) -> ArrayViewD<D> {
            unimplemented!()
        }
        pub fn as_ptr<D: Datum>(&self) -> anyhow::Result<*const D> {
            self.check_for_access::<D>()?;
            Ok(self.data as *const D)
        }
        pub unsafe fn as_ptr_mut_unchecked<D: Datum>(&mut self) -> *mut D {
            unimplemented!()
        }
        pub fn as_ptr_mut<D: Datum>(&mut self) -> anyhow::Result<*mut D> {
            self.as_ptr::<D>().map(|p| p as *mut D)
        }
        pub fn as_slice<D: Datum>(&self) -> anyhow::Result<&[D]> {
            unimplemented!()
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
        pub fn to_scalar<D: Datum>(&self) -> anyhow::Result<&D> {
            unimplemented!()
        }
        pub unsafe fn to_scalar_unchecked<D: Datum>(&self) -> &D {
            unimplemented!()
        }
        pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
            unimplemented!()
        }
        unsafe fn is_uniform_t<T: Datum>(&self) -> bool {
            let slice = self.as_slice_unchecked::<T>();
            slice[1..].iter().all(|x| x == &slice[0])
        }
        pub fn is_uniform(&self) -> bool {
            if self.len() <= 1 {
                unimplemented!()
            }
            unsafe { dispatch_datum!(Tensor::is_uniform_t(self.datum_type())(self)) }
        }
        unsafe fn as_uniform_t<T: Datum>(&self) -> Tensor {
            let v: T = self.as_slice_unchecked::<T>()[0].clone();
            litteral::tensor0(v)
        }
        pub fn as_uniform(&self) -> Option<Tensor> {
            if self.len() >= 1 && self.is_uniform() {
                unsafe {
                    let mut t = dispatch_datum!(Tensor::as_uniform_t(self.datum_type())(self));
                    t.set_datum_type(self.datum_type());
                    Some(t)
                }
            } else {
                unimplemented!()
            }
        }
        unsafe fn natural_cast<
            Source: Datum + num_traits::AsPrimitive<Target>,
            Target: Datum + Copy,
        >(
            &self,
            other: &mut Tensor,
        ) {
            unimplemented!()
        }
        unsafe fn cast_number_to_bool<Source: Datum + num_traits::Zero>(&self, other: &mut Tensor) {
            unimplemented!()
        }
        unsafe fn cast_from_string<Target: Datum + core::str::FromStr>(
            &self,
            other: &mut Tensor,
        ) -> anyhow::Result<()> {
            unimplemented!()
        }
        unsafe fn cast_to_string<Source: Datum>(&self, other: &mut Tensor) {
            unimplemented!()
        }
        pub fn cast_to<D: Datum>(&self) -> anyhow::Result<Cow<Tensor>> {
            unimplemented!()
        }
        pub fn cast_to_dt(&self, dst_dt: DatumType) -> anyhow::Result<Cow<Tensor>> {
            unimplemented!()
        }
        fn eq_dt(&self, other: &Tensor) -> anyhow::Result<bool> {
            unimplemented!()
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
                    strides: tvec!(),
                    len: 0,
                };
                t.update_strides_and_len();
                return t;
            }
            unsafe { unimplemented!() }
        }
        pub fn deep_clone(&self) -> Tensor {
            unimplemented!()
        }
    }
    impl PartialEq for Tensor {
        fn eq(&self, other: &Tensor) -> bool {
            unimplemented!()
        }
    }
    impl fmt::Debug for Tensor {
        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            unimplemented!()
        }
    }
    pub fn natural_strides(shape: &[usize]) -> TVec<isize> {
        unimplemented!()
    }
    fn compute_natural_stride_to(strides: &mut TVec<isize>, shape: &[usize]) {
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
    impl<D: ::ndarray::Dimension, T: Datum> IntoTensor for Array<T, D> {
        fn into_tensor(self) -> Tensor {
            unimplemented!()
        }
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

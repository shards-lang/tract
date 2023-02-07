pub mod lir_unary;
pub mod mir;
pub mod mir_unary;
pub mod pack;

use crate::internal::*;
use tract_itertools::Itertools;
use tract_ndarray::prelude::*;

pub use self::mir::MatMul;
pub use self::mir_unary::MatMulUnary;
use self::pack::MatMatMulPack;

#[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
pub struct MatMulAxes {
    pub a_m: usize,
    pub a_k: usize,
    pub b_k: usize,
    pub b_n: usize,
    pub c_m: usize,
    pub c_n: usize,
}

impl Default for MatMulAxes {
    fn default() -> Self {
        Self::default_for_rank(2)
    }
}

impl MatMulAxes {
    pub fn default_for_rank(rank: usize) -> Self {
        Self::default_for_ranks(rank, rank, rank)
    }

    pub fn default_for_ranks(a: usize, b: usize, c: usize) -> Self {
        MatMulAxes { a_m: a - 2, a_k: a - 1, b_k: b - 2, b_n: b - 1, c_m: c - 2, c_n: c - 1 }
    }

    pub fn transposing_a(self) -> Self {
        MatMulAxes { a_m: self.a_k, a_k: self.a_m, ..self }
    }

    pub fn transposing_b(self) -> Self {
        MatMulAxes { b_n: self.b_k, b_k: self.b_n, ..self }
    }

    pub fn transposing_c(self) -> Self {
        MatMulAxes { c_n: self.c_m, c_m: self.c_n, ..self }
    }

    pub fn transposing(self, a: bool, b: bool, c: bool) -> Self {
        let mut it = self;
        if a {
            it = it.transposing_a();
        }
        if b {
            it = it.transposing_b();
        }
        if c {
            it = it.transposing_c();
        }
        it
    }

    pub fn to_array(&self) -> [usize; 6] {
        [self.a_m, self.a_k, self.b_k, self.b_n, self.c_m, self.c_n]
    }

    pub fn from_array(array: &[usize]) -> TractResult<Self> {
        anyhow::ensure!(
            array.len() == 6,
            "MatMulAxes requires exactly six axis numbers, got {:?}",
            array
        );
        Ok(MatMulAxes {
            a_m: array[0],
            a_k: array[1],
            b_k: array[2],
            b_n: array[3],
            c_m: array[4],
            c_n: array[5],
        })
    }
}

pub fn compute_shape<D: DimLike>(
    ashape: &[D],
    bshape: &[D],
    axes: MatMulAxes,
) -> TractResult<(D, D, D, TVec<D>)> {
    let a_shape_bc: TVec<D> = ashape
        .iter()
        .enumerate()
        .filter_map(
            |(ix, dim)| if ix != axes.a_m && ix != axes.a_k { Some(dim.clone()) } else { None },
        )
        .collect();
    let b_shape_bc = bshape
        .iter()
        .enumerate()
        .filter_map(
            |(ix, dim)| if ix != axes.b_k && ix != axes.b_n { Some(dim.clone()) } else { None },
        )
        .collect();
    let mut c_shape = crate::broadcast::multi_broadcast(&[a_shape_bc, b_shape_bc])
        .ok_or_else(|| format_err!("Could not broadcast"))?;
    // dbg!(&c_shape);
    let (m, ka) = (ashape[axes.a_m].clone(), ashape[axes.a_k].clone());
    let (kb, n) = (bshape[axes.b_k].clone(), bshape[axes.b_n].clone());
    if ka != kb {
        bail!(
            "Inconsistent matmul: a: {} b: {}, axes: am:{} ak:{} bk:{} bn:{} cm:{} cn:{}",
            ashape.iter().join(","),
            bshape.iter().join(","),
            axes.a_m,
            axes.a_k,
            axes.b_k,
            axes.b_n,
            axes.c_m,
            axes.c_n
        );
    }
    if axes.c_m < axes.c_n {
        c_shape.insert(axes.c_m, m.clone());
        c_shape.insert(axes.c_n, n.clone());
    } else {
        c_shape.insert(axes.c_n, n.clone());
        c_shape.insert(axes.c_m, m.clone());
    }
    Ok((m, ka, n, c_shape))
}

pub fn output_type(input: DatumType) -> DatumType {
    if input.is_float() {
        input
    } else {
        i32::datum_type()
    }
}


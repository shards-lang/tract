use tract_core::internal::*;
use tract_ndarray::prelude::*;
use tract_core::ops::matmul::*;

#[test]
fn crasher_monterey_matmul() {
    let mut model = TypedModel::default();
    let wire = model.add_source("input", f32::fact(&[1usize,1])).unwrap();
    let a = model.add_const("a", Tensor::zero::<f32>(&[2,1]).unwrap().into_arc_tensor()).unwrap();
    let axes = MatMulAxes::default_for_rank(2).transposing(false, true, true);
    let op = MatMul { axes };
    let wire = model.wire_node("conv", op, &[a, wire]).unwrap()[0];
    model.set_output_outlets(&[wire]).unwrap();
    let decluttered = model.into_decluttered().unwrap();
    let optimized = decluttered.into_optimized().unwrap();
}

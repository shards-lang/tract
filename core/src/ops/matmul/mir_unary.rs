use super::lir_unary::{ConcreteMatMulGeometry, LirMatMulUnary, MatMulGeometry, ProtoFusedSpec};
use super::*;
use crate::internal::*;
// use crate::ops::array::TypedConcat;
use tract_ndarray::prelude::*;

/// The pseudo Unary matrix multiplier. A is constant, B is the input
#[derive(Debug, Clone, new, Hash)]
pub struct MatMulUnary {
    pub a: Arc<Tensor>,
    pub axes: MatMulAxes,
}

impl_dyn_hash!(MatMulUnary);

impl Op for MatMulUnary {
    fn name(&self) -> Cow<str> {
        "MatMulUnary".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MatMulUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let t = eval(&self.a, &inputs[0], self.axes)?;
        Ok(tvec!(t.into()))
    }
}

impl TypedOp for MatMulUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs[0].rank() == self.a.rank(),
            "Inconsistent matmul between input {:?} and attribute {:?} (rank mismatch)",
            inputs[0],
            self.a
        );
        let (_m, _k, _n, c_shape) = compute_shape(
            &self.a.shape().iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            &inputs[0].shape,
            self.axes,
        )?;
        let c_dt = output_type(inputs[0].datum_type);
        Ok(tvec!(c_dt.fact(c_shape)))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let b = args_1!(model.node_input_facts(node.id)?);
        if let Some(b_shape) = b.shape.as_concrete() {
            let patch = self.new_mat_mul_unary_finite(model, node, b_shape, b.datum_type)?;
	    std::mem::drop(patch);
	    Ok(None)
        } else {
            Ok(None)
        }
    }

    as_op!();
}

impl MatMulUnary {
    fn new_mat_mul_unary_finite(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        b_shape: &[usize],
        b_dt: DatumType,
    ) -> TractResult<TypedModelPatch> {
        let mut patch = TypedModelPatch::default();
        let mut wire = patch.tap_model(model, node.inputs[0])?;

        let c_dt = output_type(self.a.datum_type());
        let (m, k, n, c_shape) = compute_shape(self.a.shape(), b_shape, self.axes)?;

        let mmm = tract_linalg::ops()
            .mmm(self.a.datum_type(), b_dt, c_dt, Some(m), Some(k), Some(n))
            .with_context(|| {
                format!(
                    "No matrix multiplier for {:?}x{:?} to {:?}",
                    self.a.datum_type(),
                    b_dt,
                    c_dt
                )
            })?;

        let mut a_iter_shape: TVec<usize> = self.a.shape().into();
        a_iter_shape[self.axes.a_m] = 1;
        a_iter_shape[self.axes.a_k] = 1;

        let packed_as = Array::from_shape_fn(&*a_iter_shape, |a_prefix| unsafe {
            let offset = a_prefix
                .as_array_view()
                .iter()
                .zip(self.a.strides())
                .map(|(x, s)| *x as isize * s)
                .sum::<isize>()
                * self.a.datum_type().size_of() as isize;
            let mut pa = Tensor::zero_aligned_dt(
                self.a.datum_type(),
                &[mmm.a_pack().len(k, m)],
                mmm.a_pack().alignment(),
            )
            .unwrap();
/*
            mmm.a_pack().pack(
                &mut pa.view_mut(),
                TensorView::from_bytes(&self.a, offset, self.a.shape(), self.a.strides()),
                self.axes.a_k,
                self.axes.a_m,
            );
*/
//            (pa.into_arc_tensor(), vec![ProtoFusedSpec::Store])
            (pa.into_arc_tensor(), vec![
//		ProtoFusedSpec::BinScalar(AttrOrInput::Attr(rctensor0(0.0f32)), tract_linalg::mmm::BinOp::Add),
//		ProtoFusedSpec::Store,
		ProtoFusedSpec::Store,
	    ])
        });

        unsafe {
            let mut packed_b_shape: TVec<usize> = b_shape.into();
            packed_b_shape.remove(self.axes.b_k.max(self.axes.b_n));
            packed_b_shape.remove(self.axes.b_k.min(self.axes.b_n));
            packed_b_shape.push(mmm.b_pack().len(k, n));
            wire = patch.wire_node(
                format!("{}.pack", &*node.name),
                super::MatMatMulPack {
                    packer: mmm.b_pack(),
                    k_axis: self.axes.b_k,
                    mn_axis: self.axes.b_n,
                },
                &[wire],
            )?[0];
            let b_storage = mmm.b_packed(b_dt.size_of(), k);
            let geometry = ConcreteMatMulGeometry { m, k, n, b_storage };

               let op = LirMatMulUnary {
                    c_fact: c_dt.fact(&c_shape),
                    geometry: MatMulGeometry::Concrete(geometry),
                    micro_ops: packed_as,
                    c_m_axis: self.axes.c_m,
                    c_n_axis: self.axes.c_n,
                    c_final_shape: c_shape.into(),
//                    reshape_post: vec![],
                    mmm,
                };
		wire = patch.wire_node(
                format!("{}.matmatmul", &*node.name),
op, &[wire])?[0];
//	    dbg!("done wire_node");

 //           patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
 //           patch.obliterate(node.id)?;
        }

        Ok(patch)
    }

/*
    fn declutter_precusor_is_concat(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(concat) = model.nodes()[node.inputs[0].node].op().downcast_ref::<TypedConcat>()
        {
            let mut patch = TypedModelPatch::new("split over k-concatenated input");
            if concat.axis == self.axes.b_k {
                let concat_node = model.node(node.inputs[0].node);
                let offsets = concat
                    .offsets(&model.node_input_facts(concat_node.id)?)?
                    .iter()
                    .map(|x| x.to_usize())
                    .collect::<TractResult<Vec<usize>>>()?;
                let mut wires = vec![];
                for (ix, input) in concat_node.inputs.iter().enumerate() {
                    let wire = patch.tap_model(model, *input)?;
                    let a = self.a.slice(self.axes.a_k, offsets[ix], offsets[ix + 1])?;
                    let wire = patch.wire_node(
                        format!("{}.k-{}-{}", node.name, offsets[ix], offsets[ix + 1]),
                        MatMulUnary { a: a.into_arc_tensor(), ..self.clone() },
                        &[wire],
                    )?[0];
                    wires.push(wire)
                }
                let mut wire = wires[0];
                for (ix, w) in wires[1..].iter().enumerate() {
                    wire = patch.wire_node(
                        format!("{}.k-add-{}", node.name, ix),
                        crate::ops::binary::TypedBinOp(Box::new(crate::ops::math::Add)),
                        &[wire, *w],
                    )?[0];
                }
                patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }
*/
}


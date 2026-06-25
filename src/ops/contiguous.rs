use crate::{Op, Tensor};

#[derive(Debug)]
pub struct ContiguousOp;

impl Op for ContiguousOp {
    fn name(&self) -> &'static str {
        "Contiguous"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // The derivative of a memory copy is the identity function.
        // We just pass the contiguous gradient straight through to the non-contiguous parent!
        // Downstream operations (like TransposeOp) will correctly swap the strides
        // of this contiguous gradient to make it non-contiguous again.
        vec![Some(grad_output.clone())]
    }
}

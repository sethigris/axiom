use crate::{Op, Tensor};

#[derive(Debug)]
pub struct RoPEOp {
    pub dim: usize,
    pub base: f32,
}

impl Op for RoPEOp {
    fn name(&self) -> &'static str {
        "RoPE"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let dx = crate::kernels::rope_backward(grad_output, self.dim, self.base);
        vec![Some(dx)]
    }
}

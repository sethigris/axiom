use crate::Tensor;
use std::sync::Arc;

pub struct RoPE {
    pub dim: usize,
    pub base: f32,
}

impl RoPE {
    pub fn new(dim: usize, base: f32) -> Self {
        Self { dim, base }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = crate::kernels::rope_forward(x, self.dim, self.base);

        if x.requires_grad {
            let op = Arc::new(crate::ops::rope::RoPEOp {
                dim: self.dim,
                base: self.base,
            });
            out.with_node(op, vec![x.clone()])
        } else {
            out
        }
    }
}

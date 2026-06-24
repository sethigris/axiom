use crate::{Op, Tensor};

#[derive(Debug)]
pub struct MatMulOp {
    pub a: Tensor,
    pub b: Tensor,
}

impl Op for MatMulOp {
    fn name(&self) -> &'static str {
        "MatMul"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Calculus for MatMul: Y = A @ B
        // dL/dA = dL/dY @ B^T
        let grad_a = grad_output.matmul(&self.b.t());

        // dL/dB = A^T @ dL/dY
        let grad_b = self.a.t().matmul(grad_output);

        vec![Some(grad_a), Some(grad_b)]
    }
}

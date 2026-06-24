use crate::{Op, Tensor};

#[derive(Debug)]
pub struct FusedLinearOp {
    pub x: Tensor,
    pub w: Tensor,
    pub bias: Tensor,
}

impl Op for FusedLinearOp {
    fn name(&self) -> &'static str {
        "FusedLinear"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Sync to CPU to easily handle broadcasting during recomputation.
        // This avoids needing a broadcast-aware GPU add shader.
        let x = self.x.ensure_cpu();
        let w = self.w.ensure_cpu();
        let bias = self.bias.ensure_cpu();
        let grad_output = grad_output.ensure_cpu();

        // Recompute pre-activation
        let pre_act = x.matmul(&w).add(&bias);

        // Apply ReLU backward
        let grad_pre_act = crate::kernels::activations::relu_backward(&grad_output, &pre_act);

        // Calculus for Linear Layer
        let grad_bias = crate::kernels::sum_axis(&grad_pre_act, 0);
        let grad_w = x.t().matmul(&grad_pre_act);
        let grad_x = grad_pre_act.matmul(&w.t());

        vec![Some(grad_x), Some(grad_w), Some(grad_bias)]
    }
}

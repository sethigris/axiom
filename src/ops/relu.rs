use crate::{Op, Tensor};

#[derive(Debug)]
pub struct ReluOp {
    pub input: Tensor,
}

impl Op for ReluOp {
    fn name(&self) -> &'static str {
        "ReLU"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let grad_input = crate::kernels::activations::relu_backward(grad_output, &self.input);
        vec![Some(grad_input)]
    }
}

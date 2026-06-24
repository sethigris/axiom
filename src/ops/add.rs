use crate::{Op, Shape, Tensor};

#[derive(Debug)]
pub struct AddOp {
    pub a_shape: Shape,
    pub b_shape: Shape,
}

impl Op for AddOp {
    fn name(&self) -> &'static str {
        "Add"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let mut grad_a = grad_output.clone();
        let mut grad_b = grad_output.clone();

        // If the gradient rank is higher than the original input rank,
        // it means dimensions were prepended (broadcasted). Sum them away!
        while grad_a.rank() > self.a_shape.rank() {
            grad_a = crate::kernels::sum_axis(&grad_a, 0);
        }
        while grad_b.rank() > self.b_shape.rank() {
            grad_b = crate::kernels::sum_axis(&grad_b, 0);
        }

        vec![Some(grad_a), Some(grad_b)]
    }
}

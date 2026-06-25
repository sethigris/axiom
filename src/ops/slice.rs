use crate::{Op, Shape, Tensor};

#[derive(Debug)]
pub struct GetSliceOp {
    pub batch_idx: usize,
    pub parent_shape: Shape,
}

impl Op for GetSliceOp {
    fn name(&self) -> &'static str {
        "GetSlice"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Create a 3D zero-tensor to hold the accumulated gradient
        let mut parent_grad = Tensor::zeros(grad_output.dtype, self.parent_shape.clone());

        // Temporarily disable requires_grad to prevent recursion
        // when we call get_2d_slice below!
        let original_req = parent_grad.requires_grad;
        parent_grad.requires_grad = false;

        // Get the raw slice to copy data into
        let out_slice = parent_grad.get_2d_slice(self.batch_idx);

        // Restore original state
        parent_grad.requires_grad = original_req;

        // Physically copy the 2D gradient into the 3D zero-tensor slice
        crate::kernels::copy(grad_output, &out_slice);

        vec![Some(parent_grad)]
    }
}

use crate::{DType, Op, Shape, Tensor, kernels};

#[derive(Debug)]
pub struct SumOp {
    pub input_shape: Shape,
    pub dtype: DType,
}

impl Op for SumOp {
    fn name(&self) -> &'static str {
        "Sum"
    }

    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // The local derivative of sum(x) is a tensor of 1s matching the input shape.
        let ones = Tensor::zeros(self.dtype, self.input_shape.clone());

        match self.dtype {
            DType::F32 => kernels::fill(&ones, 1.0f32),
            DType::F64 => kernels::fill(&ones, 1.0f64),
            _ => panic!("Unsupported dtype for SumOp backward"),
        }

        // Chain rule: multiply the 1s by the scalar grad_output.
        // Our broadcasted add kernel automatically stretches the scalar to match `ones`!
        let grad_input = kernels::add(&ones, grad_output);
        vec![Some(grad_input)]
    }
}

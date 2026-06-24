use crate::{DType, Tensor};
use rayon::prelude::*;

// (Include SyncPtr and SyncMutPtr definitions here, same as binary.rs)
struct SyncPtr<T>(*const T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    #[inline(always)]
    fn get(&self) -> *const T {
        self.0
    }
}

struct SyncMutPtr<T>(*mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}
impl<T> SyncMutPtr<T> {
    #[inline(always)]
    fn get(&self) -> *mut T {
        self.0
    }
}

pub fn relu(tensor: &Tensor) -> Tensor {
    let out = Tensor::empty(tensor.dtype, tensor.shape.clone());
    let num_elements = tensor.shape.num_elements();

    match tensor.dtype {
        DType::F32 => relu_typed::<f32>(tensor, &out, num_elements),
        DType::F64 => relu_typed::<f64>(tensor, &out, num_elements),
        _ => panic!("Unsupported dtype for relu"),
    }
    out
}

fn relu_typed<T: bytemuck::Pod + PartialOrd + Default + Copy>(
    tensor: &Tensor,
    out: &Tensor,
    num_elements: usize,
) {
    let in_ptr = SyncPtr(tensor.storage.as_ptr() as *const T);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut T);

    (0..num_elements).into_par_iter().for_each(|i| unsafe {
        let val = *in_ptr.get().add(i);
        let zero = T::default();
        *out_ptr.get().add(i) = if val > zero { val } else { zero };
    });
}

/// ReLU backward is just passing the gradient through if input > 0, else 0.
pub fn relu_backward(grad_output: &Tensor, input: &Tensor) -> Tensor {
    let out = Tensor::empty(grad_output.dtype, grad_output.shape.clone());
    let num_elements = grad_output.shape.num_elements();

    match grad_output.dtype {
        DType::F32 => relu_backward_typed::<f32>(grad_output, input, &out, num_elements),
        DType::F64 => relu_backward_typed::<f64>(grad_output, input, &out, num_elements),
        _ => panic!("Unsupported dtype for relu_backward"),
    }
    out
}

fn relu_backward_typed<T: bytemuck::Pod + PartialOrd + Default + Copy>(
    grad: &Tensor,
    input: &Tensor,
    out: &Tensor,
    num_elements: usize,
) {
    let grad_ptr = SyncPtr(grad.storage.as_ptr() as *const T);
    let in_ptr = SyncPtr(input.storage.as_ptr() as *const T);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut T);

    (0..num_elements).into_par_iter().for_each(|i| unsafe {
        let g = *grad_ptr.get().add(i);
        let x = *in_ptr.get().add(i);
        let zero = T::default();
        *out_ptr.get().add(i) = if x > zero { g } else { zero };
    });
}

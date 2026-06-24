use crate::{DType, Device, GpuContext, Tensor};
use rayon::prelude::*;

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

/// In-place Stochastic Gradient Descent step.
/// w = w - lr * grad_w
pub fn sgd_step(weight: &Tensor, grad: &Tensor, lr: f32) {
    assert_eq!(
        weight.shape, grad.shape,
        "Weight and Gradient shapes must match"
    );
    assert_eq!(weight.dtype, DType::F32, "SGD currently only supports F32");

    // If weight is on GPU, we MUST run the GPU shader to mutate it in place in VRAM.
    if weight.device.is_gpu() {
        // If the gradient is on CPU (from our backward pass), upload it to the GPU!
        let grad_gpu = if grad.device.is_gpu() {
            grad.clone()
        } else {
            grad.to(weight.device.clone())
        };

        let ctx = match &weight.device {
            Device::Gpu(c) => c.clone(),
            _ => unreachable!(),
        };
        GpuContext::sgd_step_gpu(&ctx, weight, &grad_gpu, lr);
        return;
    }

    // CPU Fallback
    let num_elements = weight.shape.num_elements();
    let w_ptr = SyncMutPtr(weight.storage.as_mut_ptr() as *mut f32);
    let g_ptr = SyncPtr(grad.storage.as_ptr() as *const f32);

    (0..num_elements).into_par_iter().for_each(|i| unsafe {
        let w = w_ptr.get().add(i);
        let g = *g_ptr.get().add(i);
        *w = *w - (lr * g);
    });
}

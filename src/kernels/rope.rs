use crate::Tensor;
use crate::kernels::activations::{SyncMutPtr, SyncPtr};
use rayon::prelude::*;

pub fn rope_forward(x: &Tensor, dim: usize, base: f32) -> Tensor {
    let x = x.ensure_cpu();
    let out = Tensor::empty(x.dtype, x.shape.clone());

    // Expecting shape [..., Seq, Dim]
    let shape = x.shape.dims();
    let rank = shape.len();
    let seq_len = shape[rank - 2];
    let head_dim = shape[rank - 1];

    assert_eq!(head_dim, dim, "RoPE dimension mismatch");
    assert_eq!(head_dim % 2, 0, "RoPE requires an even head dimension");

    let num_tokens = x.shape.num_elements() / head_dim;

    let x_ptr = SyncPtr(x.storage.as_ptr() as *const f32);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut f32);

    (0..num_tokens).into_par_iter().for_each(|token_idx| {
        let m = (token_idx % seq_len) as f32; // The sequence position
        let offset = token_idx * head_dim;

        for i in 0..(head_dim / 2) {
            let exponent = -((2 * i) as f32) / (head_dim as f32);
            let theta = m * base.powf(exponent);
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let x0 = unsafe { *x_ptr.get().add(offset + 2 * i) };
            let x1 = unsafe { *x_ptr.get().add(offset + 2 * i + 1) };

            unsafe {
                *out_ptr.get().add(offset + 2 * i) = x0 * cos_t - x1 * sin_t;
                *out_ptr.get().add(offset + 2 * i + 1) = x0 * sin_t + x1 * cos_t;
            }
        }
    });

    out
}

pub fn rope_backward(grad: &Tensor, _dim: usize, base: f32) -> Tensor {
    // The gradient of an orthogonal rotation is the inverse rotation (negative angle)
    let grad = grad.ensure_cpu();
    let out = Tensor::empty(grad.dtype, grad.shape.clone());

    let shape = grad.shape.dims();
    let rank = shape.len();
    let seq_len = shape[rank - 2];
    let head_dim = shape[rank - 1];

    let num_tokens = grad.shape.num_elements() / head_dim;

    let g_ptr = SyncPtr(grad.storage.as_ptr() as *const f32);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut f32);

    (0..num_tokens).into_par_iter().for_each(|token_idx| {
        let m = (token_idx % seq_len) as f32;
        let offset = token_idx * head_dim;

        for i in 0..(head_dim / 2) {
            let exponent = -((2 * i) as f32) / (head_dim as f32);
            let theta = -(m * base.powf(exponent));
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let g0 = unsafe { *g_ptr.get().add(offset + 2 * i) };
            let g1 = unsafe { *g_ptr.get().add(offset + 2 * i + 1) };

            unsafe {
                *out_ptr.get().add(offset + 2 * i) = g0 * cos_t - g1 * sin_t;
                *out_ptr.get().add(offset + 2 * i + 1) = g0 * sin_t + g1 * cos_t;
            }
        }
    });

    out
}

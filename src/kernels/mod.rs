use crate::Tensor;

pub mod activations;
pub mod binary;
pub mod matmul;
pub mod reduce;

pub use binary::add;
pub use matmul::matmul;
pub use reduce::{sum_all, sum_axis};

/// A recursive helper to iterate over an N-dimensional shape,
/// respecting arbitrary strides. This avoids allocating coordinate arrays.
fn strided_loop<F: FnMut(usize)>(
    dim: usize,
    shape: &[usize],
    strides: &[isize],
    current_offset: usize,
    f: &mut F,
) {
    if dim == shape.len() {
        f(current_offset);
        return;
    }

    let stride = strides[dim] as usize;
    for i in 0..shape[dim] {
        strided_loop(dim + 1, shape, strides, current_offset + i * stride, f);
    }
}

/// Fills a tensor with a specific value.
/// Notice how we handle raw pointers: we cast the byte pointer to a typed pointer,
/// allowing us to use standard pointer arithmetic (`add`) safely.
pub fn fill<T: bytemuck::Pod>(tensor: &Tensor, value: T) {
    let shape = tensor.shape.dims();
    let strides = tensor.strides.steps();

    // Get the raw mutable byte pointer, shift it to our view's offset,
    // and cast it to the correct typed pointer.
    let base_byte_ptr = unsafe { tensor.storage.as_mut_ptr().add(tensor.byte_offset) };
    let typed_ptr = base_byte_ptr as *mut T;

    let mut write_val = |offset: usize| {
        unsafe {
            // Because `typed_ptr` is `*mut T`, `.add(offset)` correctly advances
            // the pointer by `offset * size_of::<T>()` bytes.
            typed_ptr.add(offset).write(value);
        }
    };

    strided_loop(0, shape, strides, 0, &mut write_val);
}

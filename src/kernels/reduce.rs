use crate::{DType, Shape, Tensor};
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

/// Reduces the entire tensor to a single scalar value (the sum of all elements).
pub fn sum_all(tensor: &Tensor) -> Tensor {
    let tensor = tensor.ensure_cpu(); // Auto-download if on GPU
    let num_elements = tensor.shape.num_elements();
    // A scalar tensor has rank 0, represented by an empty shape vector.
    let out = Tensor::zeros(tensor.dtype, Shape::new(Vec::<usize>::new()));

    match tensor.dtype {
        DType::F32 => sum_typed::<f32>(&tensor, &out, num_elements),
        DType::F64 => sum_typed::<f64>(&tensor, &out, num_elements),
        _ => panic!("Unsupported dtype for sum"),
    }
    out
}

fn sum_typed<T: bytemuck::Pod + std::ops::Add<Output = T> + Copy + Default>(
    tensor: &Tensor,
    out: &Tensor,
    num_elements: usize,
) {
    let shape = tensor.shape.dims();
    let strides = tensor.strides.steps();
    let base = tensor.byte_offset / std::mem::size_of::<T>();

    let in_ptr = SyncPtr(tensor.storage.as_ptr() as *const T);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut T);

    let mut total = T::default();

    // Sequential accumulation. Since we are reducing to a single scalar,
    // parallelizing this requires atomic operations or tree reductions,
    // which we will add later. For now, sequential is correct.
    for i in 0..num_elements {
        let mut offset = 0isize;
        let mut idx = i;
        for d in (0..shape.len()).rev() {
            let dim_size = shape[d];
            let coord = idx % dim_size;
            idx /= dim_size;
            offset += coord as isize * strides[d];
        }

        unsafe {
            total = total + *in_ptr.get().add(offset as usize + base);
        }
    }

    unsafe {
        *out_ptr.get() = total;
    }
}

/// Reduces a tensor along a specific axis, dropping that dimension.
pub fn sum_axis(tensor: &Tensor, axis: usize) -> Tensor {
    let tensor = tensor.ensure_cpu(); // Auto-download if on GPU
    let in_shape = tensor.shape.dims();
    let mut out_dims = in_shape.to_vec();
    let axis_size = out_dims.remove(axis);

    if out_dims.is_empty() {
        return sum_all(&tensor);
    }

    let out_shape = Shape::new(out_dims);

    // FIX: Calculate num_out BEFORE moving out_shape into Tensor::zeros
    let num_out = out_shape.num_elements();
    let out = Tensor::zeros(tensor.dtype, out_shape); // Shape is moved here

    match tensor.dtype {
        DType::F32 => sum_axis_typed::<f32>(&tensor, &out, num_out, axis, axis_size),
        DType::F64 => sum_axis_typed::<f64>(&tensor, &out, num_out, axis, axis_size),
        _ => panic!("Unsupported dtype for sum_axis"),
    }
    out
}

fn sum_axis_typed<T: bytemuck::Pod + std::ops::Add<Output = T> + Copy + Default>(
    tensor: &Tensor,
    out: &Tensor,
    num_out: usize,
    axis: usize,
    axis_size: usize,
) {
    let in_shape = tensor.shape.dims();
    let in_strides = tensor.strides.steps();
    let in_base = tensor.byte_offset / std::mem::size_of::<T>();

    let out_shape = out.shape.dims();
    let out_strides = out.strides.steps();
    let out_base = out.byte_offset / std::mem::size_of::<T>();

    let in_ptr = SyncPtr(tensor.storage.as_ptr() as *const T);
    let out_ptr = SyncMutPtr(out.storage.as_mut_ptr() as *mut T);

    (0..num_out).into_par_iter().for_each(|i| {
        // Stack-allocated coordinate array to avoid heap allocation in the hot loop.
        // 8 dimensions is enough for almost any neural network architecture.
        let mut out_coords = [0usize; 8];
        let mut temp = i;
        for d in (0..out_shape.len()).rev() {
            out_coords[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }

        let mut sum = T::default();

        for k in 0..axis_size {
            let mut in_offset = 0isize;
            // Map output coordinates back to input coordinates, inserting `k` at the reduced `axis`.
            for d in 0..in_shape.len() {
                let coord = if d < axis {
                    out_coords[d]
                } else if d == axis {
                    k
                } else {
                    out_coords[d - 1]
                };
                in_offset += coord as isize * in_strides[d];
            }

            unsafe {
                sum = sum + *in_ptr.get().add(in_offset as usize + in_base);
            }
        }

        let mut out_offset = 0isize;
        for d in 0..out_shape.len() {
            out_offset += out_coords[d] as isize * out_strides[d];
        }

        unsafe {
            *out_ptr.get().add(out_offset as usize + out_base) = sum;
        }
    });
}

use crate::{CpuStorage, DType, Shape, Tensor, TensorId};
use memmap2::Mmap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

/// A dataset backed by a memory-mapped binary file.
/// File format: [num_samples: u64][feature_size: u64][raw f32 data...]
pub struct MmapDataset {
    mmap: Arc<Mmap>,
    pub num_samples: usize,
    pub feature_size: usize,
}

impl MmapDataset {
    /// Helper to generate a dummy binary dataset file on disk.
    pub fn create_dummy_file(path: &str, num_samples: usize, feature_size: usize) {
        let mut file = File::create(path).unwrap();
        file.write_all(&(num_samples as u64).to_le_bytes()).unwrap();
        file.write_all(&(feature_size as u64).to_le_bytes())
            .unwrap();

        for i in 0..(num_samples * feature_size) {
            file.write_all(&((i as f32) * 0.01).to_le_bytes()).unwrap();
        }
    }

    /// Opens the file and maps it into virtual memory. Zero data is copied into RAM here!
    pub fn open(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        let ptr = mmap.as_ptr();

        // Read the header using read_unaligned to prevent CPU alignment faults.
        let num_samples = unsafe { std::ptr::read_unaligned(ptr as *const u64) } as usize;
        let feature_size = unsafe { std::ptr::read_unaligned(ptr.add(8) as *const u64) } as usize;

        Self {
            mmap: Arc::new(mmap),
            num_samples,
            feature_size,
        }
    }
    /// Returns a zero-copy Tensor view of a specific batch.
    pub fn get_batch(&self, start: usize, batch_size: usize) -> Tensor {
        assert!(start + batch_size <= self.num_samples);

        // Header is 16 bytes (two u64s).
        let data_start_byte = 16 + (start * self.feature_size * std::mem::size_of::<f32>());
        let shape = Shape::new([batch_size, self.feature_size]);

        // FIX: Calculate strides BEFORE moving shape into the Tensor struct!
        let strides = shape.contiguous_strides();

        Tensor {
            id: TensorId::next(),
            dtype: DType::F32,
            shape, // Shape is moved here
            strides,
            storage: Arc::new(CpuStorage::from_mmap(self.mmap.clone())),
            byte_offset: data_start_byte,
            requires_grad: false,
            grad: None,
            node: None,
            device: crate::device::Device::Cpu,
        }
    }
}

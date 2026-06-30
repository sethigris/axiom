use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use wgpu::{BindingResource, Buffer, BufferBinding, BufferUsages, Device};

/// SYSTEMS ENGINEERING RATIONALE (The Binning Slab Allocator):
/// A naive caching allocator that caches exact byte sizes suffers from severe
/// internal fragmentation and cache misses. If a tensor of 4,000 bytes is freed,
/// and the next tensor requests 4,010 bytes, the cache misses, forcing an expensive
/// driver-level VRAM allocation. By rounding all requests up to the nearest power
/// of two (with a 512-byte floor to prevent micro-fragmentation), we group similar
/// tensor sizes into the same memory pool. This dramatically increases cache hit
/// rates and eliminates driver allocation overhead during the training loop.
pub struct GpuMemoryArena {
    // Maps the binned slab size in bytes to a list of free buffers of that exact capacity.
    free_list: Mutex<HashMap<u64, Vec<Arc<Buffer>>>>,
}

impl GpuMemoryArena {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            free_list: Mutex::new(HashMap::new()),
        })
    }

    pub fn allocate(self: &Arc<Self>, device: &Device, requested_bytes: u64) -> GpuAllocation {
        // VETERAN SYSTEMS NOTE:
        // We immediately round the requested bytes to the nearest power-of-two bin.
        // This ensures that a 4,000-byte tensor and a 4,010-byte tensor both request
        // a 4,096-byte slab. This dramatically increases our cache hit rate and
        // prevents the GPU driver from fragmenting VRAM with thousands of
        // uniquely-sized micro-allocations.
        let binned_size = get_bin_size(requested_bytes);

        // 1. Check the cache for a recycled slab of this binned size
        let mut cache = self.free_list.lock().unwrap();
        if let Some(buffers) = cache.get_mut(&binned_size) {
            if let Some(buffer) = buffers.pop() {
                return GpuAllocation {
                    buffer,
                    size: binned_size, // Store the slab capacity for deterministic RAII reclamation
                    arena: Arc::downgrade(self),
                };
            }
        }
        // CRITICAL SYSTEMS RULE: Unlock the mutex before calling the slow GPU driver!
        // Holding a lock across a blocking hardware API call is a guaranteed deadlock trap.
        drop(cache);

        // 2. Cache miss! Ask the GPU driver to allocate a new slab of the binned size.
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuMemoryArena Binned Slab"),
            size: binned_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuAllocation {
            buffer: Arc::new(buffer),
            size: binned_size, // Store the slab capacity
            arena: Arc::downgrade(self),
        }
    }

    pub fn free(&self, buffer: Arc<Buffer>, binned_size: u64) {
        // VETERAN SYSTEMS NOTE:
        // The `binned_size` passed here is already the rounded slab capacity
        // stored in the GpuAllocation struct. We route it directly back into
        // its specific power-of-two bin pool.
        let mut cache = self.free_list.lock().unwrap();
        cache
            .entry(binned_size)
            .or_insert_with(Vec::new)
            .push(buffer);
    }

    pub fn reset(&self) {
        // Nuclear option: Wipe the entire cache. Used primarily during testing
        // or when explicitly tearing down the GPU context.
        self.free_list.lock().unwrap().clear();
    }
}

/// Calculates the optimal memory bin size for a given allocation request.
fn get_bin_size(requested_bytes: u64) -> u64 {
    const MIN_BIN_SIZE: u64 = 512;
    if requested_bytes <= MIN_BIN_SIZE {
        MIN_BIN_SIZE
    } else {
        requested_bytes.next_power_of_two()
    }
}

impl std::fmt::Debug for GpuMemoryArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cache = self.free_list.lock().unwrap();
        let total_cached_buffers: usize = cache.values().map(|v| v.len()).sum();
        f.debug_struct("GpuMemoryArena")
            .field("cached_unique_bins", &cache.keys().len())
            .field("total_free_slabs", &total_cached_buffers)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct GpuAllocation {
    pub buffer: Arc<Buffer>,
    /// The allocated slab capacity (binned size), not necessarily the logical tensor size.
    pub size: u64,
    arena: Weak<GpuMemoryArena>,
}

// THE KEY FEATURE: Deterministic, Reference-Counted RAII VRAM Reclamation.
impl Drop for GpuAllocation {
    fn drop(&mut self) {
        // VETERAN SYSTEMS NOTE (The Arc Reference Counting Trap):
        // GpuAllocation derives `Clone` because Tensors are constantly cloned
        // for the Autograd computational graph. If we unconditionally return the
        // buffer to the arena on `drop`, the buffer will be recycled while the
        // Autograd graph is still holding a reference to it. The next forward pass
        // will overwrite the VRAM, destroying the gradients and causing catastrophic
        // network collapse (the 0.625 variance floor).
        //
        // We must ONLY return the buffer to the slab pool when the LAST Arc
        // reference is being dropped. If strong_count is 1, this current instance
        // is the sole owner of the buffer.
        if Arc::strong_count(&self.buffer) == 1 {
            if let Some(arena) = self.arena.upgrade() {
                arena.free(self.buffer.clone(), self.size);
            }
        }
    }
}

impl GpuAllocation {
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            // We bind the entire binned slab capacity. The WGSL shaders are
            // mathematically responsible for respecting the logical tensor
            // dimensions passed via push constants or uniform buffers.
            size: std::num::NonZeroU64::new(self.size),
        })
    }
}

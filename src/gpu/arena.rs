use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use wgpu::{BindingResource, Buffer, BufferBinding, BufferUsages, Device};

pub struct GpuMemoryArena {
    // Maps buffer size in bytes to a list of free buffers of that exact size
    free_list: Mutex<HashMap<u64, Vec<Arc<Buffer>>>>,
}

impl GpuMemoryArena {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            free_list: Mutex::new(HashMap::new()),
        })
    }

    pub fn allocate(self: &Arc<Self>, device: &Device, size_bytes: u64) -> GpuAllocation {
        // 1. Check the cache for a recycled buffer of this exact size
        let mut cache = self.free_list.lock().unwrap();
        if let Some(buffers) = cache.get_mut(&size_bytes) {
            if let Some(buffer) = buffers.pop() {
                return GpuAllocation {
                    buffer,
                    size: size_bytes,
                    arena: Arc::downgrade(self), // <--- FIXED
                };
            }
        }
        drop(cache); // Unlock before calling the slow GPU driver

        // 2. Cache miss! Ask the GPU driver to allocate a new buffer.
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuMemoryArena Cached Buffer"),
            size: size_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuAllocation {
            buffer: Arc::new(buffer),
            size: size_bytes,
            arena: Arc::downgrade(self), // <--- FIXED
        }
    }

    pub fn free(&self, buffer: Arc<Buffer>, size: u64) {
        let mut cache = self.free_list.lock().unwrap();
        cache.entry(size).or_insert_with(Vec::new).push(buffer);
    }

    pub fn reset(&self) {
        self.free_list.lock().unwrap().clear();
    }
}

impl std::fmt::Debug for GpuMemoryArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cache = self.free_list.lock().unwrap();
        let total_cached_buffers: usize = cache.values().map(|v| v.len()).sum();
        f.debug_struct("GpuMemoryArena")
            .field("cached_unique_sizes", &cache.keys().len())
            .field("total_free_buffers", &total_cached_buffers)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct GpuAllocation {
    pub buffer: Arc<Buffer>,
    pub size: u64,
    arena: Weak<GpuMemoryArena>,
}

// THE KILLER FEATURE: Deterministic RAII VRAM Reclamation.
impl Drop for GpuAllocation {
    fn drop(&mut self) {
        if let Some(arena) = self.arena.upgrade() {
            arena.free(self.buffer.clone(), self.size);
        }
    }
}

impl GpuAllocation {
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: 0, // Always 0 now, since it's a dedicated buffer!
            size: std::num::NonZeroU64::new(self.size),
        })
    }
}

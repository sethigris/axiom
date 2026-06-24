use crate::gpu::GpuContext;
use std::sync::Arc;

/// Represents where a tensor's data physically lives.
#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Gpu(Arc<GpuContext>),
}

impl Device {
    pub fn cpu() -> Self {
        Device::Cpu
    }

    pub fn gpu() -> Self {
        Device::Gpu(GpuContext::new())
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu(_))
    }
}

// Manual PartialEq implementation since GpuContext isn't Eq
impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        matches!((self, other), (Device::Cpu, Device::Cpu))
    }
}

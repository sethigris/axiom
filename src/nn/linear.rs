use crate::ops::linear_fused::FusedLinearOp;
use crate::{DType, Device, GpuContext, Shape, Tensor};
use std::sync::Arc;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let bound = (1.0 / in_features as f32).sqrt();

        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| fastrand::f32() * 2.0 * bound - bound)
            .collect();

        let bias_data = vec![0.0f32; out_features];

        let weight = Tensor::from_slice(
            DType::F32,
            Shape::new([in_features, out_features]),
            &weight_data,
        )
        .requires_grad_(true);

        let bias = Tensor::from_slice(DType::F32, Shape::new([out_features]), &bias_data)
            .requires_grad_(true);

        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // DEVICE DISPATCH: If everything is on GPU, use the fused WGSL shader!
        if x.device.is_gpu() && self.weight.device.is_gpu() && self.bias.device.is_gpu() {
            let ctx = match &x.device {
                Device::Gpu(c) => c.clone(),
                _ => unreachable!(),
            };

            let out = GpuContext::fused_linear(&ctx, x, &self.weight, &self.bias);

            // Attach the Autograd node so backprop works
            if x.requires_grad || self.weight.requires_grad || self.bias.requires_grad {
                let op = Arc::new(FusedLinearOp {
                    x: x.clone(),
                    w: self.weight.clone(),
                    bias: self.bias.clone(),
                });
                out.with_node(op, vec![x.clone(), self.weight.clone(), self.bias.clone()])
            } else {
                out
            }
        } else {
            // Fallback to unfused CPU/GPU ops
            let out = x.matmul(&self.weight);
            out.add(&self.bias).relu()
        }
    }
}

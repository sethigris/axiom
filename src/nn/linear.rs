use crate::{Device, GpuContext, Shape, Tensor};
use rand::Rng;
use std::sync::Arc;

/// VETERAN SYSTEMS NOTE (The Linear Layer):
/// The foundational affine transformation layer: y = xW + b.
/// In production LLM engines, this layer is the primary consumer of VRAM and
/// PCIe bandwidth. By supporting BF16 storage with a JIT F32 Autocast boundary,
/// this layer seamlessly participates in Mixed-Precision training without
/// shattering the mathematical stability of the underlying WGSL compute shaders.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// Initializes the Linear layer with Kaiming Uniform (He) Initialization.
    ///
    /// MATHEMATICAL RATIONALE:
    /// Standard normal initialization causes the variance of activations to either
    /// explode or vanish as they propagate through deep Transformer blocks.
    /// Kaiming Uniform scales the random weights based on the `fan_in` (number of
    /// input connections) to strictly preserve activation variance across the depth
    /// of the network, ensuring stable gradient flow during backpropagation.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let fan_in = in_features as f32;

        // Kaiming Uniform Bound for Weights: sqrt(6 / fan_in)
        let weight_bound = (6.0 / fan_in).sqrt();
        // Uniform Bound for Bias: 1 / sqrt(fan_in)
        let bias_bound = 1.0 / fan_in.sqrt();

        let mut rng = rand::thread_rng();

        // Initialize Weights
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.gen_range(-weight_bound..weight_bound))
            .collect();

        // Initialize Bias
        let bias_data: Vec<f32> = (0..out_features)
            .map(|_| rng.gen_range(-bias_bound..bias_bound))
            .collect();

        Self {
            // CRITICAL AUTOGRAD FIX: Explicitly flag weights as trainable so the
            // Autograd engine registers them in the gradient map for the optimizer.
            weight: Tensor::from_slice(
                crate::DType::F32,
                crate::Shape::new([in_features, out_features]),
                &weight_data,
            )
            .requires_grad_(true),
            bias: Tensor::from_slice(
                crate::DType::F32,
                crate::Shape::new([out_features]),
                &bias_data,
            )
            .requires_grad_(true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // VETERAN SYSTEMS NOTE (JIT Autocast Boundary for Mixed Precision):
        // If the model is utilizing Mixed Precision (BF16 storage) to halve the VRAM
        // footprint and PCIe transfer time, the weights will arrive here as 16-bit floats.
        // Because our current WGSL compute shaders strictly require 32-bit floats (f32)
        // for mathematical accumulation and stability, we perform a Just-In-Time upcast.
        // This temporary F32 buffer is used for the compute dispatch and immediately
        // discarded, perfectly mimicking the autocast boundaries of production engines
        // on hardware lacking native 16-bit ALUs.
        let compute_weight = if self.weight.dtype == crate::DType::BF16 {
            self.weight.to_dtype(crate::DType::F32)
        } else {
            self.weight.clone()
        };

        let compute_bias = if self.bias.dtype == crate::DType::BF16 {
            self.bias.to_dtype(crate::DType::F32)
        } else {
            self.bias.clone()
        };

        // Auto-device sync (Ensures x is on the same device as the compute weights)
        let x = x.to(compute_weight.device.clone());

        if x.device.is_gpu() && compute_weight.device.is_gpu() && compute_bias.device.is_gpu() {
            let ctx = match &compute_weight.device {
                Device::Gpu(c) => c.clone(),
                _ => unreachable!(),
            };

            // 1. Flatten 3D [B, S, H] to 2D [B*S, H] for the fused shader
            let original_shape = x.shape.clone();
            let x_2d = if x.rank() == 3 {
                let b = x.shape.dims()[0];
                let s = x.shape.dims()[1];
                let h = x.shape.dims()[2];
                x.view(Shape::new([b * s, h]))
            } else {
                x.clone()
            };

            // 2. Dispatch the Fused Linear Shader (x @ w + b) using the F32 compute tensors
            let out_2d = GpuContext::fused_linear(&ctx, &x_2d, &compute_weight, &compute_bias);

            // 3. Attach the FusedLinearOp to the 2D tensors for perfect Autograd tracking
            // VETERAN AUTOGRAD NOTE: We track the original weights (self.weight) in the
            // computational graph, not the temporary F32 compute weights. This ensures that
            // when the backward pass fires, the gradients are correctly routed back to the
            // Master Weights (F32) or the BF16 model weights, preserving the Autograd chain.
            let out_2d_tracked =
                if x_2d.requires_grad || self.weight.requires_grad || self.bias.requires_grad {
                    let op = Arc::new(crate::ops::linear_fused::FusedLinearOp {
                        x: x_2d.clone(),
                        w: self.weight.clone(),
                        bias: self.bias.clone(),
                    });
                    out_2d.with_node(
                        op,
                        vec![x_2d.clone(), self.weight.clone(), self.bias.clone()],
                    )
                } else {
                    out_2d
                };

            // 4. Reshape back to 3D [B, S, N] using view()
            if original_shape.rank() == 3 {
                let b = original_shape.dims()[0];
                let s = original_shape.dims()[1];
                let n = compute_weight.shape.dims()[1];
                out_2d_tracked.view(Shape::new([b, s, n]))
            } else {
                out_2d_tracked
            }
        } else {
            // CPU Fallback (Uses compute_weight to ensure F32 math)
            let out = x.matmul(&compute_weight);
            out.add(&compute_bias)
        }
    }

    /// Recursively migrates the layer's parameters to the specified device.
    ///
    /// SYSTEMS ENGINEERING NOTE:
    /// This enables the `model.to(device)` paradigm, allowing downstream consumers
    /// to seamlessly move entire architectures to the GPU without manually chasing
    /// every individual weight tensor.
    pub fn to(&mut self, device: crate::Device) {
        self.weight = self.weight.to(device.clone());
        self.bias = self.bias.to(device.clone());
    }
}

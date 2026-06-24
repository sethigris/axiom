use axiom::{DType, Device, Linear, Shape, Tensor, optim};
use std::time::Instant;

fn main() {
    println!("Full GPU Training Loop (with GPU SGD)\n");

    let gpu_device = Device::gpu();

    let batch_size = 256;
    let in_features = 512;
    let hidden_features = 512;
    let out_features = 512;
    let lr = 0.01;

    let layer1 = Linear::new(in_features, hidden_features);
    let layer2 = Linear::new(hidden_features, out_features);

    // Move weights to GPU
    let w1_gpu = layer1.weight.to(gpu_device.clone());
    let b1_gpu = layer1.bias.to(gpu_device.clone());
    let w2_gpu = layer2.weight.to(gpu_device.clone());
    let b2_gpu = layer2.bias.to(gpu_device.clone());

    let gpu_layer1 = Linear {
        weight: w1_gpu.clone(),
        bias: b1_gpu.clone(),
    };
    let gpu_layer2 = Linear {
        weight: w2_gpu.clone(),
        bias: b2_gpu.clone(),
    };

    // Move input to GPU
    let x_data: Vec<f32> = (0..batch_size * in_features)
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();
    let x_gpu = Tensor::from_slice(DType::F32, Shape::new([batch_size, in_features]), &x_data)
        .to(gpu_device.clone());

    println!("Starting 10 training iterations...");
    let start = Instant::now();

    for i in 0..10 {
        // 1. Forward Pass (Executes via Fused WGSL Compute Shader)
        let h1 = gpu_layer1.forward(&x_gpu);
        let h2 = gpu_layer2.forward(&h1);
        let loss = h2.sum();

        // 2. Backward Pass (Calculates Gradients)
        let grads = loss.backward();

        // 3. SGD Updates (Executes via GPU Compute Shader!)
        optim::sgd_step(&w1_gpu, grads.get(&w1_gpu.id).unwrap(), lr);
        optim::sgd_step(&b1_gpu, grads.get(&b1_gpu.id).unwrap(), lr);
        optim::sgd_step(&w2_gpu, grads.get(&w2_gpu.id).unwrap(), lr);
        optim::sgd_step(&b2_gpu, grads.get(&b2_gpu.id).unwrap(), lr);

        if i == 0 {
            println!(
                "Iteration 0 complete. W1 is still on device: {:?}",
                if w1_gpu.device.is_gpu() { "GPU" } else { "CPU" }
            );
        }
    }

    println!("10 iterations took: {:?}", start.elapsed());
    println!("\n Training complete! Weights never left the GPU during updates.");
}

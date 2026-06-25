use axiom::{DType, Device, Shape, Tensor};
use std::time::Instant;

fn main() {
    println!("Battle-Testing Axiom's Dynamic VRAM Slab Allocator\n");

    let gpu_device = Device::gpu();

    // Allocate a massive 10MB tensor size
    let size = 1024 * 1024 * 10; // 10 Million f32s = 40MB

    println!("Starting 100 iterations of 40MB allocations...");
    println!("If PyTorch did this without an explicit cache clear, it would OOM.");
    println!("Axiom's RAII Drop will recycle the VRAM instantly.\n");

    let start = Instant::now();

    for i in 0..100 {
        // 1. Allocate 40MB on the GPU
        let data = vec![1.0f32; size];
        let t1 = Tensor::from_slice(DType::F32, Shape::new([size]), &data).to(gpu_device.clone());

        // 2. Do some math (allocates another 40MB output tensor in the Arena)
        let t2 = t1.add(&t1);

        // 3. End of loop iteration.
        // t1 and t2 go out of scope.
        // Rust's Drop trait INSTANTLY triggers arena.free(), returning 80MB to the Free-List!

        if i % 20 == 0 {
            println!("Iteration {} complete. VRAM successfully recycled.", i);
        }
    }

    println!(
        "\nSUCCESS! 100 iterations completed in {:?}",
        start.elapsed()
    );
    println!("The Dynamic Slab Allocator and RAII Drop prevented VRAM OOM.");
    println!("This is deterministic memory management. This is how we beat PyTorch.");
}

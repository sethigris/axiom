use axiom::{DType, MultiHeadAttention, Shape, Tensor};
use std::time::Instant;

fn main() {
    println!(" Running Axiom Multi-Head Attention Test\n");

    // 1. Generate the missing dummy data!
    let batch = 2;
    let seq = 16;
    let hidden = 64;
    let total_elements = batch * seq * hidden;

    let data: Vec<f32> = (0..total_elements)
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();

    // 2. Create an input sequence: [Batch, Sequence, Hidden]
    let x = Tensor::from_slice(DType::F32, Shape::new([batch, seq, hidden]), &data)
        .requires_grad_(true);

    // 3. Initialize Multi-Head Attention
    let mha = MultiHeadAttention::new(hidden, 4); // 64 hidden dim, 4 heads

    // 4. Forward pass (Utilizes zero-cost transposes and fused 2D GEMMs)
    let start = Instant::now();
    let out = mha.forward(&x);
    println!(" Forward pass took: {:?}", start.elapsed());
    println!("Output shape: {:?}\n", out.shape);

    // 5. Backward pass (Autograd traverses the 4D graph)
    let loss = out.sum();

    let start = Instant::now();
    let grads = loss.backward();
    println!(" Backward pass took: {:?}", start.elapsed());

    // We use `if let` instead of `.unwrap()` so it doesn't crash if the
    // gradient is missing due to the raw `copy` kernel boundary!
    if let Some(x_grad) = grads.get(&x.id) {
        println!("SUCCESS! Input Gradient shape: {:?}", x_grad.shape);
    } else {
        println!(
            " Gradient for 'x' missing (This is expected until I apply the `Tensor::cat` fix)."
        );
    }
}

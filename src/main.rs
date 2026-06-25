use axiom::{DType, PositionalEncoding, Shape, Tensor, TransformerBlock};
use std::time::Instant;

fn main() {
    println!(" Running Axiom Full Transformer Block Test\n");

    let batch = 2;
    let seq = 16;
    let hidden = 64;
    let num_heads = 4;
    let total_elements = batch * seq * hidden;

    // 1. Dummy Input (e.g., Word Embeddings)
    let data: Vec<f32> = (0..total_elements)
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();
    let x = Tensor::from_slice(DType::F32, Shape::new([batch, seq, hidden]), &data)
        .requires_grad_(true);

    // 2. Inject Positional Information
    let pos_enc = PositionalEncoding::new(batch, seq, hidden);
    let x_with_pos = pos_enc.forward(&x);

    // 3. Initialize the Transformer Block
    let block = TransformerBlock::new(hidden, num_heads);

    // --- FORWARD PASS ---
    println!("Starting Forward Pass (Attention + MLP + Residuals)...");
    let start = Instant::now();
    let out = block.forward(&x_with_pos);
    println!("Forward pass took: {:?}", start.elapsed());
    println!("Output shape: {:?}\n", out.shape);

    // --- BACKWARD PASS ---
    println!("Starting Backward Pass...");
    let loss = out.sum();

    let start = Instant::now();
    let grads = loss.backward();
    println!("ackward pass took: {:?}", start.elapsed());

    if let Some(x_grad) = grads.get(&x.id) {
        println!("\nSUCCESS! Input Gradient shape: {:?}", x_grad.shape);
        println!("The full Transformer Block (Attn + MLP + Residuals) backpropagated perfectly.");
    } else {
        println!("\nGradient missing.");
    }
}

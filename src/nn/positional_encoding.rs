use crate::{DType, Shape, Tensor};

pub struct PositionalEncoding {
    pub pe: Tensor, // Pre-computed [Batch, Seq, Hidden] encoding
}

impl PositionalEncoding {
    pub fn new(batch: usize, max_seq_len: usize, hidden_dim: usize) -> Self {
        let total_tokens = batch * max_seq_len;
        let mut pe_data = vec![0.0f32; total_tokens * hidden_dim];

        // Generate the sine/cosine waves
        for b in 0..batch {
            for pos in 0..max_seq_len {
                for i in (0..hidden_dim).step_by(2) {
                    let angle = pos as f32 / 10000.0f32.powf((i as f32) / (hidden_dim as f32));
                    let idx = (b * max_seq_len + pos) * hidden_dim + i;
                    pe_data[idx] = angle.sin();
                    if i + 1 < hidden_dim {
                        pe_data[idx + 1] = angle.cos();
                    }
                }
            }
        }

        let pe = Tensor::from_slice(
            DType::F32,
            Shape::new([batch, max_seq_len, hidden_dim]),
            &pe_data,
        );
        Self { pe }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Just add the positional encoding to the input embeddings!
        x.add(&self.pe)
    }
}

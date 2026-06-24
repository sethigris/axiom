use crate::Tensor;
use crate::data::dataset::MmapDataset;

/// An iterator that yields zero-copy batches from the memory-mapped dataset.
pub struct DataLoader<'a> {
    dataset: &'a MmapDataset,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a MmapDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            current_idx: 0,
        }
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.num_samples {
            return None;
        }

        let remaining = self.dataset.num_samples - self.current_idx;
        let current_batch_size = self.batch_size.min(remaining);

        let batch = self.dataset.get_batch(self.current_idx, current_batch_size);
        self.current_idx += current_batch_size;

        Some(batch)
    }
}

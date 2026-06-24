use crate::{Tensor, TensorId};
use std::collections::HashSet;
use std::sync::Arc;

/// The core trait for all neural network operations.
/// Added `std::fmt::Debug` so the Node can derive Debug.
pub trait Op: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;

    /// Given the gradient of the output, calculate the gradients of the inputs.
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
}

/// A node in the computation graph.
#[derive(Debug)]
pub struct Node {
    pub op: Arc<dyn Op>,
    pub parents: Vec<Tensor>,
}

// ... (rest of the file remains exactly the same) ...

impl Tensor {
    /// Attaches a computation node to this tensor, marking it as part of the Autograd graph.
    pub fn with_node(mut self, op: Arc<dyn Op>, parents: Vec<Tensor>) -> Self {
        self.node = Some(Arc::new(Node { op, parents }));
        self.requires_grad = true; // If it's an output of an Op, it requires grad
        self
    }

    /// Performs a Depth-First Search (DFS) to build a reverse topological sort of the graph.
    /// This is the exact order required for backpropagation (loss -> leaves).
    pub fn topo_sort(&self) -> Vec<Tensor> {
        let mut visited = HashSet::new();
        let mut sorted = Vec::new();

        fn dfs(tensor: &Tensor, visited: &mut HashSet<TensorId>, sorted: &mut Vec<Tensor>) {
            if visited.contains(&tensor.id) {
                return;
            }
            visited.insert(tensor.id);

            // If this tensor was created by an operation, visit its parents first
            if let Some(node) = &tensor.node {
                for parent in &node.parents {
                    dfs(parent, visited, sorted);
                }
            }

            // Post-order traversal: parents are added to the list before the child
            sorted.push(tensor.clone());
        }

        dfs(self, &mut visited, &mut sorted);

        // Post-order gives us Forward order (Parents -> Children).
        // We reverse it to get Backward order (Children -> Parents / Loss -> Leaves).
        sorted.reverse();
        sorted
    }
}

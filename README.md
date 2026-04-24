# Self-Pruning Neural Network

This project implements a neural network that learns to prune itself during training using learnable gates.

## Key Idea
Each weight is associated with a gate (0 to 1). Gates are learned using L1 regularization.

## Loss Function
Total Loss = CrossEntropy + λ × Sparsity Loss

## Results (Preliminary)
Accuracy: ~40%
Sparsity: Low (tuned further in progress)

## Note
Further tuning of λ improves sparsity vs accuracy trade-off.

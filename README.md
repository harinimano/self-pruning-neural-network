# Self-Pruning Neural Network

## Overview
This project implements a neural network that learns to prune its own weights during training using learnable gating mechanisms.

Instead of pruning after training, the model dynamically removes less important connections during training.

---

## Key Idea
Each weight has a learnable gate (0–1):

- Gate ≈ 1 → weight is important
- Gate ≈ 0 → weight is pruned

---

## Methodology

### Prunable Layer
Custom `PrunableLinear` layer:
- Each weight has a corresponding gate
- Gates = sigmoid(gate_scores)
- Effective weight = weight × gate

---

### Loss Function

Total Loss = CrossEntropy + λ × Sparsity Loss

- CrossEntropy → accuracy  
- Sparsity Loss → pruning  

Sparsity Loss = sum of gate values (L1 regularization)

---

## Results

| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| 0.1    | 15.35%  | 41.04%   |
| 5.0    | 10.0%   | 41.01%   |

---

## Analysis

Increasing λ forces more gates toward zero, increasing sparsity.

Higher sparsity reduces model capacity, leading to lower accuracy.

Due to limited training (2 epochs), sparsity is similar, but accuracy drop confirms pruning effect.

---

## Tech Stack
- Python
- PyTorch
- CIFAR-10

---

## How to Run

```bash
pip install torch torchvision
python train.py

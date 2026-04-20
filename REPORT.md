````markdown
# Self-Pruning Neural Network Report

## Overview

This project implements a self-pruning feed-forward neural network for CIFAR-10 image classification.

Unlike traditional pruning methods that remove weights after training, this model learns which connections are important during training itself. Each weight is associated with a learnable gate value between 0 and 1. If a gate becomes close to 0, the corresponding weight is effectively removed from the network.

The final effective weight matrix is:

```python
gates = torch.sigmoid(gate_scores)
pruned_weights = weight * gates
````

This allows the network to automatically decide which weights are useful and which ones can be pruned.

---

## Why L1 Penalty Encourages Sparsity

The training objective combines classification loss with a sparsity penalty:

```python
total_loss = classification_loss + lambda * sparsity_loss
```

Where:

* `classification_loss` is Cross Entropy Loss
* `sparsity_loss` is the sum of all gate values
* `lambda` controls the pruning strength

The sparsity loss is computed as:

```python
losses = [layer.gates.sum() for layer in prunable_layers(model)]
sparsity_loss = torch.stack(losses).sum()
```

The L1 penalty encourages gate values to move toward 0 because reducing gate values lowers the total loss. However, gates that are important for prediction remain active because the classification loss prevents them from collapsing completely.

As a result, the network keeps only the most useful connections while removing unnecessary ones.

---

## Model Architecture

The network is a Multi-Layer Perceptron using custom `PrunableLinear` layers.

```text
Input (3072)
   ↓
PrunableLinear(3072 → 512)
   ↓
ReLU
   ↓
Dropout
   ↓
PrunableLinear(512 → 256)
   ↓
ReLU
   ↓
Dropout
   ↓
PrunableLinear(256 → 10)
```

Since CIFAR-10 images are 32 × 32 × 3, the flattened input dimension is:

```text
32 × 32 × 3 = 3072
```

---

## Experimental Setup

* Dataset: CIFAR-10
* Epochs: 20
* Batch Size: 128
* Optimizer: Adam
* Learning Rate: 0.001
* Gate Threshold: 0.05
* Lambda Values Tested:

  * 0
  * 0.001
  * 0.005
  * 0.01

---

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| ------ | ----------------- | ------------------ |
| 0      | 49.99             | 0.00               |
| 0.001  | 49.57             | 99.94              |
| 0.005  | 45.00             | 99.99              |
| 0.01   | 40.00             | 100.00             |

---

## Analysis of Lambda Tradeoff

When lambda was set to 0, the model behaved like a normal neural network without any pruning pressure. This resulted in the highest accuracy but no sparsity.

As lambda increased, the model experienced stronger pressure to reduce gate values. This caused more weights to become inactive, increasing the sparsity level.

A lambda value of 0.001 produced an excellent tradeoff. The network achieved almost the same accuracy as the baseline model while pruning nearly all unnecessary weights.

Higher lambda values such as 0.005 and 0.01 caused over-pruning. Too many useful connections were removed, leading to lower test accuracy.

This demonstrates the expected tradeoff between sparsity and predictive performance.

---

## Gate Distribution

The histogram of gate values for the best model showed:

* A very large spike near 0, representing pruned connections
* A smaller cluster away from 0, representing important active connections

This confirms that the network successfully learned to prune itself during training.

![Gate Distribution](gate_distribution_best_model.png)

---

## Conclusion

The self-pruning neural network successfully learned to remove unnecessary weights during training.

The custom `PrunableLinear` layer correctly implemented learnable gates, the sparsity loss encouraged pruning, and the results demonstrated a clear tradeoff between model accuracy and sparsity.

Among all tested values, `lambda = 0.001` provided the best balance between predictive performance and pruning efficiency.

```
```

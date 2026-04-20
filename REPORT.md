# Self-Pruning Neural Network Report

## Approach

The model replaces every dense layer with a custom `PrunableLinear` layer. Each layer contains a normal weight matrix, a bias, and a same-shaped learnable `gate_scores` tensor. The forward pass applies `sigmoid(gate_scores)` to obtain gates in `[0, 1]`, multiplies the original weights by those gates, and then performs the standard linear operation.

This keeps the pruning mechanism differentiable: gradients flow into both the original weights and the gate scores.

## Why L1 on Sigmoid Gates Encourages Sparsity

The total loss is:

```text
total_loss = classification_loss + lambda * sparsity_loss
sparsity_loss = sum(sigmoid(gate_scores))
```

Because the gates are non-negative, their L1 norm is just their sum. This adds direct pressure for every gate to become smaller. Gates that are not important for classification can move close to zero, while gates that protect useful predictive weights remain open. Increasing `lambda` usually increases sparsity but can reduce accuracy if useful connections are pruned too aggressively.

## Results

Run the full CIFAR-10 experiment with:

```bash
python train_self_pruning.py --epochs 10 --lambdas 0 1e-5 1e-4
```

The script writes the completed table to `outputs/REPORT.md`.

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| ---: | ---: | ---: |
| 0 | Run pending | Run pending |
| 1e-5 | Run pending | Run pending |
| 1e-4 | Run pending | Run pending |

## Gate Distribution

The script saves the final plot as:

```text
outputs/gate_distribution_best_model.png
```

The expected successful pattern is a large concentration of gate values near zero and another cluster of higher values for connections that remain useful.

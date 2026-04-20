# Self-Pruning Neural Network

Case-study implementation for the Tredence AI Engineering Internship assignment: "The Self-Pruning Neural Network".

This project implements a feed-forward neural network for CIFAR-10 classification where every linear weight is paired with a learnable sigmoid gate. During training, an L1-based sparsity penalty encourages unnecessary connections to close automatically, allowing the network to prune itself dynamically.

---

## Project Objective

The goal is to train a neural network that learns which of its own weights are unnecessary.

Instead of pruning weights after training, the model contains learnable gate parameters:

```python
gates = torch.sigmoid(gate_scores)
pruned_weights = weight * gates
```

If a gate becomes close to 0, the corresponding connection is effectively removed.

The training objective combines:

```python
total_loss = classification_loss + lambda * sparsity_loss
```

Where:

* `classification_loss` is Cross Entropy Loss
* `sparsity_loss` is the sum of all sigmoid gate values
* `lambda` controls the sparsity-accuracy tradeoff

---

## Files

* `train_self_pruning.py` - Main training and evaluation script
* `requirements.txt` - Python dependencies
* `README.md` - Project documentation
* `outputs/results.csv` - Accuracy and sparsity metrics for all lambda values
* `outputs/REPORT.md` - Automatically generated report
* `outputs/gate_distribution_best_model.png` - Histogram of gate values for the best model
* `outputs/best_lambda_*.pt` - Saved checkpoints for the best model of each lambda value

---

## Requirements

* Python 3.10+
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pandas

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run the Full Experiment

```bash
python train_self_pruning.py --epochs 20 --lambdas 0 1e-3 5e-3 1e-2 --gate-threshold 0.05
```

This command will:

1. Download CIFAR-10 automatically
2. Train one model for each lambda value
3. Evaluate test accuracy
4. Measure sparsity level
5. Save model checkpoints
6. Generate a CSV summary
7. Generate a markdown report
8. Generate a histogram of gate values

---

## Example Output

```text
lambda=0 epoch=020 loss=1.3787 test_acc=49.52% sparsity=0.00%
lambda=0.001 epoch=020 loss=18.5101 test_acc=48.72% sparsity=99.94%
```

This shows that:

* `lambda = 0` preserves all weights and gives the highest accuracy
* `lambda = 0.001` heavily prunes the network while maintaining similar accuracy
* Higher lambda values produce stronger pruning but may reduce performance

---

## Run a Quick Smoke Test

```bash
python train_self_pruning.py --smoke-test --epochs 1 --lambdas 0 1e-3 --num-workers 0
```

This uses `torchvision.datasets.FakeData` instead of CIFAR-10 to quickly validate the training pipeline.

---

## Important Hyperparameters

### Lambda Values

Lambda controls the amount of pruning pressure.

Recommended values:

```bash
--lambdas 0 1e-3 5e-3 1e-2
```

Typical behavior:

| Lambda | Expected Accuracy | Expected Sparsity |
| ------ | ----------------- | ----------------- |
| 0      | High              | Very Low          |
| 0.001  | High              | Very High         |
| 0.005  | Medium            | Extremely High    |
| 0.01   | Lower             | Near Total        |

### Gate Threshold

A gate is considered pruned if:

```python
gate_value < gate_threshold
```

Recommended threshold:

```bash
--gate-threshold 0.05
```

---

## Model Architecture

The network is a simple Multi-Layer Perceptron:

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

Since CIFAR-10 images are `32 x 32 x 3`, the flattened input size is:

```text
32 × 32 × 3 = 3072
```

---

## Sparsity Loss

The sparsity loss is computed across all prunable layers:

```python
losses = [layer.gates.sum() for layer in prunable_layers(model)]
sparsity_loss = torch.stack(losses).sum()
```

This encourages gate values to move toward zero.

---

## Outputs Generated

After training, the script generates:

* `outputs/results.csv`
* `outputs/REPORT.md`
* `outputs/gate_distribution_best_model.png`
* `outputs/best_lambda_0.pt`
* `outputs/best_lambda_0.001.pt`
* `outputs/best_lambda_0.005.pt`
* `outputs/best_lambda_0.01.pt`

---

## Interpretation of Results

A successful run should show:

* High accuracy and low sparsity for small lambda values
* Lower accuracy and higher sparsity for large lambda values
* A gate distribution histogram with:

  * A large spike near 0
  * A smaller cluster away from 0

This demonstrates that the model learned to remove unnecessary connections automatically during training.

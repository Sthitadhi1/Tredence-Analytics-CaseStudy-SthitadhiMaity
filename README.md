# Self-Pruning Neural Network

Case-study implementation for Tredence's "Self-Pruning Neural Network" assignment.

The project implements a feed-forward CIFAR-10 classifier whose linear weights are multiplied by learnable sigmoid gates. An L1 penalty over those gates encourages unnecessary connections to close during training.

## Files

- `train_self_pruning.py` - single self-contained assignment script.
- `REPORT.md` - short case-study explanation and result table.
- `requirements.txt` - Python dependencies.
- `outputs/REPORT.md` - generated report after training.
- `outputs/gate_distribution_best_model.png` - generated gate histogram after training.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Full Case Study

```bash
python train_self_pruning.py --epochs 10 --lambdas 0 1e-5 1e-4
```

The script downloads CIFAR-10 through `torchvision.datasets.CIFAR10`, trains one model per lambda, saves checkpoints, writes `outputs/results.csv`, writes `outputs/REPORT.md`, and saves the best model's gate-distribution plot.

## Quick Smoke Test

```bash
python train_self_pruning.py --smoke-test --epochs 1 --lambdas 0 1e-4 --num-workers 0
```

The smoke test uses `torchvision.datasets.FakeData` to verify the training loop without downloading CIFAR-10.

## Notes

Use the generated `outputs/REPORT.md` as the final submission report after running the full CIFAR-10 experiment.

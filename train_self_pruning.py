"""Self-pruning neural network case study for CIFAR-10.

This single script implements:
- PrunableLinear with learnable gate scores per weight.
- L1 sparsity regularization over sigmoid gates.
- Training/evaluation for multiple lambda values.
- Result reporting and gate-distribution plotting.

Example:
    python train_self_pruning.py --epochs 10 --lambdas 0 1e-5 1e-4

For a fast CPU smoke test:
    python train_self_pruning.py --smoke-test --epochs 1 --lambdas 0 1e-4
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class PrunableLinear(nn.Module):
    """A linear layer whose weights are multiplied by learnable sigmoid gates."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Positive initialization keeps most gates initially active
        # while still allowing the optimizer to close them.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.gate_scores, 2.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pruned_weights = self.weight * self.gates
        return F.linear(inputs, pruned_weights, self.bias)


class SelfPruningMLP(nn.Module):
    """A compact feed-forward classifier using PrunableLinear layers."""

    def __init__(self, hidden_sizes: tuple[int, int] = (512, 256), num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            PrunableLinear(3 * 32 * 32, hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            PrunableLinear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            PrunableLinear(hidden_sizes[1], num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass
class RunResult:
    lambda_value: float
    test_accuracy: float
    sparsity_level: float
    best_epoch: int
    final_train_loss: float
    checkpoint_path: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prunable_layers(model: nn.Module) -> Iterable[PrunableLinear]:
    return (module for module in model.modules() if isinstance(module, PrunableLinear))


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    losses = [layer.gates.sum() for layer in prunable_layers(model)]
    if not losses:
        raise ValueError("Model contains no PrunableLinear layers.")
    return torch.stack(losses).sum()


@torch.no_grad()
def gate_values(model: nn.Module) -> torch.Tensor:
    return torch.cat([layer.gates.detach().flatten().cpu() for layer in prunable_layers(model)])


@torch.no_grad()
def sparsity_level(model: nn.Module, threshold: float) -> float:
    gates = gate_values(model)
    return (gates < threshold).float().mean().item() * 100.0


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    if args.smoke_test:
        train_data = datasets.FakeData(
            size=min(args.train_subset or 512, 512),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
        )
        test_data = datasets.FakeData(
            size=min(args.test_subset or 256, 256),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=test_transform,
        )
    else:
        train_data = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)

        if args.train_subset:
            train_data = Subset(train_data, range(min(args.train_subset, len(train_data))))
        if args.test_subset:
            test_data = Subset(test_data, range(min(args.test_subset, len(test_data))))

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_value: float,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        classification_loss = F.cross_entropy(logits, targets)
        total_loss = classification_loss + lambda_value * sparsity_loss(model)
        total_loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += total_loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(inputs).argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return correct / max(total, 1) * 100.0


def run_experiment(args: argparse.Namespace, lambda_value: float, train_loader: DataLoader, test_loader: DataLoader) -> RunResult:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SelfPruningMLP(hidden_sizes=(args.hidden1, args.hidden2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_accuracy = -1.0
    best_epoch = 0
    final_train_loss = 0.0
    checkpoint_path = Path(args.output_dir) / f"best_lambda_{lambda_value:g}.pt"

    for epoch in range(1, args.epochs + 1):
        final_train_loss = train_one_epoch(model, train_loader, optimizer, device, lambda_value)
        accuracy = evaluate(model, test_loader, device)
        current_sparsity = sparsity_level(model, args.gate_threshold)
        print(
            f"lambda={lambda_value:g} epoch={epoch:03d} "
            f"loss={final_train_loss:.4f} test_acc={accuracy:.2f}% "
            f"sparsity={current_sparsity:.2f}%"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    final_accuracy = evaluate(model, test_loader, device)
    final_sparsity = sparsity_level(model, args.gate_threshold)

    return RunResult(
        lambda_value=lambda_value,
        test_accuracy=final_accuracy,
        sparsity_level=final_sparsity,
        best_epoch=best_epoch,
        final_train_loss=final_train_loss,
        checkpoint_path=str(checkpoint_path),
    )


def plot_best_gates(args: argparse.Namespace, best_result: RunResult) -> str:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SelfPruningMLP(hidden_sizes=(args.hidden1, args.hidden2)).to(device)
    model.load_state_dict(torch.load(best_result.checkpoint_path, map_location=device))
    gates = gate_values(model).numpy()

    plot_path = Path(args.output_dir) / "gate_distribution_best_model.png"
    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=60, color="#2563eb", edgecolor="white")
    plt.axvline(args.gate_threshold, color="#dc2626", linestyle="--", label=f"threshold={args.gate_threshold:g}")
    plt.title(f"Gate distribution for best model (lambda={best_result.lambda_value:g})")
    plt.xlabel("Sigmoid gate value")
    plt.ylabel("Number of weights")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return str(plot_path)


def write_results(args: argparse.Namespace, results: list[RunResult], plot_path: str) -> None:
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "results.csv"
    report_path = output_dir / "REPORT.md"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "lambda",
                "test_accuracy",
                "sparsity_level",
                "best_epoch",
                "final_train_loss",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "lambda": result.lambda_value,
                    "test_accuracy": f"{result.test_accuracy:.4f}",
                    "sparsity_level": f"{result.sparsity_level:.4f}",
                    "best_epoch": result.best_epoch,
                    "final_train_loss": f"{result.final_train_loss:.6f}",
                    "checkpoint_path": result.checkpoint_path,
                }
            )

    rows = "\n".join(
        f"| {result.lambda_value:g} | {result.test_accuracy:.2f} | {result.sparsity_level:.2f} |"
        for result in results
    )
    best = max(results, key=lambda result: result.test_accuracy)

    report_path.write_text(
        f"""# Self-Pruning Neural Network Report

## Approach

Each dense layer is replaced with `PrunableLinear`, which owns a normal weight matrix plus a same-shaped `gate_scores` parameter. During the forward pass, `sigmoid(gate_scores)` converts those scores to values in `[0, 1]`; the effective matrix is `weight * gate`, so a gate near zero removes that individual connection while keeping the operation differentiable.

## Why L1 on Gates Encourages Sparsity

The training objective is:

```text
total_loss = cross_entropy_loss + lambda * sum(sigmoid(gate_scores))
```

The L1 penalty adds a constant pressure to reduce every active gate. Because the classification loss only protects weights that help prediction, unimportant connections can be pushed toward zero while useful connections remain open. Increasing `lambda` strengthens this pressure, usually raising sparsity at the cost of accuracy.

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| ---: | ---: | ---: |
{rows}

Best checkpoint by accuracy: `{best.checkpoint_path}`

Gate threshold for sparsity: `{args.gate_threshold:g}`

## Gate Distribution

![Gate distribution for best model]({Path(plot_path).name})

## Interpretation

Lower lambda values tend to preserve more weights and favor accuracy. Higher lambda values increase the number of near-zero gates, demonstrating self-pruning behavior, but can remove useful connections if the regularization pressure is too strong.
""",
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-pruning neural network on CIFAR-10.")
    parser.add_argument("--data-dir", default="data", help="Directory for CIFAR-10 download/cache.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for checkpoints and reports.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1e-5, 1e-4])
    parser.add_argument("--gate-threshold", type=float, default=1e-2)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--train-subset", type=int, default=None, help="Optional smaller training subset.")
    parser.add_argument("--test-subset", type=int, default=None, help="Optional smaller test subset.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="Force device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Use FakeData to validate the pipeline quickly.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_loaders(args)
    results = [
        run_experiment(args, lambda_value, train_loader, test_loader)
        for lambda_value in args.lambdas
    ]
    best_result = max(results, key=lambda result: result.test_accuracy)
    plot_path = plot_best_gates(args, best_result)
    write_results(args, results, plot_path)


if __name__ == "__main__":
    main()

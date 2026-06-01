"""Training loop for CheXpert multi-label classifier.

Features:
- AdamW optimizer with cosine annealing LR schedule
- Class-weighted BCEWithLogitsLoss for imbalanced labels
- Early stopping on validation macro AUC
- MLflow experiment tracking (local mlruns/)
- Checkpoint saving (best val AUC model)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.constants import (
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    EARLY_STOP_PATIENCE,
)
from src.data import build_mock_dataloaders, compute_class_weights
from src.model import CheXpertClassifier, build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def build_loss(class_weights: Optional[torch.Tensor] = None) -> nn.BCEWithLogitsLoss:
    """Build BCEWithLogitsLoss, optionally with per-class pos_weight.

    Args:
        class_weights: Tensor of shape (num_classes,). Typically neg/pos ratio.

    Returns:
        Configured loss function.
    """
    return nn.BCEWithLogitsLoss(pos_weight=class_weights)


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one epoch (train or eval).

    Returns:
        Tuple of (mean_loss, all_targets, all_probs) where targets and probs
        are numpy arrays of shape (N, num_classes).
    """
    model.train(is_train)
    total_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs)

    n = sum(t.shape[0] for t in all_targets)
    mean_loss = total_loss / max(n, 1)
    targets_np = np.concatenate(all_targets, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)
    return mean_loss, targets_np, probs_np


def _compute_macro_auc(targets: np.ndarray, probs: np.ndarray) -> float:
    """Compute macro-averaged AUC ignoring labels with only one class present."""
    aucs = []
    for i in range(targets.shape[1]):
        col = targets[:, i]
        if len(np.unique(col)) < 2:
            continue
        aucs.append(roc_auc_score(col, probs[:, i]))
    return float(np.mean(aucs)) if aucs else 0.0


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation metric stops improving.

    Args:
        patience: Epochs to wait before stopping.
        min_delta: Minimum improvement to reset counter.
        mode: 'max' for AUC-like metrics, 'min' for loss.
    """

    def __init__(self, patience: int = EARLY_STOP_PATIENCE, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best: Optional[float] = None
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Update state with latest metric.

        Returns:
            True if training should stop.
        """
        if self.best is None:
            self.best = metric
            return False

        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model: CheXpertClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    class_weights: Optional[torch.Tensor] = None,
    checkpoint_dir: str | Path = "checkpoints",
    experiment_name: str = "chexpert-classifier",
    run_name: Optional[str] = None,
) -> dict[str, list[float]]:
    """Full training loop with MLflow tracking.

    Args:
        model: CheXpertClassifier instance on the correct device.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Torch device.
        epochs: Max training epochs.
        lr: Initial learning rate.
        weight_decay: AdamW weight decay.
        class_weights: Optional per-class pos_weight for BCEWithLogitsLoss.
        checkpoint_dir: Directory to save best model checkpoint.
        experiment_name: MLflow experiment name.
        run_name: Optional MLflow run name.

    Returns:
        History dict with keys 'train_loss', 'val_loss', 'val_auc'.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = build_loss(class_weights)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="max")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
    }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": train_loader.batch_size,
            "num_classes": model.num_classes,
        })

        best_auc = 0.0
        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, _, _ = _run_epoch(
                model, train_loader, criterion, optimizer, device, is_train=True
            )
            val_loss, val_targets, val_probs = _run_epoch(
                model, val_loader, criterion, None, device, is_train=False
            )
            val_auc = _compute_macro_auc(val_targets, val_probs)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc},
                step=epoch,
            )

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_auc=%.4f (%.1fs)",
                epoch, epochs, train_loss, val_loss, val_auc, elapsed,
            )

            if val_auc > best_auc:
                best_auc = val_auc
                ckpt_path = checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_auc": val_auc,
                    },
                    ckpt_path,
                )
                mlflow.log_artifact(str(ckpt_path))
                logger.info("  -> Saved best checkpoint (AUC=%.4f)", best_auc)

            if early_stopper.step(val_auc):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        mlflow.log_metric("best_val_auc", best_auc)

    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Train on mock data by default (no CheXpert download needed)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Building mock dataloaders...")
    loaders = build_mock_dataloaders(
        train_samples=400,
        val_samples=100,
        batch_size=32,
    )
    train_ds = loaders["train"].dataset
    class_weights = compute_class_weights(train_ds)

    model, device = build_model(pretrained=False)  # pretrained=False for quick demo
    logger.info(
        "Model: %d trainable params / %d total",
        model.count_trainable_params(),
        model.count_total_params(),
    )

    history = train(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        epochs=3,  # short run for demo
        class_weights=class_weights,
    )
    logger.info("Training complete. Best val_auc=%.4f", max(history["val_auc"]))


if __name__ == "__main__":
    main()

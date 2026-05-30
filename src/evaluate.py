"""Evaluation module for CheXpert pathology classifier.

Computes:
- Per-label AUC-ROC
- Macro and micro AUC
- Comparison table vs Irvin et al. 2019 published results
- Calibration curve
- Saves results to results/metrics.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.constants import CHEXPERT_LABELS, PUBLISHED_AUCS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference over entire DataLoader.

    Args:
        model: Trained model (on correct device).
        loader: DataLoader yielding (images, labels).
        device: Torch device.

    Returns:
        Tuple of (targets, probs), each shape (N, num_classes).
    """
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.append(targets.numpy())
            all_probs.append(probs)

    return (
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_probs, axis=0),
    )


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------

def compute_per_label_auc(
    targets: np.ndarray,
    probs: np.ndarray,
    label_names: list[str] = CHEXPERT_LABELS,
) -> dict[str, float]:
    """Compute AUC-ROC for each label.

    Labels with fewer than 2 unique target values are skipped and set to NaN.

    Args:
        targets: Binary label array of shape (N, num_classes).
        probs: Predicted probabilities of shape (N, num_classes).
        label_names: List of label names (length == num_classes).

    Returns:
        Dictionary mapping label name -> AUC float (or NaN if undetermined).
    """
    aucs: dict[str, float] = {}
    for i, label in enumerate(label_names):
        col = targets[:, i]
        if len(np.unique(col)) < 2:
            aucs[label] = float("nan")
            logger.warning("Label '%s' has only one class — AUC set to NaN.", label)
        else:
            aucs[label] = float(roc_auc_score(col, probs[:, i]))
    return aucs


def compute_macro_auc(per_label_aucs: dict[str, float]) -> float:
    """Mean AUC across all valid (non-NaN) labels."""
    valid = [v for v in per_label_aucs.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_micro_auc(targets: np.ndarray, probs: np.ndarray) -> float:
    """Micro-averaged AUC (flatten all label predictions)."""
    try:
        return float(roc_auc_score(targets.ravel(), probs.ravel()))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Published AUC comparison
# ---------------------------------------------------------------------------

def build_comparison_table(
    per_label_aucs: dict[str, float],
    published: dict[str, float] = PUBLISHED_AUCS,
) -> list[dict]:
    """Build comparison rows between our AUCs and Irvin et al. 2019.

    Args:
        per_label_aucs: Computed AUCs from this model.
        published: Published reference AUCs.

    Returns:
        List of row dicts with keys: label, our_auc, published_auc, delta.
    """
    rows = []
    for label, pub_auc in published.items():
        our_auc = per_label_aucs.get(label, float("nan"))
        delta = our_auc - pub_auc if not np.isnan(our_auc) else float("nan")
        rows.append({
            "label": label,
            "our_auc": round(our_auc, 4),
            "published_auc": round(pub_auc, 4),
            "delta": round(delta, 4),
        })
    return rows


def print_comparison_table(rows: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Label':<35} {'Our AUC':>8} {'Published':>10} {'Delta':>8}"
    sep = "-" * len(header)
    print("\nAUC Comparison vs Irvin et al. 2019 (CheXpert)")
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        delta_str = f"{row['delta']:+.4f}" if not np.isnan(row["delta"]) else "   N/A"
        our_str = f"{row['our_auc']:.4f}" if not np.isnan(row["our_auc"]) else "   N/A"
        print(
            f"{row['label']:<35} {our_str:>8} {row['published_auc']:>10.4f} {delta_str:>8}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def plot_calibration_curves(
    targets: np.ndarray,
    probs: np.ndarray,
    label_names: list[str] = CHEXPERT_LABELS,
    save_path: str | Path = "results/calibration.png",
    n_bins: int = 10,
) -> None:
    """Plot calibration curves for each label.

    Args:
        targets: Binary label array (N, num_classes).
        probs: Probability predictions (N, num_classes).
        label_names: Label names for plot legend.
        save_path: Output file path.
        n_bins: Number of calibration bins.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()

    for i, label in enumerate(label_names):
        if i >= len(axes):
            break
        ax = axes[i]
        col = targets[:, i]
        if len(np.unique(col)) < 2:
            ax.text(0.5, 0.5, "Single class", ha="center", va="center")
            ax.set_title(label, fontsize=9)
            continue
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                col, probs[:, i], n_bins=n_bins, strategy="uniform"
            )
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=7)
        except Exception as exc:
            logger.warning("Calibration plot failed for %s: %s", label, exc)

    # Hide unused axes
    for j in range(len(label_names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Calibration Curves — CheXpert Pathology Classifier", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration curves saved to %s", save_path)


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------

def plot_roc_curves(
    targets: np.ndarray,
    probs: np.ndarray,
    label_names: list[str] = CHEXPERT_LABELS,
    save_path: str | Path = "results/roc_curves.png",
) -> None:
    """Plot ROC curves for each label on a single figure."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab20", len(label_names))

    for i, label in enumerate(label_names):
        col = targets[:, i]
        if len(np.unique(col)) < 2:
            continue
        fpr, tpr, _ = roc_curve(col, probs[:, i])
        auc = roc_auc_score(col, probs[:, i])
        ax.plot(fpr, tpr, color=colors(i), lw=1.5, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — CheXpert 14-Label Classification")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved to %s", save_path)


# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------

def save_metrics(
    per_label_aucs: dict[str, float],
    macro_auc: float,
    micro_auc: float,
    comparison_rows: list[dict],
    save_path: str | Path = "results/metrics.json",
) -> None:
    """Serialize evaluation metrics to JSON.

    Args:
        per_label_aucs: AUC per label.
        macro_auc: Macro-averaged AUC.
        micro_auc: Micro-averaged AUC.
        comparison_rows: Comparison table rows.
        save_path: Output JSON path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "per_label_auc": {
            k: (round(v, 6) if not np.isnan(v) else None)
            for k, v in per_label_aucs.items()
        },
        "macro_auc": round(macro_auc, 6) if not np.isnan(macro_auc) else None,
        "micro_auc": round(micro_auc, 6) if not np.isnan(micro_auc) else None,
        "comparison_vs_irvin_2019": comparison_rows,
    }
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", save_path)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names: list[str] = CHEXPERT_LABELS,
    results_dir: str | Path = "results",
    plot: bool = True,
) -> dict:
    """Run full evaluation pipeline.

    Args:
        model: Trained model.
        loader: DataLoader for eval split.
        device: Torch device.
        label_names: Label names.
        results_dir: Directory to save outputs.
        plot: Whether to generate and save plots.

    Returns:
        Metrics dictionary.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running inference...")
    targets, probs = run_inference(model, loader, device)

    logger.info("Computing AUCs...")
    per_label_aucs = compute_per_label_auc(targets, probs, label_names)
    macro_auc = compute_macro_auc(per_label_aucs)
    micro_auc = compute_micro_auc(targets, probs)

    comparison = build_comparison_table(per_label_aucs)
    print_comparison_table(comparison)

    logger.info("Macro AUC: %.4f | Micro AUC: %.4f", macro_auc, micro_auc)

    save_metrics(
        per_label_aucs, macro_auc, micro_auc, comparison,
        save_path=results_dir / "metrics.json",
    )

    if plot:
        plot_roc_curves(targets, probs, label_names, save_path=results_dir / "roc_curves.png")
        plot_calibration_curves(targets, probs, label_names, save_path=results_dir / "calibration.png")

    return {
        "per_label_auc": per_label_aucs,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "comparison": comparison,
    }

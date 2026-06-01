"""EfficientNet-B4 model for chest X-ray multi-label classification.

Provides a second backbone option alongside the V1 DenseNet121 classifier,
plus a utility for comparing two models on the same dataset using per-label AUC.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from src.constants import NUM_CLASSES, CHEXPERT_LABELS


def build_efficientnet_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build an EfficientNet-B4 model with a custom 14-label classification head.

    The original EfficientNet-B4 classifier (in_features=1792) is replaced with:
        Dropout(0.4) → Linear(1792, num_classes)

    This outputs raw logits. Apply sigmoid externally for probabilities.

    Args:
        num_classes: Number of output classes (default 14 for CheXpert).
        pretrained: Load ImageNet-pretrained weights when True (default True).

    Returns:
        EfficientNet-B4 nn.Module with the custom head.
    """
    weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
    model = efficientnet_b4(weights=weights)

    # Replace classifier: EfficientNet-B4 head is at model.classifier
    # Original: Sequential(Dropout, Linear(1792, 1000))
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(1792, num_classes),
    )
    return model


def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    dataloader,
    label_names: list[str] = CHEXPERT_LABELS,
) -> pd.DataFrame:
    """Compare two models by per-label AUC on a shared evaluation dataloader.

    Both models are evaluated in eval mode with no gradient tracking.
    Per-label AUC is computed using sklearn roc_auc_score. If a label has
    only one class present in the ground truth, AUC is set to NaN.

    If dataloader is None, a mock DataFrame with random AUCs is returned so
    the function can be called without a real dataset (useful for testing).

    Args:
        model_a: First model (e.g. DenseNet121 V1).
        model_b: Second model (e.g. EfficientNet-B4 V2).
        dataloader: DataLoader that yields (images, labels) batches, or None
                    to return mock data.
        label_names: List of label names for the output DataFrame.

    Returns:
        DataFrame with columns [label, model_a_auc, model_b_auc, delta].
        delta = model_b_auc - model_a_auc (positive means B is better).
    """
    if dataloader is None:
        rng = np.random.default_rng(seed=42)
        n = len(label_names)
        auc_a = rng.uniform(0.55, 0.95, size=n)
        auc_b = rng.uniform(0.55, 0.95, size=n)
        return pd.DataFrame({
            "label": label_names,
            "model_a_auc": auc_a.round(4).tolist(),
            "model_b_auc": auc_b.round(4).tolist(),
            "delta": (auc_b - auc_a).round(4).tolist(),
        })

    device_a = next(model_a.parameters()).device
    device_b = next(model_b.parameters()).device

    model_a.eval()
    model_b.eval()

    all_labels: list[np.ndarray] = []
    all_probs_a: list[np.ndarray] = []
    all_probs_b: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in dataloader:
            all_labels.append(labels.numpy())

            logits_a = model_a(images.to(device_a))
            all_probs_a.append(torch.sigmoid(logits_a).cpu().numpy())

            logits_b = model_b(images.to(device_b))
            all_probs_b.append(torch.sigmoid(logits_b).cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)      # (N, num_classes)
    y_prob_a = np.concatenate(all_probs_a, axis=0)   # (N, num_classes)
    y_prob_b = np.concatenate(all_probs_b, axis=0)   # (N, num_classes)

    num_labels = y_true.shape[1]
    auc_a_list: list[float] = []
    auc_b_list: list[float] = []

    for i in range(num_labels):
        unique_classes = np.unique(y_true[:, i])
        if len(unique_classes) < 2:
            auc_a_list.append(float("nan"))
            auc_b_list.append(float("nan"))
        else:
            auc_a_list.append(float(roc_auc_score(y_true[:, i], y_prob_a[:, i])))
            auc_b_list.append(float(roc_auc_score(y_true[:, i], y_prob_b[:, i])))

    delta = [
        round(b - a, 4) if not (np.isnan(a) or np.isnan(b)) else float("nan")
        for a, b in zip(auc_a_list, auc_b_list)
    ]

    return pd.DataFrame({
        "label": label_names[:num_labels],
        "model_a_auc": [round(v, 4) if not np.isnan(v) else float("nan") for v in auc_a_list],
        "model_b_auc": [round(v, 4) if not np.isnan(v) else float("nan") for v in auc_b_list],
        "delta": delta,
    })

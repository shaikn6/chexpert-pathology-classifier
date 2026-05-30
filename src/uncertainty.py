"""Monte Carlo Dropout uncertainty quantification for multi-label chest X-ray classification.

Performs N stochastic forward passes with dropout enabled at inference time to
estimate prediction uncertainty per label. Returns mean predictions, standard
deviations, and 95% confidence intervals.

Reference: Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class UncertaintyResult:
    """Per-label uncertainty statistics from Monte Carlo Dropout.

    Attributes:
        mean: Mean sigmoid probability per label, shape (num_classes,).
        std: Standard deviation per label, shape (num_classes,).
        ci_lower: Lower bound of 95% CI per label, shape (num_classes,).
        ci_upper: Upper bound of 95% CI per label, shape (num_classes,).
    """

    mean: np.ndarray      # (num_classes,)
    std: np.ndarray       # (num_classes,)
    ci_lower: np.ndarray  # (num_classes,)
    ci_upper: np.ndarray  # (num_classes,)


class MCDropoutPredictor:
    """Monte Carlo Dropout predictor for Bayesian uncertainty estimation.

    Keeps the backbone in eval mode (BatchNorm statistics fixed) while
    switching all Dropout layers to train mode so they remain stochastic
    during inference. This approximates a Bayesian neural network.

    Args:
        model: The trained classifier (any nn.Module with Dropout layers).
        n_samples: Number of stochastic forward passes (default 30).
    """

    def __init__(self, model: nn.Module, n_samples: int = 30) -> None:
        self.model = model
        self.n_samples = n_samples

    def enable_dropout(self) -> None:
        """Set all Dropout layers to train mode to enable stochastic passes.

        The rest of the model stays in eval mode so BatchNorm uses its
        running statistics rather than batch statistics.
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self, image_tensor: torch.Tensor
    ) -> UncertaintyResult:
        """Run N stochastic forward passes and compute per-label statistics.

        Args:
            image_tensor: Input tensor of shape (1, C, H, W) or (C, H, W).
                          Batch dim is added automatically if missing.

        Returns:
            UncertaintyResult with mean, std, ci_lower, ci_upper per label.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        # Switch to eval first (fixes BatchNorm), then re-enable dropout
        self.model.eval()
        self.enable_dropout()

        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(image_tensor)
                probs = torch.sigmoid(logits)
                predictions.append(probs.cpu().numpy()[0])  # (num_classes,)

        stacked = np.stack(predictions, axis=0)  # (n_samples, num_classes)

        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        ci_lower = np.percentile(stacked, 2.5, axis=0)
        ci_upper = np.percentile(stacked, 97.5, axis=0)

        return UncertaintyResult(
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    @staticmethod
    def classify_uncertainty(std: float) -> str:
        """Map a single-label standard deviation to a human-readable tier.

        Args:
            std: Standard deviation for one label from MC Dropout.

        Returns:
            "low" if std < 0.05, "medium" if std < 0.15, "high" otherwise.
        """
        if std < 0.05:
            return "low"
        if std < 0.15:
            return "medium"
        return "high"

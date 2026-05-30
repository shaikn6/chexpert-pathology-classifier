"""DenseNet121-based multi-label chest X-ray pathology classifier.

Architecture: DenseNet121 backbone pretrained on ImageNet, with the last two
dense blocks fine-tuned. A custom classification head with sigmoid activation
handles 14 CheXpert pathology labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

from src.constants import NUM_CLASSES, CHEXPERT_LABELS


class CheXpertClassifier(nn.Module):
    """Multi-label DenseNet121 classifier for CheXpert 14-label task.

    Fine-tunes the last two dense blocks (denseblock3 + denseblock4 +
    transition layers) while freezing earlier feature extraction layers.
    Uses sigmoid output for independent multi-label prediction.

    Args:
        num_classes: Number of pathology labels to predict (default 14).
        pretrained: Load ImageNet pretrained weights (default True).
        dropout_rate: Dropout probability in the classifier head (default 0.2).
        freeze_early: Freeze blocks 1 and 2 (default True).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_early: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.label_names = CHEXPERT_LABELS[:num_classes]

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Remove the original classifier head — keep only the feature extractor
        self.features = backbone.features
        feature_dim = backbone.classifier.in_features  # 1024

        # Freeze early dense blocks if requested
        if freeze_early:
            self._freeze_early_layers()

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def _freeze_early_layers(self) -> None:
        """Freeze denseblock1 + transition1 + denseblock2 + transition2."""
        layers_to_freeze = [
            "conv0",
            "norm0",
            "relu0",
            "pool0",
            "denseblock1",
            "transition1",
            "denseblock2",
            "transition2",
        ]
        for name, param in self.features.named_parameters():
            block = name.split(".")[0]
            if block in layers_to_freeze:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (call for second-stage fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (pre-sigmoid).

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.features(x)
        features = torch.nn.functional.relu(features, inplace=True)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for each label.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Probability tensor of shape (B, num_classes) in [0, 1].
        """
        with torch.no_grad():
            logits = self.forward(x)
        return torch.sigmoid(logits)

    def get_last_conv_layer(self) -> nn.Module:
        """Return the last convolutional layer for Grad-CAM targeting."""
        return self.features.denseblock4

    def count_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    freeze_early: bool = True,
    device: str | None = None,
) -> tuple[CheXpertClassifier, torch.device]:
    """Factory function to build and move model to the appropriate device.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet weights.
        dropout_rate: Dropout probability.
        freeze_early: Whether to freeze early dense blocks.
        device: Target device string. Auto-selects CUDA/MPS/CPU if None.

    Returns:
        Tuple of (model, device).
    """
    if device is None:
        if torch.cuda.is_available():
            resolved_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            resolved_device = torch.device("mps")
        else:
            resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)

    model = CheXpertClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_early=freeze_early,
    )
    model = model.to(resolved_device)
    return model, resolved_device

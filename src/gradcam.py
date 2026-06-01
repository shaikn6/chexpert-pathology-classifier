"""Grad-CAM implementation for DenseNet121 chest X-ray saliency visualization.

Generates class-discriminative localization maps by back-propagating gradients
through the last convolutional layer and weighting activation maps accordingly.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization." ICCV 2017.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.constants import CHEXPERT_LABELS, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook-based Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM for multi-label classification models.

    Registers forward and backward hooks on the target layer to capture
    activations and gradients needed for saliency map computation.

    Args:
        model: The classifier model.
        target_layer: The convolutional layer to target (last conv block).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove registered hooks from the model."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args) -> None:
        self.remove_hooks()

    def generate(
        self,
        image: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for one image and one class.

        Args:
            image: Input tensor of shape (1, C, H, W).
            class_idx: Target class index.

        Returns:
            Normalized heatmap as numpy array of shape (H, W) in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        image = image.requires_grad_(True)

        logits = self.model(image)
        score = logits[0, class_idx]
        score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("Gradients or activations were not captured.")

        # GAP over spatial dims -> weights (C,)
        weights = self._gradients.mean(dim=[2, 3])[0]  # (C,)
        activations = self._activations[0]              # (C, H, W)

        # Weighted combination of activation maps
        cam = torch.einsum("c,chw->hw", weights, activations)
        cam = torch.relu(cam)

        # Resize to input spatial resolution
        h, w = image.shape[2], image.shape[3]
        cam_np = cam.cpu().numpy()
        cam_resized = _resize_array(cam_np, (h, w))

        # Normalize to [0, 1]
        cam_min, cam_max = cam_resized.min(), cam_resized.max()
        if cam_max - cam_min > 1e-8:
            cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)
        else:
            cam_resized = np.zeros_like(cam_resized)

        return cam_resized


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resize_array(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2D numpy array using PIL bicubic interpolation."""
    pil_img = Image.fromarray(arr.astype(np.float32))
    pil_img = pil_img.resize((target_hw[1], target_hw[0]), Image.BICUBIC)
    return np.array(pil_img)


def _denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized (C, H, W) tensor back to uint8 (H, W, 3) array."""
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    img = tensor.cpu().numpy() * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img.transpose(1, 2, 0)  # (H, W, 3)


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a heatmap over an RGB image.

    Args:
        image_rgb: RGB image array (H, W, 3) uint8.
        heatmap: Normalized heatmap (H, W) in [0, 1].
        alpha: Heatmap opacity blend factor.
        colormap: Matplotlib colormap name.

    Returns:
        Blended RGB image (H, W, 3) uint8.
    """
    cmap = plt.get_cmap(colormap)
    heatmap_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (image_rgb.astype(float) * (1 - alpha) + heatmap_rgb.astype(float) * alpha)
    return blended.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Batch saliency generation
# ---------------------------------------------------------------------------

def generate_saliency_maps(
    model: nn.Module,
    image_tensor: torch.Tensor,
    label_names: list[str] = CHEXPERT_LABELS,
    top_k: int = 3,
    save_dir: str | Path = "results/gradcam",
    image_name: str = "sample",
    device: Optional[torch.device] = None,
) -> dict[str, np.ndarray]:
    """Generate and save Grad-CAM overlays for the top-k predicted labels.

    Args:
        model: Trained classifier.
        image_tensor: Single image tensor of shape (1, C, H, W) or (C, H, W).
        label_names: List of all label names.
        top_k: Number of top-confidence labels to visualize.
        save_dir: Directory to save overlay images.
        image_name: Base filename for saved overlays.
        device: Torch device. Inferred from model if None.

    Returns:
        Dictionary mapping label name -> heatmap array.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Get top-k predicted class indices
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)[0]
    top_k_indices = probs.argsort(descending=True)[:top_k].tolist()

    image_rgb = _denormalize_image(image_tensor[0])
    heatmaps: dict[str, np.ndarray] = {}

    target_layer = _get_target_layer(model)

    with GradCAM(model, target_layer) as gcam:
        for idx in top_k_indices:
            label = label_names[idx]
            confidence = probs[idx].item()

            try:
                heatmap = gcam.generate(image_tensor.clone(), class_idx=idx)
            except RuntimeError as exc:
                logger.warning("Grad-CAM failed for %s: %s", label, exc)
                continue

            overlay = overlay_heatmap(image_rgb, heatmap)
            heatmaps[label] = heatmap

            _save_overlay(
                image_rgb, overlay, heatmap,
                title=f"{label} (conf={confidence:.3f})",
                save_path=save_dir / f"{image_name}_{label.replace(' ', '_')}.png",
            )

    return heatmaps


def _get_target_layer(model: nn.Module) -> nn.Module:
    """Retrieve the last convolutional block for Grad-CAM targeting."""
    if hasattr(model, "get_last_conv_layer"):
        return model.get_last_conv_layer()
    if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        return model.features.denseblock4
    raise ValueError(
        "Cannot find last conv layer. Implement `get_last_conv_layer()` on your model."
    )


def _save_overlay(
    original: np.ndarray,
    overlay: np.ndarray,
    heatmap: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Save a 3-panel figure: original | heatmap | overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Grad-CAM overlay: %s", save_path)

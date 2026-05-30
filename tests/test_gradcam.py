"""Tests for Grad-CAM saliency visualization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.constants import CHEXPERT_LABELS, IMAGE_SIZE, NUM_CLASSES
from src.gradcam import (
    GradCAM,
    _denormalize_image,
    _get_target_layer,
    _resize_array,
    generate_saliency_maps,
    overlay_heatmap,
)
from src.model import CheXpertClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model() -> CheXpertClassifier:
    """Untrained model on CPU."""
    m = CheXpertClassifier(pretrained=False, freeze_early=False)
    m.eval()
    return m


@pytest.fixture
def small_image() -> torch.Tensor:
    """Single image tensor (1, 3, 64, 64).
    Must be >=64px so DenseNet BatchNorm doesn't receive a 1x1 spatial map
    in the intermediate layers when in training mode.
    """
    return torch.randn(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# GradCAM context manager
# ---------------------------------------------------------------------------

class TestGradCAM:
    def test_hooks_removed_after_context(self, small_model):
        target_layer = _get_target_layer(small_model)
        with GradCAM(small_model, target_layer) as gcam:
            pass
        # Hooks are internal; ensure model still works
        dummy = torch.randn(1, 3, 32, 32)
        out = small_model(dummy)
        assert out.shape == (1, NUM_CLASSES)

    def test_generate_returns_2d_array(self, small_model, small_image):
        target_layer = _get_target_layer(small_model)
        small_model.train()  # needs grad
        with GradCAM(small_model, target_layer) as gcam:
            heatmap = gcam.generate(small_image.clone().requires_grad_(True), class_idx=0)
        assert heatmap.ndim == 2

    def test_heatmap_shape_matches_input(self, small_model, small_image):
        target_layer = _get_target_layer(small_model)
        small_model.train()
        with GradCAM(small_model, target_layer) as gcam:
            heatmap = gcam.generate(small_image.clone().requires_grad_(True), class_idx=0)
        h, w = small_image.shape[2], small_image.shape[3]
        assert heatmap.shape == (h, w)

    def test_heatmap_normalized(self, small_model, small_image):
        target_layer = _get_target_layer(small_model)
        small_model.train()
        with GradCAM(small_model, target_layer) as gcam:
            heatmap = gcam.generate(small_image.clone().requires_grad_(True), class_idx=0)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_different_class_indices_may_differ(self, small_model, small_image):
        target_layer = _get_target_layer(small_model)
        small_model.train()
        with GradCAM(small_model, target_layer) as gcam:
            h0 = gcam.generate(small_image.clone().requires_grad_(True), class_idx=0)
            h1 = gcam.generate(small_image.clone().requires_grad_(True), class_idx=1)
        # They could be the same in untrained models, but the call must not error
        assert h0.shape == h1.shape


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestDenormalizeImage:
    def test_output_shape(self):
        tensor = torch.randn(3, 32, 32)
        result = _denormalize_image(tensor)
        assert result.shape == (32, 32, 3)

    def test_output_dtype(self):
        tensor = torch.randn(3, 32, 32)
        result = _denormalize_image(tensor)
        assert result.dtype == np.uint8

    def test_values_in_byte_range(self):
        tensor = torch.randn(3, 32, 32)
        result = _denormalize_image(tensor)
        assert result.min() >= 0
        assert result.max() <= 255


class TestResizeArray:
    def test_resize_up(self):
        arr = np.random.rand(8, 8).astype(np.float32)
        resized = _resize_array(arr, (32, 32))
        assert resized.shape == (32, 32)

    def test_resize_down(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        resized = _resize_array(arr, (16, 16))
        assert resized.shape == (16, 16)

    def test_resize_identity(self):
        arr = np.random.rand(16, 16).astype(np.float32)
        resized = _resize_array(arr, (16, 16))
        assert resized.shape == (16, 16)


class TestOverlayHeatmap:
    def test_output_shape(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        heatmap = np.zeros((32, 32), dtype=np.float32)
        result = overlay_heatmap(image, heatmap)
        assert result.shape == (32, 32, 3)

    def test_output_dtype(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        heatmap = np.zeros((32, 32), dtype=np.float32)
        result = overlay_heatmap(image, heatmap)
        assert result.dtype == np.uint8

    def test_values_in_byte_range(self):
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        heatmap = np.random.rand(32, 32).astype(np.float32)
        result = overlay_heatmap(image, heatmap)
        assert result.min() >= 0
        assert result.max() <= 255


class TestGetTargetLayer:
    def test_returns_module(self, small_model):
        import torch.nn as nn
        layer = _get_target_layer(small_model)
        assert isinstance(layer, nn.Module)

    def test_raises_on_invalid_model(self):
        import torch.nn as nn
        plain_model = nn.Linear(10, 5)
        with pytest.raises(ValueError):
            _get_target_layer(plain_model)


# ---------------------------------------------------------------------------
# generate_saliency_maps (integration)
# ---------------------------------------------------------------------------

class TestGenerateSaliencyMaps:
    def test_returns_dict(self, small_model, small_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_saliency_maps(
                model=small_model,
                image_tensor=small_image,
                top_k=2,
                save_dir=tmpdir,
                image_name="test",
                device=torch.device("cpu"),
            )
        assert isinstance(result, dict)

    def test_returns_top_k_entries(self, small_model, small_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_saliency_maps(
                model=small_model,
                image_tensor=small_image,
                top_k=3,
                save_dir=tmpdir,
                image_name="test",
                device=torch.device("cpu"),
            )
        assert len(result) <= 3

    def test_saves_overlay_files(self, small_model, small_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_saliency_maps(
                model=small_model,
                image_tensor=small_image,
                top_k=2,
                save_dir=tmpdir,
                image_name="xray",
                device=torch.device("cpu"),
            )
            png_files = list(Path(tmpdir).glob("*.png"))
        assert len(png_files) >= 1

    def test_accepts_3d_tensor(self, small_model):
        """generate_saliency_maps should accept (C, H, W) without explicit batch dim."""
        img = torch.randn(3, 64, 64)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_saliency_maps(
                model=small_model,
                image_tensor=img,
                top_k=1,
                save_dir=tmpdir,
                image_name="test",
                device=torch.device("cpu"),
            )
        assert isinstance(result, dict)

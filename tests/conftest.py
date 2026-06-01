"""Shared pytest fixtures for CheXpert classifier tests.

All fixtures use mock data — no CheXpert download required.
"""

from __future__ import annotations

import io
import base64

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.constants import NUM_CLASSES
from src.data import (
    MockCheXpertDataset,
    build_dataloaders,
    get_train_transforms,
    get_val_transforms,
)
from src.model import CheXpertClassifier, build_model


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> torch.device:
    """Use CPU for all tests (CI compatibility)."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_train_dataset() -> MockCheXpertDataset:
    return MockCheXpertDataset(
        num_samples=80,
        image_size=64,  # small for speed
        transform=get_train_transforms(64),
        seed=0,
    )


@pytest.fixture(scope="session")
def mock_val_dataset() -> MockCheXpertDataset:
    return MockCheXpertDataset(
        num_samples=40,
        image_size=64,
        transform=get_val_transforms(64),
        seed=1,
    )


@pytest.fixture(scope="session")
def mock_test_dataset() -> MockCheXpertDataset:
    return MockCheXpertDataset(
        num_samples=40,
        image_size=64,
        transform=get_val_transforms(64),
        seed=2,
    )


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dataloaders(
    mock_train_dataset: MockCheXpertDataset,
    mock_val_dataset: MockCheXpertDataset,
    mock_test_dataset: MockCheXpertDataset,
) -> dict[str, DataLoader]:
    return build_dataloaders(
        mock_train_dataset,
        mock_val_dataset,
        mock_test_dataset,
        batch_size=8,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model(device: torch.device) -> CheXpertClassifier:
    """Untrained model on CPU — avoids ImageNet download in CI."""
    m, _ = build_model(pretrained=False, device="cpu", freeze_early=False)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Image tensors and batch
# ---------------------------------------------------------------------------

@pytest.fixture
def single_image_tensor() -> torch.Tensor:
    """Single normalized image tensor (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def batch_image_tensor() -> torch.Tensor:
    """Batch of 4 images (4, 3, 64, 64)."""
    return torch.randn(4, 3, 64, 64)


@pytest.fixture
def sample_targets() -> torch.Tensor:
    """Binary target tensor (4, NUM_CLASSES)."""
    rng = np.random.default_rng(42)
    labels = (rng.random((4, NUM_CLASSES)) > 0.5).astype(np.float32)
    return torch.tensor(labels)


# ---------------------------------------------------------------------------
# Base64 image fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_b64_image() -> str:
    """A synthetic grayscale image encoded as base64 JPEG."""
    img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

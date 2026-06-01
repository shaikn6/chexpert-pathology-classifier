"""Data pipeline for CheXpert chest X-ray dataset.

Handles real CheXpert CSV format, mock dataset generation for testing,
and PyTorch DataLoader construction with appropriate transforms.

CheXpert label convention:
  1.0  -> positive
  0.0  -> negative
  -1.0 -> uncertain (U-zeroing: treated as 0 during training)
  NaN  -> unmentioned (treated as 0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.constants import (
    CHEXPERT_LABELS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_PREVALENCE,
    DEFAULT_BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Augmented transforms for training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic transforms for validation/test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Real CheXpert dataset
# ---------------------------------------------------------------------------

class CheXpertDataset(Dataset):
    """PyTorch Dataset for CheXpert CSV format.

    Args:
        csv_path: Path to CheXpert CSV file (train.csv / valid.csv).
        data_root: Root directory containing images. Image paths in CSV are
            relative to this root.
        transform: Optional torchvision transform pipeline.
        uncertain_to_zero: Map uncertain (-1.0) labels to 0 (U-zeroing).
    """

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        transform: Optional[Callable] = None,
        uncertain_to_zero: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transform or get_val_transforms()
        self.uncertain_to_zero = uncertain_to_zero

        self.df = pd.read_csv(csv_path)
        self._validate_columns()
        self._preprocess_labels()

    def _validate_columns(self) -> None:
        missing = [c for c in CHEXPERT_LABELS if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing label columns: {missing}")

    def _preprocess_labels(self) -> None:
        """Fill NaN with 0; optionally map -1 to 0 (U-zeroing)."""
        for label in CHEXPERT_LABELS:
            self.df[label] = self.df[label].fillna(0.0)
            if self.uncertain_to_zero:
                self.df[label] = self.df[label].replace(-1.0, 0.0)
        self.labels = self.df[CHEXPERT_LABELS].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        # Resolve the full path and verify it stays within data_root to prevent
        # path traversal via malicious CSV entries (e.g. "../../etc/passwd").
        candidate = (self.data_root / row["Path"]).resolve()
        if not str(candidate).startswith(str(self.data_root.resolve())):
            raise ValueError(
                f"Image path escapes data_root: {row['Path']!r}"
            )
        image_path = candidate
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ---------------------------------------------------------------------------
# Mock dataset (no real data required)
# ---------------------------------------------------------------------------

class MockCheXpertDataset(Dataset):
    """Synthetic CheXpert-compatible dataset for testing and CI.

    Generates random grayscale 320x320 images with realistic label
    distributions drawn from published CheXpert prevalence rates.

    Args:
        num_samples: Number of synthetic samples to generate.
        image_size: Spatial resolution of generated images.
        transform: Optional torchvision transform pipeline.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 200,
        image_size: int = IMAGE_SIZE,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform or get_val_transforms(image_size)

        rng = np.random.default_rng(seed)
        prevalences = np.array(
            [LABEL_PREVALENCE[label] for label in CHEXPERT_LABELS],
            dtype=np.float32,
        )
        self.labels = (
            rng.random(size=(num_samples, len(CHEXPERT_LABELS))) < prevalences
        ).astype(np.float32)

        # Pre-generate pixel arrays as uint8 to save memory
        self._pixel_data = rng.integers(
            0, 256, size=(num_samples, image_size, image_size), dtype=np.uint8
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = self._pixel_data[idx]
        image = Image.fromarray(pixels, mode="L")  # grayscale
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

    def get_label_dataframe(self) -> pd.DataFrame:
        """Return labels as a DataFrame for analysis."""
        return pd.DataFrame(self.labels, columns=CHEXPERT_LABELS)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Build DataLoader dictionary for train / val / (optional) test splits.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Optional test dataset.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        Dictionary with keys 'train', 'val', and optionally 'test'.
    """
    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def build_mock_dataloaders(
    train_samples: int = 400,
    val_samples: int = 100,
    test_samples: int = 100,
    image_size: int = IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, DataLoader]:
    """Convenience factory for all-mock dataloaders (CI/testing/demo).

    Args:
        train_samples: Number of mock training samples.
        val_samples: Number of mock validation samples.
        test_samples: Number of mock test samples.
        image_size: Spatial resolution.
        batch_size: Samples per batch.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders.
    """
    train_ds = MockCheXpertDataset(
        num_samples=train_samples,
        image_size=image_size,
        transform=get_train_transforms(image_size),
        seed=0,
    )
    val_ds = MockCheXpertDataset(
        num_samples=val_samples,
        image_size=image_size,
        transform=get_val_transforms(image_size),
        seed=1,
    )
    test_ds = MockCheXpertDataset(
        num_samples=test_samples,
        image_size=image_size,
        transform=get_val_transforms(image_size),
        seed=2,
    )
    return build_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=0
    )


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced multi-label data.

    Args:
        dataset: Dataset with .labels attribute (N x C float32 array).

    Returns:
        Weight tensor of shape (num_classes,).
    """
    if hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise AttributeError("Dataset must expose a `.labels` numpy array.")

    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    # Weight = neg / pos; clip to avoid division-by-zero
    weights = neg_counts / np.clip(pos_counts, a_min=1, a_max=None)
    weights = weights / weights.mean()  # normalize so mean weight ≈ 1
    return torch.tensor(weights, dtype=torch.float32)

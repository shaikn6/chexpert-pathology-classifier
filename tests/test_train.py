"""Tests for training utilities: loss, early stopping, and short training runs."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.constants import NUM_CLASSES
from src.data import MockCheXpertDataset
from src.model import CheXpertClassifier
from src.train import (
    EarlyStopping,
    build_loss,
    _compute_macro_auc,
    _run_epoch,
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestBuildLoss:
    def test_no_weights(self):
        loss_fn = build_loss()
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_with_weights(self):
        weights = torch.ones(NUM_CLASSES)
        loss_fn = build_loss(class_weights=weights)
        assert loss_fn.pos_weight is not None

    def test_loss_computation(self):
        loss_fn = build_loss()
        logits = torch.zeros(4, NUM_CLASSES)
        targets = torch.zeros(4, NUM_CLASSES)
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_loss_decreases_with_correct_predictions(self):
        loss_fn = build_loss()
        # Perfect logits -> low loss
        targets = torch.ones(4, NUM_CLASSES)
        good_logits = torch.ones(4, NUM_CLASSES) * 10
        bad_logits = torch.ones(4, NUM_CLASSES) * -10
        good_loss = loss_fn(good_logits, targets).item()
        bad_loss = loss_fn(bad_logits, targets).item()
        assert good_loss < bad_loss

    def test_loss_scalar(self):
        loss_fn = build_loss()
        logits = torch.randn(4, NUM_CLASSES)
        targets = torch.randint(0, 2, (4, NUM_CLASSES)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_stops_when_no_improvement(self):
        es = EarlyStopping(patience=3, mode="max")
        es.step(0.5)
        es.step(0.5)
        es.step(0.5)
        assert es.step(0.5)  # 4th non-improvement triggers stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3, mode="max")
        es.step(0.5)
        es.step(0.5)
        result = es.step(0.6)  # improvement
        assert not result
        assert es.counter == 0

    def test_first_call_never_stops(self):
        es = EarlyStopping(patience=1, mode="max")
        assert not es.step(0.0)

    def test_min_delta_threshold(self):
        es = EarlyStopping(patience=2, min_delta=0.1, mode="max")
        es.step(0.5)
        es.step(0.55)  # improvement < min_delta -> counter increments
        result = es.step(0.55)
        assert result  # triggered

    def test_mode_min(self):
        es = EarlyStopping(patience=2, mode="min")
        es.step(1.0)
        es.step(0.9)  # improvement
        assert not es.should_stop

    def test_best_tracks_maximum(self):
        es = EarlyStopping(patience=5, mode="max")
        es.step(0.5)
        es.step(0.8)
        assert es.best == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# _compute_macro_auc
# ---------------------------------------------------------------------------

class TestComputeMacroAUC:
    def test_perfect_predictions(self):
        targets = np.eye(4, dtype=np.float32)
        probs = np.eye(4, dtype=np.float32)
        auc = _compute_macro_auc(targets, probs)
        assert auc == pytest.approx(1.0)

    def test_random_predictions(self):
        rng = np.random.default_rng(42)
        targets = (rng.random((100, 5)) > 0.5).astype(np.float32)
        probs = rng.random((100, 5)).astype(np.float32)
        auc = _compute_macro_auc(targets, probs)
        assert 0.0 <= auc <= 1.0

    def test_single_class_label_skipped(self):
        targets = np.zeros((10, 3), dtype=np.float32)
        targets[:, 0] = 1  # only col 0 has variation -> rest skipped
        targets[0, 1] = 1
        targets[0, 2] = 1
        probs = np.random.rand(10, 3).astype(np.float32)
        auc = _compute_macro_auc(targets, probs)
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# One-epoch run
# ---------------------------------------------------------------------------

class TestRunEpoch:
    @pytest.fixture
    def small_loader(self):
        # image_size >= 64 required: DenseNet BN collapses 32x32 to 1x1 spatially
        ds = MockCheXpertDataset(num_samples=16, image_size=64, seed=0)
        from torch.utils.data import DataLoader
        return DataLoader(ds, batch_size=8)

    @pytest.fixture
    def small_model(self):
        return CheXpertClassifier(pretrained=False, freeze_early=False)

    def test_returns_loss_and_arrays(self, small_model, small_loader):
        loss_fn = build_loss()
        device = torch.device("cpu")
        loss, targets, probs = _run_epoch(
            small_model, small_loader, loss_fn, None, device, is_train=False
        )
        assert isinstance(loss, float)
        assert targets.shape[1] == NUM_CLASSES
        assert probs.shape[1] == NUM_CLASSES
        assert targets.shape[0] == probs.shape[0]

    def test_probs_in_unit_interval(self, small_model, small_loader):
        loss_fn = build_loss()
        device = torch.device("cpu")
        _, _, probs = _run_epoch(
            small_model, small_loader, loss_fn, None, device, is_train=False
        )
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


# ---------------------------------------------------------------------------
# Full train (smoke test)
# ---------------------------------------------------------------------------

class TestTrainFunction:
    def test_train_returns_history(self, dataloaders, model):
        """Smoke test: one epoch of training produces expected history keys."""
        from src.train import train as train_fn
        # Use a fresh trainable model
        m = CheXpertClassifier(pretrained=False, freeze_early=False)
        m.train()
        history = train_fn(
            model=m,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            device=torch.device("cpu"),
            epochs=1,
            lr=1e-4,
            checkpoint_dir="/tmp/test_ckpts",
        )
        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_auc" in history
        assert len(history["train_loss"]) == 1

    def test_history_values_numeric(self, dataloaders):
        from src.train import train as train_fn
        m = CheXpertClassifier(pretrained=False, freeze_early=False)
        history = train_fn(
            model=m,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            device=torch.device("cpu"),
            epochs=1,
            checkpoint_dir="/tmp/test_ckpts2",
        )
        for val in history["train_loss"]:
            assert np.isfinite(val)

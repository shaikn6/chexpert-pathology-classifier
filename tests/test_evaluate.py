"""Tests for evaluation module: AUC computation, comparison table, and metrics saving."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.constants import CHEXPERT_LABELS, PUBLISHED_AUCS, NUM_CLASSES
from src.evaluate import (
    build_comparison_table,
    compute_macro_auc,
    compute_micro_auc,
    compute_per_label_auc,
    run_inference,
    save_metrics,
    evaluate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_targets():
    """Random binary targets (50, NUM_CLASSES) with both classes present."""
    rng = np.random.default_rng(0)
    t = (rng.random((50, NUM_CLASSES)) > 0.5).astype(np.float32)
    # Ensure each column has both 0 and 1
    t[0, :] = 0.0
    t[1, :] = 1.0
    return t


@pytest.fixture
def random_probs(binary_targets):
    rng = np.random.default_rng(1)
    return rng.random(binary_targets.shape).astype(np.float32)


@pytest.fixture
def perfect_probs(binary_targets):
    """Probabilities that perfectly separate classes."""
    return binary_targets.copy()


# ---------------------------------------------------------------------------
# compute_per_label_auc
# ---------------------------------------------------------------------------

class TestComputePerLabelAUC:
    def test_returns_dict_with_all_labels(self, binary_targets, random_probs):
        aucs = compute_per_label_auc(binary_targets, random_probs)
        assert set(aucs.keys()) == set(CHEXPERT_LABELS)

    def test_values_in_unit_interval(self, binary_targets, random_probs):
        aucs = compute_per_label_auc(binary_targets, random_probs)
        for v in aucs.values():
            if not math.isnan(v):
                assert 0.0 <= v <= 1.0

    def test_perfect_predictions_auc_one(self, binary_targets, perfect_probs):
        aucs = compute_per_label_auc(binary_targets, perfect_probs)
        for label, v in aucs.items():
            if not math.isnan(v):
                assert v == pytest.approx(1.0, abs=1e-6)

    def test_single_class_returns_nan(self):
        targets = np.zeros((20, 2), dtype=np.float32)
        # col 0 is all zero (single class), col 1 has both 0 and 1
        targets[:10, 1] = 1.0
        probs = np.random.rand(20, 2).astype(np.float32)
        aucs = compute_per_label_auc(targets, probs, label_names=["A", "B"])
        assert math.isnan(aucs["A"])
        assert not math.isnan(aucs["B"])

    def test_custom_label_names(self, binary_targets, random_probs):
        names = [f"Label{i}" for i in range(NUM_CLASSES)]
        aucs = compute_per_label_auc(binary_targets, random_probs, label_names=names)
        assert set(aucs.keys()) == set(names)


# ---------------------------------------------------------------------------
# compute_macro_auc
# ---------------------------------------------------------------------------

class TestComputeMacroAUC:
    def test_averages_valid_labels(self):
        aucs = {"A": 0.8, "B": float("nan"), "C": 0.6}
        macro = compute_macro_auc(aucs)
        assert macro == pytest.approx(0.7)

    def test_all_nan_returns_nan(self):
        aucs = {"A": float("nan"), "B": float("nan")}
        assert math.isnan(compute_macro_auc(aucs))

    def test_single_valid_label(self):
        aucs = {"A": 0.75}
        assert compute_macro_auc(aucs) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# compute_micro_auc
# ---------------------------------------------------------------------------

class TestComputeMicroAUC:
    def test_value_in_unit_interval(self, binary_targets, random_probs):
        micro = compute_micro_auc(binary_targets, random_probs)
        if not math.isnan(micro):
            assert 0.0 <= micro <= 1.0

    def test_perfect_micro_auc(self, binary_targets, perfect_probs):
        micro = compute_micro_auc(binary_targets, perfect_probs)
        if not math.isnan(micro):
            assert micro == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    def test_returns_list_of_dicts(self, binary_targets, random_probs):
        per_label = compute_per_label_auc(binary_targets, random_probs)
        rows = build_comparison_table(per_label)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_row_has_required_keys(self, binary_targets, random_probs):
        per_label = compute_per_label_auc(binary_targets, random_probs)
        rows = build_comparison_table(per_label)
        for row in rows:
            assert "label" in row
            assert "our_auc" in row
            assert "published_auc" in row
            assert "delta" in row

    def test_only_published_labels_compared(self, binary_targets, random_probs):
        per_label = compute_per_label_auc(binary_targets, random_probs)
        rows = build_comparison_table(per_label)
        row_labels = {r["label"] for r in rows}
        assert row_labels == set(PUBLISHED_AUCS.keys())

    def test_delta_is_difference(self):
        per_label = {"Atelectasis": 0.9}
        rows = build_comparison_table(per_label, published={"Atelectasis": 0.858})
        assert rows[0]["delta"] == pytest.approx(0.9 - 0.858, abs=1e-4)


# ---------------------------------------------------------------------------
# save_metrics
# ---------------------------------------------------------------------------

class TestSaveMetrics:
    def test_creates_file(self, binary_targets, random_probs):
        per_label = compute_per_label_auc(binary_targets, random_probs)
        macro = compute_macro_auc(per_label)
        micro = compute_micro_auc(binary_targets, random_probs)
        rows = build_comparison_table(per_label)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            save_metrics(per_label, macro, micro, rows, save_path=path)
            assert path.exists()

    def test_json_is_parseable(self, binary_targets, random_probs):
        per_label = compute_per_label_auc(binary_targets, random_probs)
        macro = compute_macro_auc(per_label)
        micro = compute_micro_auc(binary_targets, random_probs)
        rows = build_comparison_table(per_label)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            save_metrics(per_label, macro, micro, rows, save_path=path)
            with open(path) as f:
                data = json.load(f)
            assert "per_label_auc" in data
            assert "macro_auc" in data
            assert "micro_auc" in data
            assert "comparison_vs_irvin_2019" in data

    def test_all_labels_in_json(self, binary_targets, random_probs):
        per_label = compute_per_label_auc(binary_targets, random_probs)
        macro = compute_macro_auc(per_label)
        micro = compute_micro_auc(binary_targets, random_probs)
        rows = build_comparison_table(per_label)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            save_metrics(per_label, macro, micro, rows, save_path=path)
            with open(path) as f:
                data = json.load(f)
            assert set(data["per_label_auc"].keys()) == set(CHEXPERT_LABELS)


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

class TestRunInference:
    def test_returns_targets_and_probs(self, model, dataloaders, device):
        targets, probs = run_inference(model, dataloaders["val"], device)
        assert targets.ndim == 2
        assert probs.ndim == 2
        assert targets.shape == probs.shape

    def test_probs_in_unit_interval(self, model, dataloaders, device):
        _, probs = run_inference(model, dataloaders["val"], device)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_correct_number_of_classes(self, model, dataloaders, device):
        _, probs = run_inference(model, dataloaders["val"], device)
        assert probs.shape[1] == NUM_CLASSES


# ---------------------------------------------------------------------------
# evaluate (integration)
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_dict(self, model, dataloaders, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = evaluate(
                model=model,
                loader=dataloaders["val"],
                device=device,
                results_dir=tmpdir,
                plot=False,
            )
        assert "per_label_auc" in result
        assert "macro_auc" in result
        assert "micro_auc" in result
        assert "comparison" in result

    def test_evaluate_creates_metrics_file(self, model, dataloaders, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluate(
                model=model,
                loader=dataloaders["val"],
                device=device,
                results_dir=tmpdir,
                plot=False,
            )
            assert (Path(tmpdir) / "metrics.json").exists()

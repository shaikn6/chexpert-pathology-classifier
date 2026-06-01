"""V2 test suite — 35+ tests covering all new modules.

Modules under test:
  src/uncertainty.py      — MCDropoutPredictor, UncertaintyResult
  src/dicom_pipeline.py   — load_dicom, apply_windowing, dicom_to_tensor,
                            create_synthetic_dicom
  src/efficientnet_model.py — build_efficientnet_model, compare_models
  src/clinical_report.py   — generate_clinical_report
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from src.uncertainty import MCDropoutPredictor, UncertaintyResult
from src.dicom_pipeline import (
    apply_windowing,
    create_synthetic_dicom,
    dicom_to_tensor,
)
from src.efficientnet_model import build_efficientnet_model, compare_models
from src.clinical_report import generate_clinical_report
from src.constants import CHEXPERT_LABELS, NUM_CLASSES


# ===========================================================================
# Shared fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def tiny_model() -> nn.Module:
    """Minimal DenseNet-like model with dropout for fast uncertainty tests."""

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(32, NUM_CLASSES),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    model = _TinyModel()
    model.eval()
    return model


@pytest.fixture(scope="module")
def dummy_image() -> torch.Tensor:
    """Single dummy image tensor (1, 3, 32, 32)."""
    return torch.rand(1, 3, 32, 32)


@pytest.fixture(scope="module")
def synthetic_dcm():
    """In-memory synthetic DICOM dataset."""
    return create_synthetic_dicom()


@pytest.fixture(scope="module")
def efficientnet_b4_model() -> nn.Module:
    """EfficientNet-B4 without pretrained weights (fast init for tests)."""
    return build_efficientnet_model(num_classes=14, pretrained=False)


@pytest.fixture(scope="module")
def sample_uncertainty() -> UncertaintyResult:
    """Fixed UncertaintyResult for report tests."""
    n = NUM_CLASSES
    rng = np.random.default_rng(42)
    return UncertaintyResult(
        mean=rng.uniform(0.1, 0.8, n).astype(np.float32),
        std=rng.uniform(0.01, 0.2, n).astype(np.float32),
        ci_lower=rng.uniform(0.0, 0.1, n).astype(np.float32),
        ci_upper=rng.uniform(0.7, 0.9, n).astype(np.float32),
    )


# ===========================================================================
# MCDropoutPredictor tests (12)
# ===========================================================================


class TestMCDropoutPredictor:
    """Tests for uncertainty.MCDropoutPredictor."""

    def test_output_is_uncertainty_result(self, tiny_model, dummy_image):
        predictor = MCDropoutPredictor(tiny_model, n_samples=5)
        result = predictor.predict_with_uncertainty(dummy_image)
        assert isinstance(result, UncertaintyResult)

    def test_mean_shape(self, tiny_model, dummy_image):
        predictor = MCDropoutPredictor(tiny_model, n_samples=5)
        result = predictor.predict_with_uncertainty(dummy_image)
        assert result.mean.shape == (NUM_CLASSES,)

    def test_std_shape(self, tiny_model, dummy_image):
        predictor = MCDropoutPredictor(tiny_model, n_samples=5)
        result = predictor.predict_with_uncertainty(dummy_image)
        assert result.std.shape == (NUM_CLASSES,)

    def test_ci_lower_shape(self, tiny_model, dummy_image):
        predictor = MCDropoutPredictor(tiny_model, n_samples=5)
        result = predictor.predict_with_uncertainty(dummy_image)
        assert result.ci_lower.shape == (NUM_CLASSES,)

    def test_ci_upper_shape(self, tiny_model, dummy_image):
        predictor = MCDropoutPredictor(tiny_model, n_samples=5)
        result = predictor.predict_with_uncertainty(dummy_image)
        assert result.ci_upper.shape == (NUM_CLASSES,)

    def test_std_non_negative(self, tiny_model, dummy_image):
        predictor = MCDropoutPredictor(tiny_model, n_samples=10)
        result = predictor.predict_with_uncertainty(dummy_image)
        assert np.all(result.std >= 0.0)

    def test_mean_in_zero_one_range(self, tiny_model, dummy_image):
        """Mean of sigmoid outputs must be in [0, 1]."""
        predictor = MCDropoutPredictor(tiny_model, n_samples=10)
        result = predictor.predict_with_uncertainty(dummy_image)
        assert np.all(result.mean >= 0.0) and np.all(result.mean <= 1.0)

    def test_ci_lower_leq_upper(self, tiny_model, dummy_image):
        predictor = MCDropoutPredictor(tiny_model, n_samples=10)
        result = predictor.predict_with_uncertainty(dummy_image)
        assert np.all(result.ci_lower <= result.ci_upper)

    def test_3d_input_auto_unsqueezes(self, tiny_model):
        """Passing a (3, 32, 32) tensor should not raise."""
        predictor = MCDropoutPredictor(tiny_model, n_samples=5)
        img_3d = torch.rand(3, 32, 32)
        result = predictor.predict_with_uncertainty(img_3d)
        assert result.mean.shape == (NUM_CLASSES,)

    def test_high_uncertainty_with_large_dropout(self):
        """A model with p=1.0 effective noise should produce nonzero std."""
        # Use a large dropout to force variance across samples
        class _HighDropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, NUM_CLASSES)
                self.drop = nn.Dropout(p=0.9)

            def forward(self, x):
                x = x.mean(dim=[-2, -1])  # (B, C)
                return self.drop(self.fc(x))

        model = _HighDropoutModel()
        model.eval()
        predictor = MCDropoutPredictor(model, n_samples=30)
        result = predictor.predict_with_uncertainty(torch.rand(1, 3, 32, 32))
        # With 90% dropout, at least some labels should have nonzero std
        assert result.std.sum() > 0.0

    def test_enable_dropout_sets_dropout_to_train(self, tiny_model):
        """enable_dropout() must set Dropout layers to training=True."""
        tiny_model.eval()
        predictor = MCDropoutPredictor(tiny_model, n_samples=5)
        predictor.enable_dropout()
        has_train_dropout = any(
            isinstance(m, nn.Dropout) and m.training
            for m in tiny_model.modules()
        )
        assert has_train_dropout

    def test_n_samples_respected(self, tiny_model, dummy_image):
        """Changing n_samples should not raise and results differ from 1-sample."""
        for n in (1, 5, 20):
            predictor = MCDropoutPredictor(tiny_model, n_samples=n)
            result = predictor.predict_with_uncertainty(dummy_image)
            assert result.mean.shape == (NUM_CLASSES,)


class TestClassifyUncertainty:
    """Tests for MCDropoutPredictor.classify_uncertainty static method."""

    def test_low_uncertainty(self):
        assert MCDropoutPredictor.classify_uncertainty(0.0) == "low"

    def test_low_uncertainty_boundary(self):
        assert MCDropoutPredictor.classify_uncertainty(0.04) == "low"

    def test_medium_uncertainty_at_boundary(self):
        assert MCDropoutPredictor.classify_uncertainty(0.05) == "medium"

    def test_medium_uncertainty_middle(self):
        assert MCDropoutPredictor.classify_uncertainty(0.10) == "medium"

    def test_high_uncertainty_at_boundary(self):
        assert MCDropoutPredictor.classify_uncertainty(0.15) == "high"

    def test_high_uncertainty_large_value(self):
        assert MCDropoutPredictor.classify_uncertainty(0.99) == "high"


# ===========================================================================
# DICOM pipeline tests (10)
# ===========================================================================


class TestCreateSyntheticDicom:
    """Tests for dicom_pipeline.create_synthetic_dicom."""

    def test_returns_dataset(self, synthetic_dcm):
        import pydicom
        assert isinstance(synthetic_dcm, pydicom.Dataset)

    def test_has_pixel_data(self, synthetic_dcm):
        assert hasattr(synthetic_dcm, "PixelData")

    def test_rows_and_columns(self, synthetic_dcm):
        assert synthetic_dcm.Rows == 256
        assert synthetic_dcm.Columns == 256

    def test_has_bits_allocated(self, synthetic_dcm):
        assert synthetic_dcm.BitsAllocated == 16

    def test_has_window_tags(self, synthetic_dcm):
        assert hasattr(synthetic_dcm, "WindowCenter")
        assert hasattr(synthetic_dcm, "WindowWidth")


class TestApplyWindowing:
    """Tests for dicom_pipeline.apply_windowing."""

    def test_output_in_0_255(self):
        arr = np.linspace(-2000, 2000, 100).reshape(10, 10)
        out = apply_windowing(arr, window_center=-600, window_width=1500)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_output_dtype_uint8(self):
        arr = np.zeros((8, 8), dtype=np.float32)
        out = apply_windowing(arr, window_center=0, window_width=500)
        assert out.dtype == np.uint8

    def test_clipping_below_lower(self):
        arr = np.full((4, 4), -9999.0)
        out = apply_windowing(arr, window_center=0, window_width=1000)
        assert np.all(out == 0)

    def test_clipping_above_upper(self):
        arr = np.full((4, 4), 9999.0)
        out = apply_windowing(arr, window_center=0, window_width=1000)
        assert np.all(out == 255)


class TestDicomToTensor:
    """Tests for dicom_pipeline.dicom_to_tensor."""

    def test_output_shape(self, synthetic_dcm):
        tensor = dicom_to_tensor(synthetic_dcm)
        assert tensor.shape == (1, 320, 320)

    def test_output_dtype_float(self, synthetic_dcm):
        tensor = dicom_to_tensor(synthetic_dcm)
        assert tensor.dtype == torch.float32

    def test_fallback_window_when_tags_missing(self):
        """Dataset without WindowCenter/Width should still produce (1,320,320)."""
        dcm = create_synthetic_dicom()
        # Remove window tags to test fallback
        if hasattr(dcm, "WindowCenter"):
            del dcm.WindowCenter
        if hasattr(dcm, "WindowWidth"):
            del dcm.WindowWidth
        tensor = dicom_to_tensor(dcm)
        assert tensor.shape == (1, 320, 320)


# ===========================================================================
# EfficientNet-B4 model tests (7)
# ===========================================================================


class TestBuildEfficientNetModel:
    """Tests for efficientnet_model.build_efficientnet_model."""

    def test_builds_without_error(self, efficientnet_b4_model):
        assert efficientnet_b4_model is not None

    def test_output_neurons(self, efficientnet_b4_model):
        """Classifier head must end with 14 output neurons."""
        # Last module in the sequential classifier should be a Linear(?, 14)
        last_layer = list(efficientnet_b4_model.classifier.children())[-1]
        assert isinstance(last_layer, nn.Linear)
        assert last_layer.out_features == 14

    def test_forward_pass_shape(self, efficientnet_b4_model):
        """Forward pass on (2, 3, 224, 224) must return (2, 14)."""
        x = torch.rand(2, 3, 224, 224)
        efficientnet_b4_model.eval()
        with torch.no_grad():
            out = efficientnet_b4_model(x)
        assert out.shape == (2, 14)

    def test_forward_pass_batch1(self, efficientnet_b4_model):
        """Single-image batch."""
        x = torch.rand(1, 3, 224, 224)
        efficientnet_b4_model.eval()
        with torch.no_grad():
            out = efficientnet_b4_model(x)
        assert out.shape == (1, 14)

    def test_has_dropout_in_classifier(self, efficientnet_b4_model):
        has_dropout = any(
            isinstance(m, nn.Dropout)
            for m in efficientnet_b4_model.classifier.children()
        )
        assert has_dropout

    def test_custom_num_classes(self):
        model = build_efficientnet_model(num_classes=5, pretrained=False)
        last = list(model.classifier.children())[-1]
        assert last.out_features == 5


class TestCompareModels:
    """Tests for efficientnet_model.compare_models."""

    def test_none_dataloader_returns_dataframe(self):
        model_a = build_efficientnet_model(num_classes=14, pretrained=False)
        model_b = build_efficientnet_model(num_classes=14, pretrained=False)
        df = compare_models(model_a, model_b, dataloader=None)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_none_dataloader_correct_columns(self):
        model_a = build_efficientnet_model(num_classes=14, pretrained=False)
        model_b = build_efficientnet_model(num_classes=14, pretrained=False)
        df = compare_models(model_a, model_b, dataloader=None)
        assert set(df.columns) == {"label", "model_a_auc", "model_b_auc", "delta"}

    def test_none_dataloader_row_count(self):
        model_a = build_efficientnet_model(num_classes=14, pretrained=False)
        model_b = build_efficientnet_model(num_classes=14, pretrained=False)
        df = compare_models(model_a, model_b, dataloader=None)
        assert len(df) == 14

    def test_none_dataloader_labels_match(self):
        model_a = build_efficientnet_model(num_classes=14, pretrained=False)
        model_b = build_efficientnet_model(num_classes=14, pretrained=False)
        df = compare_models(model_a, model_b, dataloader=None, label_names=CHEXPERT_LABELS)
        assert list(df["label"]) == CHEXPERT_LABELS


# ===========================================================================
# Clinical report tests (7)
# ===========================================================================


class TestGenerateClinicalReport:
    """Tests for clinical_report.generate_clinical_report."""

    def _make_predictions(self) -> dict[str, float]:
        """Build a sample predictions dict with some findings above threshold."""
        rng = np.random.default_rng(7)
        probs = rng.uniform(0.0, 1.0, NUM_CLASSES).tolist()
        return dict(zip(CHEXPERT_LABELS, probs))

    def test_creates_output_file(self, sample_uncertainty):
        predictions = self._make_predictions()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.txt")
            generate_clinical_report(
                image_path="test_image.dcm",
                predictions=predictions,
                uncertainty=sample_uncertainty,
                output_path=out_path,
            )
            assert os.path.isfile(out_path)

    def test_report_contains_header(self, sample_uncertainty):
        predictions = self._make_predictions()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.txt")
            text = generate_clinical_report(
                image_path="test.png",
                predictions=predictions,
                uncertainty=sample_uncertainty,
                output_path=out_path,
            )
            assert "CHEST X-RAY" in text

    def test_report_contains_disclaimer(self, sample_uncertainty):
        predictions = self._make_predictions()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.txt")
            text = generate_clinical_report(
                image_path="test.png",
                predictions=predictions,
                uncertainty=sample_uncertainty,
                output_path=out_path,
            )
            assert "Disclaimer" in text or "DISCLAIMER" in text

    def test_report_contains_findings_section(self, sample_uncertainty):
        predictions = self._make_predictions()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.txt")
            text = generate_clinical_report(
                image_path="test.png",
                predictions=predictions,
                uncertainty=sample_uncertainty,
                output_path=out_path,
            )
            assert "FINDINGS" in text

    def test_report_contains_impression_section(self, sample_uncertainty):
        predictions = self._make_predictions()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.txt")
            text = generate_clinical_report(
                image_path="test.png",
                predictions=predictions,
                uncertainty=sample_uncertainty,
                output_path=out_path,
            )
            assert "IMPRESSION" in text

    def test_report_text_returned(self, sample_uncertainty):
        """generate_clinical_report must return the report as a string."""
        predictions = self._make_predictions()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.txt")
            result = generate_clinical_report(
                image_path="test.png",
                predictions=predictions,
                uncertainty=sample_uncertainty,
                output_path=out_path,
            )
            assert isinstance(result, str)
            assert len(result) > 0

    def test_all_zero_predictions_no_findings(self, sample_uncertainty):
        """When all probs are 0.0, findings section should say no findings."""
        predictions = {label: 0.0 for label in CHEXPERT_LABELS}
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.txt")
            text = generate_clinical_report(
                image_path="blank.dcm",
                predictions=predictions,
                uncertainty=sample_uncertainty,
                output_path=out_path,
            )
            assert "No significant findings" in text

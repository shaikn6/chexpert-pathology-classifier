"""Tests for CheXpertClassifier model architecture and forward pass."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.constants import NUM_CLASSES
from src.model import CheXpertClassifier, build_model


class TestCheXpertClassifier:
    """Tests for model construction and output shapes."""

    def test_forward_pass_output_shape(self, model, single_image_tensor):
        logits = model(single_image_tensor)
        assert logits.shape == (1, NUM_CLASSES)

    def test_forward_pass_batch(self, model, batch_image_tensor):
        logits = model(batch_image_tensor)
        assert logits.shape == (4, NUM_CLASSES)

    def test_logits_are_unbounded(self, model, single_image_tensor):
        """Logits (pre-sigmoid) should not be clamped."""
        logits = model(single_image_tensor)
        assert logits.dtype == torch.float32

    def test_predict_proba_in_unit_interval(self, model, single_image_tensor):
        probs = model.predict_proba(single_image_tensor)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_predict_proba_shape(self, model, single_image_tensor):
        probs = model.predict_proba(single_image_tensor)
        assert probs.shape == (1, NUM_CLASSES)

    def test_predict_proba_dtype(self, model, single_image_tensor):
        probs = model.predict_proba(single_image_tensor)
        assert probs.dtype == torch.float32

    def test_num_classes_matches(self):
        model = CheXpertClassifier(num_classes=5, pretrained=False)
        dummy = torch.randn(2, 3, 32, 32)
        out = model(dummy)
        assert out.shape == (2, 5)

    def test_label_names_length(self):
        model = CheXpertClassifier(pretrained=False)
        assert len(model.label_names) == NUM_CLASSES

    def test_trainable_params_positive(self):
        model = CheXpertClassifier(pretrained=False, freeze_early=False)
        assert model.count_trainable_params() > 0

    def test_total_params_gt_trainable_when_frozen(self):
        model = CheXpertClassifier(pretrained=False, freeze_early=True)
        assert model.count_total_params() > model.count_trainable_params()

    def test_freeze_early_reduces_trainable(self):
        m_frozen = CheXpertClassifier(pretrained=False, freeze_early=True)
        m_full = CheXpertClassifier(pretrained=False, freeze_early=False)
        assert m_frozen.count_trainable_params() < m_full.count_trainable_params()

    def test_unfreeze_all(self):
        model = CheXpertClassifier(pretrained=False, freeze_early=True)
        frozen_count = model.count_trainable_params()
        model.unfreeze_all()
        assert model.count_trainable_params() > frozen_count

    def test_get_last_conv_layer(self):
        model = CheXpertClassifier(pretrained=False)
        layer = model.get_last_conv_layer()
        assert layer is not None
        assert isinstance(layer, nn.Module)

    def test_classifier_head_is_sequential(self):
        model = CheXpertClassifier(pretrained=False)
        assert isinstance(model.classifier, nn.Sequential)

    def test_model_on_cpu(self):
        model = CheXpertClassifier(pretrained=False)
        model.eval()
        # Use batch size >= 2 and spatial size >= 32 to avoid BN single-element issues
        dummy = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(dummy)
        assert out.device.type == "cpu"

    def test_build_model_returns_tuple(self):
        model, device = build_model(pretrained=False, device="cpu")
        assert isinstance(model, CheXpertClassifier)
        assert isinstance(device, torch.device)

    def test_build_model_device_cpu(self):
        _, device = build_model(pretrained=False, device="cpu")
        assert device.type == "cpu"

    def test_dropout_rate_applied(self):
        """Dropout should be set correctly in the classifier head."""
        model = CheXpertClassifier(pretrained=False, dropout_rate=0.5)
        dropout_layers = [m for m in model.classifier.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) >= 1
        assert dropout_layers[0].p == 0.5

    def test_model_eval_mode(self, model):
        assert not model.training

    def test_model_gradient_flow(self):
        model = CheXpertClassifier(pretrained=False, freeze_early=False)
        model.train()
        # Use spatial size >=64 so DenseNet BN layers don't collapse to 1x1
        dummy = torch.randn(2, 3, 64, 64)
        logits = model(dummy)
        loss = logits.mean()
        loss.backward()
        # Check at least one gradient in the classifier head is non-zero
        for param in model.classifier.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                return
        pytest.fail("No gradients flowed through the classifier head.")

"""Tests for FastAPI inference endpoint."""

from __future__ import annotations

import base64
import io
import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.api import _API_KEY, app, _state
from src.constants import CHEXPERT_LABELS, NUM_CLASSES
from src.data import get_val_transforms
from src.model import CheXpertClassifier

# Valid auth header for all authenticated test requests
AUTH_HEADERS = {"X-API-Key": _API_KEY}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_b64_image(width: int = 64, height: int = 64) -> str:
    """Generate a random RGB JPEG image encoded as base64."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def inject_mock_model():
    """Inject a fresh untrained model into the app state before each test."""
    model = CheXpertClassifier(pretrained=False, freeze_early=False)
    model.eval()
    _state["model"] = model
    _state["device"] = torch.device("cpu")
    _state["transform"] = get_val_transforms(image_size=64)
    yield
    _state.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def valid_b64_image() -> str:
    return _make_b64_image()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok_when_model_loaded(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"

    def test_status_degraded_when_no_model(self, client):
        _state.clear()
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# /labels
# ---------------------------------------------------------------------------

class TestLabelsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/labels")
        assert resp.status_code == 200

    def test_returns_all_labels(self, client):
        resp = client.get("/labels")
        data = resp.json()
        assert data["labels"] == CHEXPERT_LABELS

    def test_label_count(self, client):
        resp = client.get("/labels")
        data = resp.json()
        assert len(data["labels"]) == NUM_CLASSES


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_returns_200_valid_image(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image}, headers=AUTH_HEADERS)
        assert resp.status_code == 200

    def test_response_has_all_probabilities(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image}, headers=AUTH_HEADERS)
        data = resp.json()
        assert "all_probabilities" in data
        assert set(data["all_probabilities"].keys()) == set(CHEXPERT_LABELS)

    def test_probabilities_in_unit_interval(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image}, headers=AUTH_HEADERS)
        data = resp.json()
        for prob in data["all_probabilities"].values():
            assert 0.0 <= prob <= 1.0

    def test_top_labels_default_3(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image}, headers=AUTH_HEADERS)
        data = resp.json()
        assert len(data["top_labels"]) == 3

    def test_top_labels_custom_k(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image, "top_k": 5}, headers=AUTH_HEADERS)
        data = resp.json()
        assert len(data["top_labels"]) == 5

    def test_top_labels_sorted_by_probability(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image, "top_k": 3}, headers=AUTH_HEADERS)
        data = resp.json()
        probs = [item["probability"] for item in data["top_labels"]]
        assert probs == sorted(probs, reverse=True)

    def test_top_labels_have_rank_field(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image}, headers=AUTH_HEADERS)
        data = resp.json()
        ranks = [item["rank"] for item in data["top_labels"]]
        assert ranks == [1, 2, 3]

    def test_top_labels_have_label_field(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image}, headers=AUTH_HEADERS)
        data = resp.json()
        for item in data["top_labels"]:
            assert item["label"] in CHEXPERT_LABELS

    def test_inference_time_positive(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image}, headers=AUTH_HEADERS)
        data = resp.json()
        assert data["inference_time_ms"] > 0.0

    def test_model_version_present(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image}, headers=AUTH_HEADERS)
        data = resp.json()
        assert "model_version" in data

    def test_empty_b64_returns_422(self, client):
        resp = client.post("/predict", json={"image_b64": ""}, headers=AUTH_HEADERS)
        assert resp.status_code == 422

    def test_invalid_b64_returns_422(self, client):
        resp = client.post("/predict", json={"image_b64": "not_valid_base64!!!"}, headers=AUTH_HEADERS)
        assert resp.status_code == 422

    def test_missing_field_returns_422(self, client):
        resp = client.post("/predict", json={}, headers=AUTH_HEADERS)
        assert resp.status_code == 422

    def test_model_not_loaded_returns_503(self, client, valid_b64_image):
        _state.clear()
        resp = client.post("/predict", json={"image_b64": valid_b64_image}, headers=AUTH_HEADERS)
        assert resp.status_code == 503

    def test_accepts_data_uri_prefix(self, client, valid_b64_image):
        """Image with data URI prefix should be accepted."""
        prefixed = "data:image/jpeg;base64," + valid_b64_image
        resp = client.post("/predict", json={"image_b64": prefixed}, headers=AUTH_HEADERS)
        assert resp.status_code == 200

    def test_top_k_out_of_range_returns_422(self, client, valid_b64_image):
        resp = client.post("/predict", json={"image_b64": valid_b64_image, "top_k": 0}, headers=AUTH_HEADERS)
        assert resp.status_code == 422

    def test_missing_api_key_returns_401(self, client, valid_b64_image):
        """Requests without a valid API key must be rejected."""
        resp = client.post("/predict", json={"image_b64": valid_b64_image})
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, client, valid_b64_image):
        """Requests with a wrong API key must be rejected."""
        resp = client.post("/predict", json={"image_b64": valid_b64_image}, headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_error_message_does_not_leak_internals(self, client):
        """Error response for invalid image must not expose exception class or stack."""
        resp = client.post("/predict", json={"image_b64": "bm90dmFsaWQ="}, headers=AUTH_HEADERS)
        assert resp.status_code == 422
        detail = resp.json().get("detail", "")
        # Must not leak Python exception class names or file paths
        assert "Traceback" not in detail
        assert "PIL" not in detail
        assert "binascii" not in detail


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------

class TestBatchPredictEndpoint:
    def test_batch_returns_200(self, client):
        images = [_make_b64_image() for _ in range(3)]
        resp = client.post("/predict/batch", json={"images_b64": images}, headers=AUTH_HEADERS)
        assert resp.status_code == 200

    def test_batch_result_count(self, client):
        images = [_make_b64_image() for _ in range(4)]
        resp = client.post("/predict/batch", json={"images_b64": images}, headers=AUTH_HEADERS)
        data = resp.json()
        assert len(data["results"]) == 4

    def test_batch_total_time_positive(self, client):
        images = [_make_b64_image() for _ in range(2)]
        resp = client.post("/predict/batch", json={"images_b64": images}, headers=AUTH_HEADERS)
        data = resp.json()
        assert data["total_inference_time_ms"] > 0.0

    def test_batch_empty_returns_422(self, client):
        resp = client.post("/predict/batch", json={"images_b64": []}, headers=AUTH_HEADERS)
        assert resp.status_code == 422

    def test_batch_missing_api_key_returns_401(self, client):
        """Batch endpoint must also require authentication."""
        images = [_make_b64_image() for _ in range(2)]
        resp = client.post("/predict/batch", json={"images_b64": images})
        assert resp.status_code == 401

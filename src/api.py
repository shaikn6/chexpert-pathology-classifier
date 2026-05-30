"""FastAPI inference endpoint for CheXpert pathology classifier.

Accepts a base64-encoded JPEG image and returns per-label probabilities
along with the top-3 predicted pathology labels.

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          Liveness check
    GET  /labels          List all 14 CheXpert labels
    POST /predict         Predict pathologies from a chest X-ray image
    POST /predict/batch   Batch prediction (up to 16 images)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from src.constants import CHEXPERT_LABELS
from src.data import get_val_transforms
from src.model import CheXpertClassifier, build_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security configuration
# ---------------------------------------------------------------------------

# Maximum decoded image size: 20 MB uncompressed (prevents memory DoS)
MAX_IMAGE_BYTES: int = 20 * 1024 * 1024

# API key authentication — set API_KEY env var in production.
# If not set, a random key is generated per process (effectively disabling
# unauthenticated access while still allowing tests that inject _state directly).
_API_KEY: str = os.environ.get("API_KEY", secrets.token_hex(32))

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(api_key: Optional[str] = Depends(_api_key_header)) -> None:
    """Dependency that validates the X-API-Key header."""
    if not api_key or not secrets.compare_digest(api_key, _API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# Allowed CORS origins — override with CORS_ORIGINS env var (comma-separated)
_raw_origins = os.environ.get("CORS_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else []
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single image prediction request."""
    image_b64: str = Field(
        ...,
        description="Base64-encoded JPEG or PNG chest X-ray image.",
        examples=["<base64 string>"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=14,
        description="Number of top labels to return.",
    )

    @field_validator("image_b64")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("image_b64 must not be empty.")
        # Strip data URI prefix before size check
        raw_b64 = v.split(",", 1)[1] if "," in v else v
        # Reject oversized payloads before decoding (base64 overhead ~4/3)
        if len(raw_b64) > MAX_IMAGE_BYTES * 4 // 3 + 64:
            raise ValueError(
                f"image_b64 exceeds maximum allowed size ({MAX_IMAGE_BYTES // (1024 * 1024)} MB)."
            )
        return v


class LabelPrediction(BaseModel):
    """Single label prediction."""
    label: str
    probability: float
    rank: int


class PredictResponse(BaseModel):
    """Prediction response with all label probabilities and top-k labels."""
    all_probabilities: dict[str, float]
    top_labels: list[LabelPrediction]
    inference_time_ms: float
    model_version: str = "1.0.0"


class BatchPredictRequest(BaseModel):
    """Batch prediction request (up to 16 images)."""
    images_b64: list[str] = Field(..., min_length=1, max_length=16)
    top_k: int = Field(default=3, ge=1, le=14)

    @field_validator("images_b64", mode="before")
    @classmethod
    def validate_each_image(cls, v: list) -> list:
        for item in v:
            raw_b64 = item.split(",", 1)[1] if "," in item else item
            if len(raw_b64) > MAX_IMAGE_BYTES * 4 // 3 + 64:
                raise ValueError(
                    f"One or more images exceed the maximum allowed size "
                    f"({MAX_IMAGE_BYTES // (1024 * 1024)} MB)."
                )
        return v


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    results: list[PredictResponse]
    total_inference_time_ms: float


# ---------------------------------------------------------------------------
# Model registry (singleton, loaded at startup)
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("Loading CheXpert classifier...")
    checkpoint_path = os.environ.get("MODEL_CHECKPOINT", "")
    pretrained = os.environ.get("USE_PRETRAINED", "false").lower() == "true"

    model, device = build_model(pretrained=pretrained, freeze_early=False)

    if checkpoint_path and Path(checkpoint_path).exists():
        # weights_only=True prevents arbitrary code execution via malicious
        # pickle payloads embedded in checkpoint files (CWE-502).
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        logger.warning(
            "No checkpoint found at '%s'. Using untrained weights.", checkpoint_path
        )

    model.eval()
    _state["model"] = model
    _state["device"] = device
    _state["transform"] = get_val_transforms()
    logger.info("Model ready on device: %s", device)

    yield

    _state.clear()
    logger.info("Model unloaded.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CheXpert Pathology Classifier",
    description=(
        "Multi-label chest X-ray pathology classification using DenseNet121. "
        "Identifies 14 radiological findings from the CheXpert label set."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — only allow explicitly configured origins (deny all by default)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    """Attach OWASP-recommended security headers to every response."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(image_b64: str) -> Image.Image:
    """Decode base64 string to PIL Image.

    Args:
        image_b64: Base64-encoded image data (with or without data URI prefix).

    Returns:
        PIL Image in RGB mode.

    Raises:
        HTTPException: If decoding or conversion fails.
    """
    try:
        # Strip data URI prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        raw = base64.b64decode(image_b64, validate=True)
        # Enforce decoded size limit before PIL parses (prevents decompression bombs)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Decoded image exceeds maximum permitted size.")
        image = Image.open(io.BytesIO(raw))
        # Verify the image is loadable before trusting it (catches truncated files)
        image.verify()
        # Re-open after verify() (verify() consumes the stream)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return image
    except HTTPException:
        raise
    except Exception:
        # Do not propagate internal exception details to the caller — they may
        # reveal file paths, library internals, or parser state.
        logger.debug("Image decode failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid image data: unable to decode the provided image.",
        )


def _run_single_prediction(image_b64: str, top_k: int) -> tuple[dict[str, float], list[LabelPrediction], float]:
    """Core inference logic for a single image.

    Returns:
        Tuple of (all_probabilities, top_labels, inference_ms).
    """
    model: CheXpertClassifier = _state["model"]
    device: torch.device = _state["device"]
    transform = _state["transform"]

    image = _decode_image(image_b64)
    tensor = transform(image).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().tolist()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    all_probs = {label: round(p, 6) for label, p in zip(CHEXPERT_LABELS, probs)}
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top_labels = [
        LabelPrediction(
            label=CHEXPERT_LABELS[i],
            probability=round(probs[i], 6),
            rank=rank + 1,
        )
        for rank, i in enumerate(sorted_indices[:top_k])
    ]
    return all_probs, top_labels, elapsed_ms


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    """Liveness check."""
    model_loaded = "model" in _state
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": str(model_loaded),
    }


@app.get("/labels", tags=["Info"])
async def get_labels() -> dict[str, list[str]]:
    """Return all 14 CheXpert label names."""
    return {"labels": CHEXPERT_LABELS}


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    dependencies=[Depends(_require_api_key)],
)
async def predict(request: PredictRequest) -> PredictResponse:
    """Classify a single chest X-ray image.

    Accepts a base64-encoded JPEG/PNG chest X-ray and returns per-label
    sigmoid probabilities and the top-k most likely pathologies.

    Args:
        request: PredictRequest with base64 image.

    Returns:
        PredictResponse with probabilities and top labels.
    """
    if "model" not in _state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    all_probs, top_labels, elapsed_ms = _run_single_prediction(
        request.image_b64, request.top_k
    )
    return PredictResponse(
        all_probabilities=all_probs,
        top_labels=top_labels,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Inference"],
    dependencies=[Depends(_require_api_key)],
)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Classify a batch of chest X-ray images (up to 16).

    Args:
        request: BatchPredictRequest with list of base64 images.

    Returns:
        BatchPredictResponse with per-image results.
    """
    if "model" not in _state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    t_batch_start = time.perf_counter()
    results = []
    for img_b64 in request.images_b64:
        all_probs, top_labels, elapsed_ms = _run_single_prediction(img_b64, request.top_k)
        results.append(
            PredictResponse(
                all_probabilities=all_probs,
                top_labels=top_labels,
                inference_time_ms=round(elapsed_ms, 2),
            )
        )
    total_ms = (time.perf_counter() - t_batch_start) * 1000

    return BatchPredictResponse(
        results=results,
        total_inference_time_ms=round(total_ms, 2),
    )

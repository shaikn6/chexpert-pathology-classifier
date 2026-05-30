# chexpert-pathology-classifier

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)
![Tests](https://img.shields.io/badge/Tests-passing-22c55e)
![Stack](https://img.shields.io/badge/Stack-PyTorch%20·%20DenseNet121%20·%20Grad--CAM-6366f1)

Multi-label chest X-ray pathology classification on the CheXpert dataset using DenseNet121 with Grad-CAM saliency visualization. Benchmarked against Irvin et al. 2019 (Stanford CheXpert paper).

## What's New in V2

- **Monte Carlo Dropout Uncertainty Quantification** — 30 stochastic forward passes with dropout enabled at inference time, producing per-label mean predictions, standard deviations, and 95% confidence intervals to flag high-uncertainty findings for radiologist review.
- **DICOM Input Pipeline** — Native `.dcm` file support with window/level normalization (WindowCenter/WindowWidth DICOM tags), automatic fallback to the lung window preset (WC=-600, WW=1500), and conversion to normalized tensors.
- **EfficientNet-B4 Backbone** — A second model option with a 1792-dimensional feature head and a model comparison utility that reports per-label AUC deltas between any two classifiers.
- **Clinical Report Generator** — Structured plain-text radiology reports with labeled findings above a 30% confidence threshold, per-finding uncertainty tiers, impression summary, and a mandatory AI disclaimer.

## About

This project implements a clinically-oriented multi-label classifier for 14 radiological findings in frontal chest X-rays. The model uses a DenseNet121 backbone pretrained on ImageNet with selective layer fine-tuning, trained using BCEWithLogitsLoss with class-weighted positive sampling to handle the severe label imbalance in real clinical data.

Key capabilities:

- 14-label simultaneous pathology prediction (sigmoid multi-label, not softmax)
- Grad-CAM saliency maps localize which image regions drove each prediction
- AUC benchmarking against the original CheXpert paper (Irvin et al. 2019)
- FastAPI REST endpoint for production inference (base64 JSON)
- MLflow experiment tracking with checkpoint management
- Full mock-data test suite — runs in CI without downloading real data

## Architecture

```
Input (320x320 chest X-ray)
         |
 ┌───────▼────────┐
 │  DenseNet121   │   pretrained ImageNet backbone
 │  ─────────────│
 │  DenseBlock 1  │  ← frozen
 │  Transition 1  │  ← frozen
 │  DenseBlock 2  │  ← frozen
 │  Transition 2  │  ← frozen
 │  DenseBlock 3  │  ← fine-tuned
 │  Transition 3  │  ← fine-tuned
 │  DenseBlock 4  │  ← fine-tuned  ← Grad-CAM target
 └───────┬────────┘
         │
 ┌───────▼────────┐
 │ AdaptiveAvgPool│  (1024 features)
 │   Dropout 0.2  │
 │   Linear 512   │
 │     ReLU       │
 │   Dropout 0.1  │
 │   Linear 14    │
 └───────┬────────┘
         │
     Sigmoid → 14 probabilities (one per pathology)
```

## AUC Comparison vs Irvin et al. 2019

| Pathology          | Our AUC | Published AUC | Delta  |
|--------------------|---------|---------------|--------|
| Atelectasis        | *       | 0.858         | —      |
| Cardiomegaly       | *       | 0.831         | —      |
| Consolidation      | *       | 0.937         | —      |
| Edema              | *       | 0.941         | —      |
| Pleural Effusion   | *       | 0.934         | —      |

*Run `python -m src.evaluate` on trained weights to populate.

The comparison table is generated automatically after evaluation and saved to `results/metrics.json`.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/shaikn6/chexpert-pathology-classifier.git
cd chexpert-pathology-classifier

# 2. Install dependencies (Python 3.11+)
pip install -r requirements.txt

# 3. Train on mock data (no CheXpert download needed)
python -m src.train

# 4. Run the full test suite
python -m pytest tests/ -v

# 5. Start the FastAPI inference server
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 6. Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<base64-encoded-jpeg>", "top_k": 3}'
```

### Using Real CheXpert Data

```bash
# Download from Stanford AIMI:
# https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2

# Then train with real data
python -m src.train \
  --train-csv data/CheXpert-v1.0/train.csv \
  --val-csv data/CheXpert-v1.0/valid.csv \
  --data-root data/

# Evaluate and generate comparison table
python -m src.evaluate \
  --checkpoint checkpoints/best_model.pt \
  --val-csv data/CheXpert-v1.0/valid.csv \
  --data-root data/
```

## Dataset

CheXpert (Stanford AIMI, 2019):

- 224,316 frontal + lateral chest X-rays
- 14 pathology labels with uncertainty annotations
- Label policy: U-zeroing (uncertain = 0) for multi-label training
- Class imbalance handled via inverse-frequency pos_weight in BCEWithLogitsLoss

## Training Details

| Hyperparameter     | Value                  |
|--------------------|------------------------|
| Backbone           | DenseNet121 (ImageNet) |
| Optimizer          | AdamW                  |
| Learning rate      | 1e-4 (cosine decay)    |
| Batch size         | 32                     |
| Image size         | 320x320                |
| Loss               | BCEWithLogitsLoss + pos_weight |
| Early stopping     | patience=5 on val macro AUC |
| Frozen layers      | DenseBlocks 1-2 + transitions |
| Fine-tuned layers  | DenseBlocks 3-4 + classifier |

## Project Structure

```
chexpert-pathology-classifier/
├── src/
│   ├── constants.py     # Label names, published AUCs, hyperparameters
│   ├── data.py          # CheXpert dataset, mock dataset, DataLoader factory
│   ├── model.py         # DenseNet121 classifier architecture
│   ├── train.py         # Training loop + MLflow + early stopping
│   ├── evaluate.py      # AUC computation, calibration, comparison table
│   ├── gradcam.py       # Grad-CAM saliency visualization
│   └── api.py           # FastAPI inference endpoint
├── tests/
│   ├── conftest.py      # Shared fixtures (all mock, no real data)
│   ├── test_data.py     # Dataset and DataLoader tests
│   ├── test_model.py    # Model architecture tests
│   ├── test_train.py    # Loss, early stopping, training loop tests
│   ├── test_evaluate.py # AUC, calibration, metrics saving tests
│   ├── test_gradcam.py  # Grad-CAM shape and output tests
│   └── test_api.py      # FastAPI endpoint tests
├── results/
│   ├── metrics.json     # Evaluation metrics (generated)
│   ├── roc_curves.png   # ROC curves (generated)
│   ├── calibration.png  # Calibration curves (generated)
│   └── gradcam/         # Saliency overlay images (generated)
├── requirements.txt
├── LICENSE
└── CHANGELOG.md
```

## API Reference

### POST /predict

```json
Request:
{
  "image_b64": "<base64-encoded JPEG or PNG>",
  "top_k": 3
}

Response:
{
  "all_probabilities": {
    "Atelectasis": 0.142,
    "Cardiomegaly": 0.089,
    ...
  },
  "top_labels": [
    {"label": "Support Devices", "probability": 0.812, "rank": 1},
    {"label": "Pleural Effusion", "probability": 0.634, "rank": 2},
    {"label": "Edema",            "probability": 0.521, "rank": 3}
  ],
  "inference_time_ms": 42.3,
  "model_version": "1.0.0"
}
```

### GET /labels

Returns all 14 CheXpert label names.

### GET /health

Liveness check. Returns `{"status": "ok"}` when model is loaded.

## References

- Irvin et al. (2019). "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." AAAI 2019. [arXiv:1901.07031](https://arxiv.org/abs/1901.07031)
- Huang et al. (2017). "Densely Connected Convolutional Networks." CVPR 2017.
- Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.

## License

MIT — see [LICENSE](LICENSE) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

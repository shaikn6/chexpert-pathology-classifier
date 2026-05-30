# Changelog

All notable changes to this project will be documented here.

## v1.0.0 — 2026-05-30

- DenseNet121 multi-label classifier on CheXpert 14 labels
- Selective fine-tuning: DenseBlocks 3-4 trainable, Blocks 1-2 frozen
- BCEWithLogitsLoss with inverse-frequency class weighting for imbalanced labels
- AdamW optimizer with cosine annealing LR schedule
- Early stopping on validation macro AUC (patience=5)
- MLflow experiment tracking with local mlruns/ storage
- Checkpoint saving (best val AUC model)
- Grad-CAM saliency visualization targeting DenseBlock4
- AUC comparison table vs Irvin et al. 2019 (CheXpert paper)
- Calibration curves and ROC curve plots
- FastAPI inference endpoint with base64 image input
- Batch inference endpoint (up to 16 images)
- Mock dataset generator for CI/testing (no real data download required)
- 50+ tests with full mock-data coverage
- Results saved to results/metrics.json

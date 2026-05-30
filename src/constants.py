"""Global constants for CheXpert pathology classifier."""

CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

NUM_CLASSES = len(CHEXPERT_LABELS)

# Published AUCs from Irvin et al. 2019 (CheXpert paper)
# https://arxiv.org/abs/1901.07031
PUBLISHED_AUCS = {
    "Atelectasis": 0.858,
    "Cardiomegaly": 0.831,
    "Consolidation": 0.937,
    "Edema": 0.941,
    "Pleural Effusion": 0.934,
}

# ImageNet normalization stats (used for pretrained DenseNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 320

# Training hyperparameters
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
EARLY_STOP_PATIENCE = 5

# Approximate positive label rates in CheXpert (for mock data generation)
LABEL_PREVALENCE = {
    "Atelectasis": 0.29,
    "Cardiomegaly": 0.23,
    "Consolidation": 0.12,
    "Edema": 0.29,
    "Enlarged Cardiomediastinum": 0.08,
    "Fracture": 0.05,
    "Lung Lesion": 0.08,
    "Lung Opacity": 0.34,
    "No Finding": 0.16,
    "Pleural Effusion": 0.40,
    "Pleural Other": 0.05,
    "Pneumonia": 0.06,
    "Pneumothorax": 0.06,
    "Support Devices": 0.57,
}

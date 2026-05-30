"""DICOM input pipeline for chest X-ray inference.

Loads DICOM files, applies window/level normalization, and converts to
the normalized PyTorch tensor format expected by the classifier.

DICOM windowing maps raw Hounsfield-like pixel values to a displayable
[0, 255] range using clinical window center and window width settings.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pydicom
import torch
from PIL import Image
from torchvision import transforms

# Default chest window preset (lung window)
DEFAULT_WINDOW_CENTER: float = -600.0
DEFAULT_WINDOW_WIDTH: float = 1500.0

# Target spatial size for the classifier
TARGET_SIZE: int = 320

# Grayscale ImageNet-compatible normalization (single channel)
GRAYSCALE_MEAN: list[float] = [0.485]
GRAYSCALE_STD: list[float] = [0.229]


def load_dicom(path: str) -> pydicom.Dataset:
    """Load a DICOM file from disk.

    Args:
        path: Filesystem path to the .dcm file.

    Returns:
        pydicom Dataset object with all DICOM tags and pixel data.
    """
    return pydicom.dcmread(path)


def apply_windowing(
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """Apply DICOM window/level to normalize pixel values to [0, 255].

    Linear windowing maps pixels below (center - width/2) to 0 and
    above (center + width/2) to 255, with a linear ramp in between.

    Args:
        pixel_array: Raw 2D pixel array from the DICOM dataset.
        window_center: Center of the display window (W/L center).
        window_width: Width of the display window (W/L width).

    Returns:
        uint8 numpy array of shape (H, W) with values in [0, 255].
    """
    img = pixel_array.astype(np.float64)
    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0
    img = np.clip(img, lower, upper)
    # Linearly scale to [0, 255]
    if upper > lower:
        img = (img - lower) / (upper - lower) * 255.0
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


def dicom_to_tensor(dcm: pydicom.Dataset) -> torch.Tensor:
    """Convert a DICOM dataset to a normalized PyTorch tensor.

    Pipeline:
      1. Extract the raw pixel array.
      2. Apply windowing using WindowCenter/WindowWidth DICOM tags
         (falls back to WC=-600, WW=1500 lung window if tags are absent).
      3. Resize to 320x320 using bilinear interpolation.
      4. Normalize with grayscale ImageNet stats (mean=0.485, std=0.229).

    Args:
        dcm: pydicom Dataset loaded from a .dcm file.

    Returns:
        Float tensor of shape (1, 320, 320) ready for model inference.
    """
    pixel_array = dcm.pixel_array.astype(np.float32)

    # Read window/level tags; fall back to lung window preset
    try:
        wc = float(dcm.WindowCenter)
        ww = float(dcm.WindowWidth)
    except AttributeError:
        wc = DEFAULT_WINDOW_CENTER
        ww = DEFAULT_WINDOW_WIDTH

    windowed = apply_windowing(pixel_array, wc, ww)

    # Convert to PIL Image (grayscale)
    pil_img = Image.fromarray(windowed, mode="L")

    transform = transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),           # → (1, H, W) in [0, 1]
        transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
    ])

    tensor = transform(pil_img)  # (1, 320, 320)
    return tensor


def create_synthetic_dicom() -> pydicom.Dataset:
    """Create an in-memory DICOM dataset with synthetic chest X-ray pixel data.

    Generates a 256x256 grayscale image with random uint16 pixels and sets
    all mandatory DICOM tags required for a valid pixel-data dataset. The
    WindowCenter and WindowWidth tags are set to the default lung window so
    that dicom_to_tensor can process it without falling back.

    Compatible with pydicom >= 2.4 and pydicom 3.x (which requires
    TransferSyntaxUID in file_meta for pixel_array access).

    Returns:
        pydicom Dataset suitable for passing to dicom_to_tensor().
    """
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    rows, cols = 256, 256
    rng = np.random.default_rng(seed=0)
    pixels = rng.integers(0, 4096, size=(rows, cols), dtype=np.uint16)

    ds = pydicom.Dataset()
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    # file_meta with TransferSyntaxUID is required by pydicom 3.x for
    # pixel_array to work without writing to disk first.
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()

    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = generate_uid()

    # Required pixel-data tags
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0          # unsigned
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Pack uint16 array into bytes
    ds.PixelData = pixels.tobytes()

    # Window/level tags (lung window)
    ds.WindowCenter = str(DEFAULT_WINDOW_CENTER)
    ds.WindowWidth = str(DEFAULT_WINDOW_WIDTH)

    return ds

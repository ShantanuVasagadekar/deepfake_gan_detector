"""
utils.py — Shared utilities for the Deepfake GAN Detector project.

Provides:
  - Device selection (CUDA / CPU)
  - Weight initialization for GAN layers
  - Image tensor ↔ display helpers
  - Directory creation
  - Logging configuration
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str = "training.log", level=logging.INFO):
    """Configure root logger to write to both console and *log_file*."""
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(level)
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return root


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device (CUDA → CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] CUDA not available — using CPU")
    return device


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def weights_init(m):
    """Custom weight initialization for Conv / BatchNorm layers (DCGAN paper)."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

# Standard normalization used throughout the project (ImageNet stats)
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor (0-1 range) to a PIL Image."""
    tensor = tensor.detach().cpu().clamp(0, 1)
    np_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def pil_to_tensor(img: Image.Image, size: int = 128) -> torch.Tensor:
    """Resize a PIL Image and convert to a normalised CHW tensor."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    return transform(img)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dir(*paths: str):
    """Create directories if they do not already exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, filepath: str):
    """Save a training checkpoint dictionary to *filepath*."""
    ensure_dir(os.path.dirname(filepath))
    torch.save(state, filepath)
    print(f"[CHECKPOINT] Saved → {filepath}")


def load_checkpoint(filepath: str, device: torch.device) -> dict:
    """Load a checkpoint from *filepath* onto *device*."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    print(f"[CHECKPOINT] Loaded ← {filepath}")
    return ckpt

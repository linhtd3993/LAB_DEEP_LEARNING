"""Central Configuration & Hardware Environment Manager.

Quan ly tap trung toan bo hang so, duong dan thu muc va thong so thich ung phan cung.
"""

import os
from pathlib import Path
from typing import List
import torch

# 1. ImageNet Pre-trained Normalization Constants
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# 2. Dataset Constants (CIFAR-10)
CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
NUM_CLASSES: int = 10
INPUT_IMAGE_SIZE: int = 224
DEFAULT_SEED: int = 42

# 3. Directory Layout (Tu dong xac dinh dua tren vi tri file config)
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"


def ensure_directories() -> None:
    """Khoi tao cac thu muc lam viec can thiet neu chua ton tai."""
    for directory in [DATA_DIR, CHECKPOINT_DIR, RUNS_DIR, FIGURES_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# 4. Hardware-Tailored Device Detection & Hyperparameter Defaults
def get_device() -> torch.device:
    """Tu dong phat hien va tra ve thiet bi tinh toan toi uu nhat."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Gia tri mac dinh duoc toi uu hoa theo phan cung RTX 3050 (6GB VRAM) va CPU 16 threads
IS_CUDA_AVAILABLE: bool = torch.cuda.is_available()
DEFAULT_BATCH_SIZE: int = 64 if IS_CUDA_AVAILABLE else 32
DEFAULT_NUM_WORKERS: int = 4 if os.name != "nt" else 0
DEFAULT_PIN_MEMORY: bool = IS_CUDA_AVAILABLE
DEFAULT_USE_AMP: bool = IS_CUDA_AVAILABLE
DEFAULT_EPOCHS: int = 15
DEFAULT_LR: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_OPTIMIZER: str = "adamw"


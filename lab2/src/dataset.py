"""Leak-Free CIFAR-10 Data Pipeline for Transfer Learning.

Pipeline xu ly du lieu tieu chuan, dam bao phan tach doc lap tap Train, Validation va Test,
hoan toan ngan chan ro ri phep Data Augmentation sang tap danh gia.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
    DEFAULT_SEED,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_IMAGE_SIZE,
)


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Khoi tao phep bien doi hinh anh cho tap Huan luyen va tap Danh gia.

    Returns:
        Tuple gom (train_transform, eval_transform).
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def get_cifar10_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
    subset_ratio: float = 1.0,
    dry_run: bool = False,
    seed: int = DEFAULT_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tai tap du lieu CIFAR-10 va phan chia thanh Train (40k), Val (10k), Test (10k).

    Co che chong ro ri du lieu (Leak-free Protocol):
    Khoi tao 2 thuc the CIFAR10 rieng biet cho tap train=True voi 2 phep transform khac nhau.
    Sau do chia tap bang chi so hoan vi ngau nhien co dinh thong qua torch.utils.data.Subset.

    Args:
        data_dir: Thu muc luu tru du lieu (mac dinh lay tu DATA_DIR).
        batch_size: Kich thuoc mini-batch.
        num_workers: So luong luong doc du lieu song song.
        pin_memory: Khoa trang bo nho may chu de tang toc nap len GPU.
        subset_ratio: Ty le mau duoc su dung (0.0 < subset_ratio <= 1.0).
        dry_run: Neu True, lay 128 mau train, 64 mau val va 64 mau test de kiem thu nhanh.
        seed: Hat giong ngau nhien dam bao tinh tai lap.

    Returns:
        Tuple gom (train_loader, val_loader, test_loader).
    """
    target_data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    target_data_dir.mkdir(parents=True, exist_ok=True)

    train_transform, eval_transform = get_transforms()

    # Khoi tao 2 doi tuong Dataset doc lap de gan transform rieng biet
    train_dataset_raw = datasets.CIFAR10(
        root=str(target_data_dir),
        train=True,
        transform=train_transform,
        download=True,
    )
    val_dataset_raw = datasets.CIFAR10(
        root=str(target_data_dir),
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_dataset_raw = datasets.CIFAR10(
        root=str(target_data_dir),
        train=False,
        transform=eval_transform,
        download=True,
    )

    # Sinh danh sach chi so ngau nhien co dinh bang Generator
    total_train_samples = len(train_dataset_raw)  # 50,000
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total_train_samples, generator=generator).tolist()

    # 40,000 mau Train va 10,000 mau Validation
    train_indices = shuffled_indices[:40000]
    val_indices = shuffled_indices[40000:]
    test_indices = list(range(len(test_dataset_raw)))  # 10,000

    # Xu ly che do kiem thu nhanh hoac cat tap con
    if dry_run:
        train_indices = train_indices[:128]
        val_indices = val_indices[:64]
        test_indices = test_indices[:64]
    elif 0.0 < subset_ratio < 1.0:
        n_train = max(1, int(len(train_indices) * subset_ratio))
        n_val = max(1, int(len(val_indices) * subset_ratio))
        n_test = max(1, int(len(test_indices) * subset_ratio))
        train_indices = train_indices[:n_train]
        val_indices = val_indices[:n_val]
        test_indices = test_indices[:n_test]

    # Bọc thanh cac doi tuong Subset doc lap
    train_subset = Subset(train_dataset_raw, train_indices)
    val_subset = Subset(val_dataset_raw, val_indices)
    test_subset = Subset(test_dataset_raw, test_indices)

    # Tao cac DataLoader tuong ung
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


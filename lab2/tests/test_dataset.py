"""Unit tests for Milestone 1: Config & Data Pipeline."""

import unittest
import torch
from torchvision import transforms

from src.config import (
    CIFAR10_CLASSES,
    DEFAULT_BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    ensure_directories,
    get_device,
)
from src.dataset import get_cifar10_dataloaders, get_transforms


class TestConfigAndDataPipeline(unittest.TestCase):
    """Test suite cho config.py va dataset.py."""

    def test_config_constants_and_directories(self) -> None:
        """Kiem tra hang so ImageNet, so lop CIFAR-10 va tao thu muc."""
        self.assertEqual(len(CIFAR10_CLASSES), 10)
        self.assertEqual(NUM_CLASSES, 10)
        self.assertEqual(len(IMAGENET_MEAN), 3)
        self.assertEqual(len(IMAGENET_STD), 3)

        ensure_directories()
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_transforms_properties(self) -> None:
        """Kiem tra phep transform cua Train phai co Augmentation, Val phai la phep bien doi tinh."""
        train_tf, eval_tf = get_transforms()

        # Train transform phai co RandomCrop va RandomHorizontalFlip
        train_types = [type(t) for t in train_tf.transforms]
        self.assertIn(transforms.RandomCrop, train_types)
        self.assertIn(transforms.RandomHorizontalFlip, train_types)

        # Eval transform khong duoc chua bat ky phep bien doi ngau nhien nao
        eval_types = [type(t) for t in eval_tf.transforms]
        self.assertNotIn(transforms.RandomCrop, eval_types)
        self.assertNotIn(transforms.RandomHorizontalFlip, eval_types)

    def test_dataloaders_dry_run_and_shapes(self) -> None:
        """Kiem tra khoi tao DataLoader o che do dry_run, tensor shape va nhan target."""
        batch_size = 16
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(
            batch_size=batch_size,
            num_workers=0,  # Dung 0 workers cho unit test de tranh da tien trinh
            dry_run=True,
            seed=42,
        )

        # Kiem tra kich thuoc mau dry_run
        self.assertEqual(len(train_loader.dataset), 128)
        self.assertEqual(len(val_loader.dataset), 64)
        self.assertEqual(len(test_loader.dataset), 64)

        # Kiem tra tinh roi nhau (Disjointness) giua train va val
        train_indices = train_loader.dataset.indices
        val_indices = val_loader.dataset.indices
        self.assertTrue(set(train_indices).isdisjoint(set(val_indices)))

        # Kiem tra Tensor shapes tren batch dau tien cua tap train
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, torch.Size([batch_size, 3, 224, 224]))
        self.assertEqual(labels.shape, torch.Size([batch_size]))
        self.assertTrue(torch.all((labels >= 0) & (labels <= 9)))


if __name__ == "__main__":
    unittest.main()


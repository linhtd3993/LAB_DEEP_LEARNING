"""Unit tests for Milestone 4: Evaluator and Visualization Engine."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CIFAR10_CLASSES
from src.evaluate import (
    evaluate_test_set,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_training_history,
)


class TestEvaluatorAndVisualizer(unittest.TestCase):
    """Test suite cho evaluate.py."""

    @classmethod
    def setUpClass(cls) -> None:
        """Khoi tao du lieu gia lap va thu muc tam."""
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.tmp_dir = tempfile.mkdtemp()

        # Dummy model
        cls.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 10),
        ).to(cls.device)

        # Dummy test data: 32 mau
        images = torch.randn(32, 3, 224, 224)
        labels = torch.randint(0, 10, (32,))
        dataset = TensorDataset(images, labels)
        cls.test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        cls.criterion = nn.CrossEntropyLoss()

    @classmethod
    def tearDownClass(cls) -> None:
        """Don dep thu muc tam sau khi kiem thu."""
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_evaluate_test_set_metrics(self) -> None:
        """Kiem tra ham evaluate_test_set tra ve dung cau truc va chi so hop le."""
        results = evaluate_test_set(
            model=self.model,
            test_loader=self.test_loader,
            criterion=self.criterion,
            device=self.device,
            class_names=CIFAR10_CLASSES,
        )

        # Kiem tra day du cac truong du lieu
        self.assertIn("test_loss", results)
        self.assertIn("overall_acc", results)
        self.assertIn("per_class_acc", results)
        self.assertIn("confusion_matrix", results)
        self.assertIn("classification_report_str", results)
        self.assertIn("classification_report_dict", results)

        # Kiem tra gia tri hop le
        self.assertGreaterEqual(results["test_loss"], 0.0)
        self.assertGreaterEqual(results["overall_acc"], 0.0)
        self.assertLessEqual(results["overall_acc"], 100.0)

        # Kiem tra ma tran nham lan
        cm = results["confusion_matrix"]
        self.assertEqual(cm.shape, (10, 10))
        self.assertEqual(cm.sum(), 32)

        # Kiem tra per-class accuracy co du 10 lop
        self.assertEqual(len(results["per_class_acc"]), 10)
        for class_name in CIFAR10_CLASSES:
            self.assertIn(class_name, results["per_class_acc"])
            self.assertGreaterEqual(results["per_class_acc"][class_name], 0.0)
            self.assertLessEqual(results["per_class_acc"][class_name], 100.0)

    def test_plot_generation_and_file_creation(self) -> None:
        """Kiem tra 3 ham ve bieu do xuat ra file anh .png chat luong cao."""
        tmp_path = Path(self.tmp_dir)

        # 1. Test plot_training_history
        dummy_history = {
            "train_loss": [2.3, 1.8, 1.2],
            "val_loss": [2.2, 1.7, 1.3],
            "train_acc": [10.0, 45.0, 70.0],
            "val_acc": [12.0, 42.0, 68.0],
        }
        history_path = str(tmp_path / "test_history.png")
        plot_training_history(dummy_history, history_path)
        self.assertTrue(Path(history_path).exists())
        self.assertGreater(Path(history_path).stat().st_size, 1000)

        # 2. Test plot_per_class_accuracy
        dummy_per_class = {c: float(50 + i * 4) for i, c in enumerate(CIFAR10_CLASSES)}
        class_acc_path = str(tmp_path / "test_class_acc.png")
        plot_per_class_accuracy(dummy_per_class, class_acc_path)
        self.assertTrue(Path(class_acc_path).exists())
        self.assertGreater(Path(class_acc_path).stat().st_size, 1000)

        # 3. Test plot_confusion_matrix
        dummy_cm = np.eye(10, dtype=int) * 3
        cm_path = str(tmp_path / "test_cm.png")
        plot_confusion_matrix(dummy_cm, CIFAR10_CLASSES, cm_path)
        self.assertTrue(Path(cm_path).exists())
        self.assertGreater(Path(cm_path).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

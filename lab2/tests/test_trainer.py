"""Unit tests for Milestone 3: Training Engine & TensorBoard Logger."""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.trainer import Trainer


class TestTrainer(unittest.TestCase):
    """Test suite cho trainer.py."""

    @classmethod
    def setUpClass(cls) -> None:
        """Tao du lieu gia lap va mo hinh don gian de test nhanh."""
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.tmp_dir = tempfile.mkdtemp()
        cls.ckpt_dir = str(Path(cls.tmp_dir) / "checkpoints")
        cls.log_dir = str(Path(cls.tmp_dir) / "runs")

        # Mo hinh tuyen tinh don gian: input (3, 224, 224) -> 10 lop
        cls.simple_model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 10),
        )

        # Du lieu gia lap: 32 mau ngau nhien
        images = torch.randn(32, 3, 224, 224)
        labels = torch.randint(0, 10, (32,))
        dataset = TensorDataset(images, labels)
        cls.train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        cls.val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    @classmethod
    def tearDownClass(cls) -> None:
        """Don dep thu muc tam sau khi chay xong tat ca tests."""
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _make_trainer(self, model: nn.Module, epochs: int = 1) -> Trainer:
        """Ham tien ich: Tao Trainer voi cau hinh test nhanh."""
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        )
        criterion = nn.CrossEntropyLoss()
        return Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            epochs=epochs,
            checkpoint_dir=self.ckpt_dir,
            log_dir=self.log_dir,
            experiment_name="test_exp",
            use_amp=False,  # Tat AMP cho unit test de dam bao deterministic
        )

    def test_initial_loss_sanity_check(self) -> None:
        """Kiem tra Cross-Entropy Loss khoi tao xap xi -ln(1/10) ~ 2.3025."""
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 10),
        )
        trainer = self._make_trainer(model, epochs=1)
        val_loss, _ = trainer.validate(epoch=0)

        # Cho phep sai so rong [1.8, 3.2] do model nho va du lieu ngau nhien
        expected_loss = -math.log(1 / 10)  # 2.3025
        self.assertGreater(val_loss, 1.8, f"Loss khoi tao = {val_loss:.4f} qua thap")
        self.assertLess(val_loss, 3.2, f"Loss khoi tao = {val_loss:.4f} qua cao")

    def test_single_batch_overfit(self) -> None:
        """Huan luyen tren duy nhat 1 batch trong nhieu buoc de xac minh gradient toi uu dung."""
        # Mo hinh du lon de overfit 1 batch: AdaptiveAvgPool2d(4) -> 3*16=48 features -> 10
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        # Lay 1 batch co dinh
        inputs, targets = next(iter(self.train_loader))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Ghi nhan loss dau tien
        model.train()
        with torch.no_grad():
            initial_output = model(inputs)
            initial_loss = criterion(initial_output, targets).item()

        # Cap nhat 50 buoc tren chinh batch do
        for _ in range(50):
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        self.assertLess(
            final_loss,
            initial_loss * 0.5,
            f"Loss khong giam du sau 50 buoc: {initial_loss:.4f} -> {final_loss:.4f}",
        )

    def test_fit_creates_checkpoint_and_tensorboard(self) -> None:
        """Kiem tra ham fit() tao file checkpoint va thu muc TensorBoard log."""
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 10),
        )
        trainer = self._make_trainer(model, epochs=2)
        history = trainer.fit()

        # Kiem tra lich su co du 2 epoch
        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)
        self.assertEqual(len(history["train_acc"]), 2)
        self.assertEqual(len(history["val_acc"]), 2)
        self.assertEqual(len(history["epoch_time"]), 2)
        self.assertEqual(len(history["lr"]), 2)

        # Kiem tra checkpoint tot nhat da duoc luu
        best_ckpt = Path(self.ckpt_dir) / "best_test_exp.pth"
        self.assertTrue(best_ckpt.exists(), f"Checkpoint khong ton tai: {best_ckpt}")
        self.assertGreater(
            best_ckpt.stat().st_size, 0, "Checkpoint co dung luong 0 byte"
        )

        # Kiem tra noi dung checkpoint
        ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", ckpt_data)
        self.assertIn("optimizer_state_dict", ckpt_data)
        self.assertIn("val_acc", ckpt_data)
        self.assertIn("epoch", ckpt_data)

        # Kiem tra thu muc TensorBoard log ton tai va co events file
        self.assertTrue(
            Path(trainer.tb_dir).exists(),
            f"Thu muc TensorBoard log khong ton tai: {trainer.tb_dir}",
        )
        tb_files = list(Path(trainer.tb_dir).glob("events.out.tfevents.*"))
        self.assertGreater(
            len(tb_files),
            0,
            "Khong tim thay file events TensorBoard",
        )


if __name__ == "__main__":
    unittest.main()

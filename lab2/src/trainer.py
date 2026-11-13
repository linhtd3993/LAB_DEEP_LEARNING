"""Training Engine with TensorBoard Logging and Model Checkpointing.

Dong co huan luyen va kiem dinh mo hinh, tich hop Automatic Mixed Precision (AMP),
ghi nhat ky thoi gian thuc vao TensorBoard va tu dong luu checkpoint tot nhat
dua tren Validation Accuracy.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import CHECKPOINT_DIR, RUNS_DIR, ensure_directories


class Trainer:
    """Lop quan ly dong co huan luyen, kiem dinh va ghi log TensorBoard."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epochs: int = 15,
        checkpoint_dir: str = "",
        log_dir: str = "",
        experiment_name: str = "exp",
        use_amp: bool = True,
    ) -> None:
        """Khoi tao Trainer voi day du cac thanh phan can thiet.

        Args:
            model: Mo hinh da duoc bien doi va thiet lap requires_grad.
            train_loader: DataLoader cho tap huan luyen.
            val_loader: DataLoader cho tap kiem dinh.
            optimizer: Optimizer chi chua cac tham so co requires_grad=True.
            criterion: Ham mat mat (CrossEntropyLoss).
            device: Thiet bi tinh toan (cuda / cpu).
            epochs: So luong chu ky huan luyen.
            checkpoint_dir: Thu muc luu checkpoint (mac dinh lay tu config).
            log_dir: Thu muc luu log TensorBoard (mac dinh lay tu config).
            experiment_name: Ten dinh danh thuc nghiem.
            use_amp: Kich hoat Automatic Mixed Precision (FP16) tren GPU.
        """
        ensure_directories()

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.use_amp = use_amp and (device.type == "cuda")
        self.experiment_name = experiment_name

        # Thu muc luu tru
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer voi thu muc rieng theo timestamp va ten thuc nghiem
        tb_log_dir = Path(log_dir) if log_dir else RUNS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tb_dir = tb_log_dir / f"{timestamp}_{experiment_name}"
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        # AMP GradScaler de tang toc tinh toan tren GPU (FP16)
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Theo doi trang thai tot nhat
        self.best_val_acc: float = -1.0
        self.best_checkpoint_path: str = ""

        # Lich su huan luyen de phuc vu ve bieu do sau nay
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "epoch_time": [],
            "lr": [],
        }

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Thuc thi huan luyen trong 1 epoch.

        Args:
            epoch: Chi so epoch hien tai (0-indexed).

        Returns:
            Tuple gom (train_loss_trung_binh, train_accuracy_phan_tram).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs} [Train]",
            leave=False,
        )

        for inputs, targets in progress_bar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass voi AMP neu duoc kich hoat
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Thong ke tich luy
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

            # Cap nhat thanh tien trinh
            current_acc = 100.0 * correct / total
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%"
            )

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate(self, epoch: int) -> Tuple[float, float]:
        """Thuc thi kiem dinh mo hinh tren tap validation.

        Args:
            epoch: Chi so epoch hien tai (0-indexed).

        Returns:
            Tuple gom (val_loss_trung_binh, val_accuracy_phan_tram).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs} [Val]",
            leave=False,
        )

        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                total += batch_size
                correct += predicted.eq(targets).sum().item()

                current_acc = 100.0 * correct / total
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%"
                )

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def save_checkpoint(
        self, epoch: int, val_acc: float, is_best: bool = False
    ) -> str:
        """Luu checkpoint trong so mo hinh xuong o dia.

        Args:
            epoch: Epoch hien tai.
            val_acc: Do chinh xac validation dat duoc.
            is_best: Co danh dau checkpoint co val_acc cao nhat.

        Returns:
            Duong dan den file checkpoint da luu.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "experiment_name": self.experiment_name,
        }

        if is_best:
            filename = f"best_{self.experiment_name}.pth"
        else:
            filename = f"epoch{epoch + 1}_{self.experiment_name}.pth"

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        return str(filepath)

    def fit(self) -> Dict[str, List[float]]:
        """Chay toan bo qua trinh huan luyen va kiem dinh qua tat ca cac epoch.

        Returns:
            Dictionary luu lich su gom 'train_loss', 'val_loss',
            'train_acc', 'val_acc', 'epoch_time', 'lr'.
        """
        print("=" * 70)
        print(f"BAT DAU HUAN LUYEN: {self.experiment_name}")
        print(f"  Thiet bi       : {self.device}")
        print(f"  So epoch       : {self.epochs}")
        print(f"  Mixed Precision: {'ON' if self.use_amp else 'OFF'}")
        print(f"  TensorBoard    : {self.tb_dir}")
        print("=" * 70)

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Huan luyen 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Kiem dinh tren tap validation
            val_loss, val_acc = self.validate(epoch)

            epoch_time = time.time() - epoch_start

            # Lay learning rate hien tai tu optimizer
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Ghi nhan lich su
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_time"].append(epoch_time)
            self.history["lr"].append(current_lr)

            # Ghi log vao TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch + 1)
            self.writer.add_scalar("Loss/val", val_loss, epoch + 1)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch + 1)
            self.writer.add_scalar("Learning_Rate", current_lr, epoch + 1)

            # Luu checkpoint tot nhat dua tren Validation Accuracy
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_checkpoint_path = self.save_checkpoint(
                    epoch, val_acc, is_best=True
                )

            # In ket qua epoch ra console
            marker = " [BEST]" if is_best else ""
            print(
                f"Epoch [{epoch + 1:3d}/{self.epochs}] "
                f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%{marker} "
                f"| Time: {epoch_time:.1f}s | LR: {current_lr:.2e}"
            )

        # Dong TensorBoard writer
        self.writer.close()

        # Don bo nho GPU cache sau khi huan luyen xong
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        print("-" * 70)
        print(f"HOAN TAT HUAN LUYEN: {self.experiment_name}")
        print(f"  Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"  Checkpoint luu tai       : {self.best_checkpoint_path}")
        print("-" * 70)

        return self.history

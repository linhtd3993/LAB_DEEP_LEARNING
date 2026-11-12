"""Evaluator and Visualization Engine.

Thuc hien danh gia mo hinh tren tap Test doc lap, tinh toan cac chi so chi tiet
(Overall Accuracy, Per-class Accuracy, Confusion Matrix, Classification Report)
va ve 3 bieu do chuan xuat ban 300 DPI phuc vu nghien cuu va bao cao.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, Any]:
    """Danh gia toan dien mo hinh tren tap Test doc lap.

    Args:
        model: Mo hinh PyTorch da duoc nap trong so tot nhat.
        test_loader: DataLoader cho tap Test.
        criterion: Ham mat mat (CrossEntropyLoss).
        device: Thiet bi tinh toan (cuda / cpu).
        class_names: Danh sach ten cac lop (10 lop CIFAR-10).

    Returns:
        Dictionary chua cac thong so:
        - 'test_loss': Mat mat trung binh tren tap Test.
        - 'overall_acc': Do chinh xac tong the Top-1 (%).
        - 'per_class_acc': Dictionary do chinh xac theo tung lop (%).
        - 'confusion_matrix': Ma tran nham lan dang numpy array.
        - 'classification_report_str': Chuoi bao cao classification dang bang.
        - 'classification_report_dict': Dictionary chua precision, recall, f1-score.
    """
    model.eval()
    running_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(preds.cpu().numpy().tolist())

    total_samples = len(test_loader.dataset)
    test_loss = running_loss / total_samples

    targets_np = np.array(all_targets)
    preds_np = np.array(all_predictions)

    # Do chinh xac tong the (Top-1 Accuracy)
    overall_acc = 100.0 * np.mean(targets_np == preds_np)

    # Tinh Confusion Matrix
    cm = confusion_matrix(targets_np, preds_np, labels=list(range(len(class_names))))

    # Tinh Do chinh xac cho tung lop rieng biet (Per-class Accuracy)
    per_class_acc: Dict[str, float] = {}
    for idx, class_name in enumerate(class_names):
        row_sum = cm[idx].sum()
        if row_sum > 0:
            acc = 100.0 * (cm[idx, idx] / row_sum)
        else:
            acc = 0.0
        per_class_acc[class_name] = float(acc)

    # Tao Classification Report chi tiet (Precision, Recall, F1)
    report_str = classification_report(
        targets_np,
        preds_np,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        targets_np,
        preds_np,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # In bao cao truc quan ra man hinh console
    print("=" * 70)
    print("KET QUA DANH GIA TREN TAP TEST DOC LAP (10,000 ANH CIFAR-10)")
    print("=" * 70)
    print(f"  Test Loss            : {test_loss:.4f}")
    print(f"  Top-1 Overall Accuracy: {overall_acc:.2f}%\n")
    print("Per-Class Accuracy Breakdown:")
    print("-" * 70)
    for class_name, acc in per_class_acc.items():
        print(f"  Class {class_name:<12}: {acc:6.2f}%")
    print("-" * 70)
    print("\nDetailed Classification Report:")
    print(report_str)
    print("=" * 70)

    return {
        "test_loss": float(test_loss),
        "overall_acc": float(overall_acc),
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
        "classification_report_str": report_str,
        "classification_report_dict": report_dict,
    }


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str,
    title_suffix: str = "",
) -> None:
    """Ve bieu do kep the hien duong cong Loss va Accuracy giua Train va Val qua cac epoch.

    Do phan giai 300 DPI, phu hop chen vao bai bao khoa hoc va bao cao thuc hanh.

    Args:
        history: Dictionary lich su tra ve boi Trainer.fit().
        output_path: Duong dan file anh dich (.png).
        title_suffix: Hau to bo sung cho tieu de (vi du ten model va strategy).
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Cross-Entropy Loss
    ax1.plot(
        epochs_range,
        history["train_loss"],
        marker="o",
        linewidth=2,
        label="Train Loss",
        color="#1f77b4",
    )
    ax1.plot(
        epochs_range,
        history["val_loss"],
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Val Loss",
        color="#ff7f0e",
    )
    ax1.set_title(f"Training & Validation Loss {title_suffix}".strip(), fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2: Classification Accuracy
    ax2.plot(
        epochs_range,
        history["train_acc"],
        marker="o",
        linewidth=2,
        label="Train Accuracy",
        color="#2ca02c",
    )
    ax2.plot(
        epochs_range,
        history["val_acc"],
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Val Accuracy",
        color="#d62728",
    )
    ax2.set_title(
        f"Training & Validation Accuracy {title_suffix}".strip(), fontsize=12
    )
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_accuracy(
    per_class_acc: Dict[str, float],
    output_path: str,
    title_suffix: str = "",
) -> None:
    """Ve bieu do cot the hien do chinh xac tung lop tren tap Test kem duong trung binh (300 DPI).

    Args:
        per_class_acc: Dictionary ten lop -> do chinh xac (%).
        output_path: Duong dan file anh dich (.png).
        title_suffix: Hau to bo sung cho tieu de.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    classes = list(per_class_acc.keys())
    accuracies = list(per_class_acc.values())
    avg_accuracy = float(np.mean(accuracies))

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        classes,
        accuracies,
        color="#3498db",
        edgecolor="#2980b9",
        width=0.6,
        alpha=0.85,
    )

    # Duong nguong trung binh
    ax.axhline(
        y=avg_accuracy,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Mean Accuracy: {avg_accuracy:.2f}%",
    )

    # Ghi chu ty le phan tram tren dinh tung cot
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_title(f"Per-Class Test Accuracy {title_suffix}".strip(), fontsize=13)
    ax.set_xlabel("CIFAR-10 Class", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title_suffix: str = "",
) -> None:
    """Ve heatmap ma tran nham lan duoc chuan hoa theo hang tren tap Test (300 DPI).

    Args:
        cm: Ma tran nham lan nguyen ban (chua chuan hoa).
        class_names: Danh sach ten cac lop.
        output_path: Duong dan file anh dich (.png).
        title_suffix: Hau to bo sung cho tieu de.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Chuan hoa theo hang (Recall): ty le phan tram mau thuc te duoc du doan vao tung lop
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(
        cm.astype("float"),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Rate (Recall)"},
        ax=ax,
    )

    ax.set_title(
        f"Normalized Confusion Matrix (Test Set) {title_suffix}".strip(),
        fontsize=13,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

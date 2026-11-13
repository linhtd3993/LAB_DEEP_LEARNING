"""CLI Interface and Pipeline Orchestrator.

Diem khoi chay duy nhat cua ung dung (CLI Entry Point), dieu phoi toan bo pipeline:
cau hinh sieu tham so -> nap du lieu CIFAR-10 -> khoi tao mo hinh pre-trained ->
huan luyen va ghi log TensorBoard -> danh gia tap Test doc lap -> xuat bao cao va bieu do 300 DPI.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    CIFAR10_CLASSES,
    CHECKPOINT_DIR,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OPTIMIZER,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    FIGURES_DIR,
    NUM_CLASSES,
    REPORTS_DIR,
    RUNS_DIR,
    ensure_directories,
    get_device,
)
from src.dataset import get_cifar10_dataloaders
from src.evaluate import (
    evaluate_test_set,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_training_history,
)
from src.model_factory import build_model, count_parameters, inspect_model
from src.trainer import Trainer


def set_seed(seed: int = 42) -> None:
    """Thiet lap random seed co dinh de dam bao tinh tai lap tuyet doi cua thuc nghiem."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments() -> argparse.Namespace:
    """Phan tich cac tham so dong lenh (CLI arguments)."""
    parser = argparse.ArgumentParser(
        description="PyTorch Pre-trained CNN Benchmark & Fine-Tuning Pipeline on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 1. Mo hinh va Chien luoc
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50", "vgg16_bn", "densenet121"],
        help="Kien truc mang CNN pre-trained can thuc nghiem",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="freeze_all",
        choices=["freeze_all", "unfreeze_last_block"],
        help="Chien luoc dong bang / mo khoa trong so",
    )

    # 2. Sieu tham so huan luyen
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="So luong epoch huan luyen",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Kich thuoc mini-batch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Toc do hoc (learning rate)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=DEFAULT_OPTIMIZER,
        choices=["adamw", "sgd"],
        help="Thuat toan toi uu hoa",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="He so suy giam trong so L2 regularization",
    )

    # 3. Pipeline du lieu va phan cung
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="So luong tien trinh con nạp du lieu (DataLoader workers)",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=1.0,
        help="Ty le phan tram tap du lieu su dung (0.0 < ratio <= 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Che do kiem thu nhanh: chi lay 100 mau cho moi tap du lieu",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Vo hieu hoa Automatic Mixed Precision (AMP) tren GPU",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Gia tri hat giong ngau nhien (random seed)",
    )

    # 4. Che do dac biet
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Chi in cau truc va thong ke tham so mo hinh roi thoat, khong huan luyen",
    )

    # 5. Duong dan tuy chon
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Thu muc chua tap du lieu CIFAR-10",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Thu muc luu checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Thu muc luu TensorBoard logs",
    )

    return parser.parse_args()


def main() -> int:
    """Ham dieu phoi chinh cua toan bo chuong trinh."""
    args = parse_arguments()

    # Khoi tao cac thu muc can thiet
    ensure_directories()
    set_seed(args.seed)

    # Xac dinh thiet bi tinh toan
    device = get_device()
    print("=" * 70)
    print("KHOI CHAY HE THONG BENCHMARK PRE-TRAINED CNN")
    print(f"  Thiet bi su dung     : {device}")
    print(f"  Mo hinh              : {args.model}")
    print(f"  Chien luoc           : {args.strategy}")
    print(f"  Seed                 : {args.seed}")
    print("=" * 70)

    # Khoi tao mo hinh pre-trained
    model = build_model(
        model_name=args.model,
        strategy=args.strategy,
        num_classes=NUM_CLASSES,
    )

    # Che do chi inspect tham so
    if args.inspect_only:
        inspect_model(model, args.model)
        return 0

    # In thong tin tom tat tham so
    inspect_model(model, args.model)

    # Xac dinh duong dan
    data_dir = args.data_dir if args.data_dir else str(DATA_DIR)
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else str(CHECKPOINT_DIR)
    log_dir = args.log_dir if args.log_dir else str(RUNS_DIR)

    # Nap du lieu CIFAR-10
    print("\nDang nap tap du lieu CIFAR-10...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_ratio=args.subset_ratio,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    print(
        f"  Train samples: {len(train_loader.dataset)} | "
        f"Val samples: {len(val_loader.dataset)} | "
        f"Test samples: {len(test_loader.dataset)}"
    )

    # Loc chi nhung tham so can cap nhat (requires_grad = True) dua vao Optimizer
    trainable_parameters = list(
        filter(lambda p: p.requires_grad, model.parameters())
    )

    if args.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            trainable_parameters,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer khong ho tro: {args.optimizer}")

    criterion = nn.CrossEntropyLoss()

    # Dinh danh thuc nghiem chuan hoa
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.model}_{args.strategy}_lr{args.lr}_bs{args.batch_size}"

    # Khoi tao Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_amp=(not args.no_amp),
    )

    # Huan luyen mo hinh
    history = trainer.fit()

    # Nap checkpoint tot nhat de danh gia tren tap Test
    print("\nDang nap checkpoint tot nhat de danh gia tren tap Test...")
    best_checkpoint_path = trainer.best_checkpoint_path
    if Path(best_checkpoint_path).exists():
        checkpoint_data = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        print(f"  Da nap thanh cong trong so tu: {best_checkpoint_path}")
    else:
        print("  Canh bao: Khong tim thay checkpoint, su dung trong so hien tai.")

    # Danh gia toan dien tren tap Test doc lap
    eval_results = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=CIFAR10_CLASSES,
    )

    # Xuat 3 bieu do 300 DPI chuan xuat ban
    print("\nDang xuat 3 bieu do danh gia 300 DPI tai outputs/figures/...")
    history_fig_path = str(FIGURES_DIR / f"{experiment_name}_history.png")
    class_acc_fig_path = str(FIGURES_DIR / f"{experiment_name}_class_accuracy.png")
    cm_fig_path = str(FIGURES_DIR / f"{experiment_name}_confusion_matrix.png")

    plot_training_history(
        history, history_fig_path, title_suffix=f"({args.model} - {args.strategy})"
    )
    plot_per_class_accuracy(
        eval_results["per_class_acc"],
        class_acc_fig_path,
        title_suffix=f"({args.model} - {args.strategy})",
    )
    plot_confusion_matrix(
        eval_results["confusion_matrix"],
        CIFAR10_CLASSES,
        cm_fig_path,
        title_suffix=f"({args.model} - {args.strategy})",
    )

    # Luu them ban sao voi ten mac dinh de tien xem nhanh
    plot_training_history(history, str(FIGURES_DIR / "training_history.png"))
    plot_per_class_accuracy(
        eval_results["per_class_acc"], str(FIGURES_DIR / "class_accuracy.png")
    )
    plot_confusion_matrix(
        eval_results["confusion_matrix"],
        CIFAR10_CLASSES,
        str(FIGURES_DIR / "confusion_matrix.png"),
    )

    print(f"  1. Training History   : {history_fig_path}")
    print(f"  2. Per-class Accuracy : {class_acc_fig_path}")
    print(f"  3. Confusion Matrix   : {cm_fig_path}")

    # Xuat bao cao chi tiet dang JSON tai outputs/reports/
    param_counts = count_parameters(model)
    report_data: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "model": args.model,
        "strategy": args.strategy,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "subset_ratio": args.subset_ratio,
        "dry_run": args.dry_run,
        "parameters": param_counts,
        "best_val_accuracy": trainer.best_val_acc,
        "test_loss": eval_results["test_loss"],
        "test_overall_accuracy": eval_results["overall_acc"],
        "per_class_accuracy": eval_results["per_class_acc"],
        "classification_report": eval_results["classification_report_dict"],
        "training_history": history,
    }

    report_json_path = REPORTS_DIR / f"{experiment_name}_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"  Bao cao thuc nghiem da luu tai: {report_json_path}")
    print("=" * 70)
    print("HOAN TAT THUC NGHIEM THANH CONG!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Dynamic Model Factory & Inspection for Transfer Learning.

Khoi tao mo hinh pre-trained bang Modern Weights API, tu dong thay the Classification Head
phu hop voi tung kien truc (ResNet, VGG, DenseNet) va thuc thi 2 chien luoc dong bang trong so.
"""

from typing import Dict, List
import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS: List[str] = ["resnet18", "resnet50", "vgg16_bn", "densenet121"]
SUPPORTED_STRATEGIES: List[str] = ["freeze_all", "unfreeze_last_block"]


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Dem va thong ke so luong tham so trong mo hinh.

    Args:
        model: Mo hinh PyTorch can phan tich.

    Returns:
        Dict chua 'total', 'trainable', va 'frozen' parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }


def inspect_model(model: nn.Module, model_name: str) -> None:
    """In cau truc phan tang va bang thong ke tham so cua mo hinh ra console.

    Args:
        model: Mo hinh PyTorch can in.
        model_name: Ten dinh danh cua mo hinh.
    """
    stats = count_parameters(model)
    pct_trainable = (
        (stats["trainable"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
    )

    print("=" * 70)
    print(f"MODEL INSPECTION REPORT: {model_name.upper()}")
    print("=" * 70)
    print(f"Architecture Type   : {type(model).__name__}")
    print(f"Total Parameters    : {stats['total']:,}")
    print(f"Trainable Parameters: {stats['trainable']:,} ({pct_trainable:.2f}%)")
    print(f"Frozen Parameters   : {stats['frozen']:,} ({100 - pct_trainable:.2f}%)")
    print("-" * 70)


def build_model(
    model_name: str = "resnet18",
    strategy: str = "freeze_all",
    num_classes: int = 10,
) -> nn.Module:
    """Khoi tao mo hinh pre-trained, thay the classification head va ap dung chien luoc dong bang.

    Args:
        model_name: Ten kien truc ('resnet18', 'resnet50', 'vgg16_bn', 'densenet121').
        strategy: Chien luoc dong bang trong so ('freeze_all', 'unfreeze_last_block').
        num_classes: So luong lop dau ra (mac dinh 10 cho CIFAR-10).

    Returns:
        nn.Module da duoc bien doi va thiet lap requires_grad hop le.

    Raises:
        ValueError: Neu model_name hoac strategy khong nam trong danh sach ho tro.
    """
    model_key = model_name.lower().strip()
    strategy_key = strategy.lower().strip()

    if model_key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Mo hinh '{model_name}' khong duoc ho tro. Danh sach ho tro: {SUPPORTED_MODELS}"
        )
    if strategy_key not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Chien luoc '{strategy}' khong hop le. Danh sach ho tro: {SUPPORTED_STRATEGIES}"
        )

    # 1. Nap mo hinh pre-trained bang Torchvision Modern Weights API
    if model_key == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_key == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_key == "vgg16_bn":
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
    elif model_key == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    # 2. Ap dung chien luoc dong bang trong so backbone truoc khi thay the head
    # Buoc 2.1: Dong bang toan bo cac tang
    for param in model.parameters():
        param.requires_grad = False

    # Buoc 2.2: Neu la unfreeze_last_block, mo khoa khoi tich chap cuoi cung
    if strategy_key == "unfreeze_last_block":
        if model_key in ["resnet18", "resnet50"]:
            # Mo khoa khoi tich chap cuoi cung layer4 cua ResNet
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif model_key == "vgg16_bn":
            # Mo khoa khoi conv cuoi cung (tu layer 34 den het features) cua VGG-16 BN
            for param in model.features[34:].parameters():
                param.requires_grad = True
        elif model_key == "densenet121":
            # Mo khoa khoi denseblock4 va norm5 cuoi cung cua DenseNet-121
            for param in model.features.denseblock4.parameters():
                param.requires_grad = True
            for param in model.features.norm5.parameters():
                param.requires_grad = True

    # 3. Dynamic Head Replacement: Thay the classification head sang num_classes
    # Tang moi khoi tao mac dinh se co requires_grad = True
    if model_key in ["resnet18", "resnet50"]:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_key == "vgg16_bn":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif model_key == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    return model


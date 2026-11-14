"""Unit tests for Milestone 2: Dynamic Model Factory & Inspection."""

import unittest
import torch
import torch.nn as nn

from src.model_factory import (
    SUPPORTED_MODELS,
    SUPPORTED_STRATEGIES,
    build_model,
    count_parameters,
    inspect_model,
)


class TestModelFactory(unittest.TestCase):
    """Test suite cho model_factory.py."""

    def test_unsupported_model_or_strategy(self) -> None:
        """Kiem tra ngoai le ValueError khi truyen sai ten model hoac strategy."""
        with self.assertRaises(ValueError):
            build_model("unsupported_cnn", "freeze_all")

        with self.assertRaises(ValueError):
            build_model("resnet18", "invalid_strategy")

    def test_resnet18_freeze_all(self) -> None:
        """Xac minh ResNet-18 freeze_all: output shape va so tham so trainable = 5,130."""
        model = build_model("resnet18", strategy="freeze_all", num_classes=10)
        self.assertIsInstance(model.fc, nn.Linear)
        self.assertEqual(model.fc.out_features, 10)

        # Thong ke tham so
        stats = count_parameters(model)
        # Tang fc moi: 512 * 10 weights + 10 biases = 5,130
        self.assertEqual(stats["trainable"], 5130)
        self.assertEqual(stats["total"], 11181642)
        self.assertEqual(stats["frozen"], 11181642 - 5130)

        # Forward pass voi dummy tensor
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, torch.Size([2, 10]))

    def test_resnet18_unfreeze_last_block(self) -> None:
        """Xac minh ResNet-18 unfreeze_last_block: layer4 va fc co requires_grad=True."""
        model = build_model("resnet18", strategy="unfreeze_last_block", num_classes=10)
        stats = count_parameters(model)

        # Trainable params phai bao gom toan bo layer4 va fc (khoang ~8.4M params)
        self.assertGreater(stats["trainable"], 5130)
        self.assertLess(stats["trainable"], stats["total"])

        # Kiem tra truc tiep layer4 va layer3
        for p in model.layer4.parameters():
            self.assertTrue(p.requires_grad)
        for p in model.layer3.parameters():
            self.assertFalse(p.requires_grad)

    def test_vgg16_bn_freeze_all(self) -> None:
        """Xac minh VGG-16 BN freeze_all: classifier[6] thay the thanh 10 lop."""
        model = build_model("vgg16_bn", strategy="freeze_all", num_classes=10)
        self.assertIsInstance(model.classifier[6], nn.Linear)
        self.assertEqual(model.classifier[6].out_features, 10)

        stats = count_parameters(model)
        # Classifier[6] moi: 4096 * 10 + 10 = 40,970
        self.assertEqual(stats["trainable"], 40970)

        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, torch.Size([2, 10]))

    def test_densenet121_freeze_all(self) -> None:
        """Xac minh DenseNet-121 freeze_all: classifier thay the thanh 10 lop."""
        model = build_model("densenet121", strategy="freeze_all", num_classes=10)
        self.assertIsInstance(model.classifier, nn.Linear)
        self.assertEqual(model.classifier.out_features, 10)

        stats = count_parameters(model)
        # Classifier moi: 1024 * 10 + 10 = 10,250
        self.assertEqual(stats["trainable"], 10250)

        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, torch.Size([2, 10]))

    def test_inspect_model_runs_cleanly(self) -> None:
        """Kiem tra ham inspect_model khong quang ra Exception."""
        model = build_model("resnet18", strategy="freeze_all", num_classes=10)
        try:
            inspect_model(model, "resnet18")
        except Exception as e:
            self.fail(f"inspect_model failed with error: {e}")


if __name__ == "__main__":
    unittest.main()


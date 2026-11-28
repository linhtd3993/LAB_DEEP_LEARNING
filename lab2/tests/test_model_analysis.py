"""Offline checks for checkpoint identity, CAM math, hooks and CSV provenance."""

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import CIFAR10_CLASSES
from src.model_factory import build_model, SUPPORTED_MODELS, SUPPORTED_STRATEGIES
from src.model_analysis import grad_cam, load_analysis_checkpoint, prediction_groups, validate_prediction_csv


class TestModelAnalysis(unittest.TestCase):
    def test_cam_math_and_cleanup(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer4 = nn.Conv2d(1, 1, 1, bias=False)
                self.layer4.weight.data.fill_(1)
                self.lab2_config = {"model": "resnet18"}

            def forward(self, x):
                score = self.layer4(x).mean((2, 3))
                return score.repeat(1, 10)

        model = Tiny().eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        cam, _ = grad_cam(model, image, 0)
        np.testing.assert_allclose(cam, image.numpy()[0, 0] / 15, atol=1e-6)
        self.assertEqual(len(model.layer4._forward_hooks), 0)
        self.assertIsNone(model.layer4.weight.grad)
        with self.assertRaises(ValueError):
            grad_cam(model, image, 10)
        self.assertEqual(len(model.layer4._forward_hooks), 0)
        self.assertFalse(model.training)

    def test_cam_all_architectures_and_strategies(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for name in SUPPORTED_MODELS:
                for strategy in SUPPORTED_STRATEGIES:
                    with self.subTest(model=name, strategy=strategy):
                        model = build_model(name, strategy, pretrained=False).eval()
                        flags = [p.requires_grad for p in model.parameters()]
                        cam, layer = grad_cam(model, torch.randn(1, 3, 64, 64), 0)
                        self.assertEqual(cam.shape, (64, 64))
                        self.assertTrue(np.isfinite(cam).all())
                        self.assertGreaterEqual(float(cam.min()), 0)
                        self.assertLessEqual(float(cam.max()), 1)
                        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
                        self.assertTrue(all(p.grad is None for p in model.parameters()))
                        self.assertTrue(all(len(m._forward_hooks) == 0 for m in model.modules()))
                        del model
                        gc.collect()
        finally:
            torch.set_num_threads(previous_threads)

    def test_checkpoint_metadata_legacy_and_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            model = build_model("resnet18", "unfreeze_last_block", pretrained=False).eval()
            original = model.fc.weight.detach().clone()
            torch.save({"model_state_dict": model.state_dict(), "config": model.lab2_config}, path)
            loaded, config = load_analysis_checkpoint(path, torch.device("cpu"))
            self.assertEqual(config["strategy"], "unfreeze_last_block")
            self.assertTrue(torch.equal(original, loaded.fc.weight))
            self.assertTrue(next(loaded.layer4.parameters()).requires_grad)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                load_analysis_checkpoint(path, torch.device("cpu"), {"strategy": "freeze_all"})
            torch.save({"model_state_dict": model.state_dict(),
                        "experiment_name": "resnet18_unfreeze_last_block_lr0.001"}, path)
            _, config = load_analysis_checkpoint(path, torch.device("cpu"))
            self.assertEqual(config["model"], "resnet18")
            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_analysis_checkpoint(path, torch.device("cpu"))
            _, config = load_analysis_checkpoint(path, torch.device("cpu"),
                                                {"model": "resnet18", "strategy": "unfreeze_last_block"})
            self.assertEqual(config["strategy"], "unfreeze_last_block")

    def test_groups_empty_and_thresholds(self):
        frame = pd.DataFrame({"true_label": ["cat"] * 4,
                              "predicted_label": ["cat", "cat", "dog", "dog"],
                              "confidence": [.9, .3, .95, .2]})
        groups = prediction_groups(frame)
        self.assertTrue(all(len(group) == 1 for group in groups.values()))
        self.assertEqual(sum(len(g) for g in prediction_groups(frame.iloc[:0]).values()), 0)
        with self.assertRaises(ValueError):
            prediction_groups(frame, high=.3, low=.8)

    def test_csv_consistency(self):
        row = {"sample_index": 1, "true_label": "cat", "predicted_label": "cat", "confidence": .1}
        row.update({f"prob_{name}": .1 for name in CIFAR10_CLASSES})
        fresh = pd.DataFrame([row])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            fresh.to_csv(path, index=False)
            self.assertEqual(len(validate_prediction_csv(path, fresh)), 1)
            wrong = fresh.copy()
            wrong["confidence"] = .5
            wrong.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "probabilities"):
                validate_prediction_csv(path, fresh)
            pd.concat([fresh, fresh]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicated"):
                validate_prediction_csv(path, fresh)


if __name__ == "__main__":
    unittest.main()

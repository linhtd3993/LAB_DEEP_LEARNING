"""Offline unit tests for CLI artifacts and checkpoint failures."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

import main
from src.evaluate import evaluate_test_set
from src.trainer import Trainer


class TestRunArtifacts(unittest.TestCase):
    def test_frozen_batchnorm_statistics_stay_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Sequential(nn.BatchNorm1d(10), nn.Linear(10, 10))
            for parameter in model[0].parameters():
                parameter.requires_grad = False
            loader = DataLoader(TensorDataset(torch.randn(8, 10), torch.arange(8)), batch_size=4)
            trainer = Trainer(model, loader, loader,
                              torch.optim.SGD(model[1].parameters(), lr=0.01),
                              nn.CrossEntropyLoss(), torch.device("cpu"),
                              checkpoint_dir=directory, log_dir=directory, use_amp=False)
            before = model[0].running_mean.clone()
            try:
                trainer.train_one_epoch(0)
                self.assertTrue(torch.equal(before, model[0].running_mean))
                self.assertTrue(model[1].training)
            finally:
                trainer.writer.close()

    def test_shuffled_loader_rejected(self):
        loader = DataLoader(TensorDataset(torch.eye(10), torch.arange(10)), shuffle=True)
        with self.assertRaisesRegex(ValueError, "sequential"):
            evaluate_test_set(nn.Identity(), loader, nn.CrossEntropyLoss(),
                              torch.device("cpu"), main.CIFAR10_CLASSES)

    def test_incorrect_predictions_match_confusion_matrix(self):
        loader = DataLoader(TensorDataset(torch.eye(10) * 8,
                                         (torch.arange(10) + 1) % 10), batch_size=3)
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_test_set(nn.Identity(), loader, nn.CrossEntropyLoss(),
                                       torch.device("cpu"), main.CIFAR10_CLASSES)
        self.assertEqual(result["overall_acc"], 0)
        self.assertEqual(result["confusion_matrix"].trace(), 0)
        self.assertTrue(all(not row["is_correct"] for row in result["predictions"]))
        self.assertTrue(all(row["confidence"] == row["top1_probability"]
                            for row in result["predictions"]))

    def test_unique_run_id(self):
        self.assertNotEqual(main.create_run_id("resnet18", "freeze_all"),
                            main.create_run_id("resnet18", "freeze_all"))

    def test_missing_and_invalid_checkpoint(self):
        model = nn.Linear(2, 10)
        with self.assertRaises(FileNotFoundError):
            main.load_best_checkpoint(model, "", torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pth"
            torch.save({}, path)
            with self.assertRaisesRegex(RuntimeError, "Cannot load best checkpoint"):
                main.load_best_checkpoint(model, str(path), torch.device("cpu"))

    def test_predictions_preserve_nested_subset_indices(self):
        data = TensorDataset(torch.eye(10) * 8, torch.arange(10))
        loader = DataLoader(Subset(Subset(data, [9, 3, 1, 7]), [2, 0, 1]), batch_size=2)
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_test_set(nn.Identity(), loader, nn.CrossEntropyLoss(),
                                       torch.device("cpu"), main.CIFAR10_CLASSES)
        self.assertEqual([r["sample_index"] for r in result["predictions"]], [1, 9, 3])
        self.assertEqual(result["overall_acc"], 100)
        self.assertEqual(result["confusion_matrix"].trace(), 3)
        for row in result["predictions"]:
            self.assertTrue(row["is_correct"])
            self.assertEqual(row["predicted_label"], row["top1_label"])
            self.assertAlmostEqual(sum(row[f"prob_{name}"] for name in main.CIFAR10_CLASSES), 1, places=6)

    def test_main_exports_consistent_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loader = DataLoader(TensorDataset(torch.eye(10), torch.arange(10)), batch_size=5)
            model = nn.Linear(10, 10)

            class FakeTrainer:
                def __init__(self, **kwargs):
                    self.best_checkpoint_path = str(root / "best.pth")
                    self.tb_dir = root / "logs"
                    self.tb_dir.mkdir()
                    (self.tb_dir / "events.test").write_bytes(b"unit-test")
                    self.best_val_acc = 50.0

                def fit(self):
                    torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, self.best_checkpoint_path)
                    return {"train_loss": [2.3], "val_loss": [2.3], "train_acc": [10],
                            "val_acc": [10], "epoch_time": [1.0], "lr": [0.001]}

            def figure(*args, **kwargs):
                Path(args[1] if len(args) == 2 else args[2]).write_bytes(b"unit-test-figure")

            with patch("sys.argv", ["main.py", "--epochs", "1", "--output-dir", directory]), \
                 patch.object(main, "ensure_directories"), \
                 patch.object(main, "get_device", return_value=torch.device("cpu")), \
                 patch.object(main, "build_model", return_value=model), \
                 patch.object(main, "get_cifar10_dataloaders", return_value=(loader, loader, loader)), \
                 patch.object(main, "Trainer", FakeTrainer), \
                 patch.object(main, "plot_training_history", side_effect=figure), \
                 patch.object(main, "plot_per_class_accuracy", side_effect=figure), \
                 patch.object(main, "plot_confusion_matrix", side_effect=figure), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(), 0)
            report_path = next(root.glob("*/report.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], "1.0")
            for key in ("run_id", "started_at", "finished_at", "config", "environment",
                        "dataset", "training", "evaluation", "artifacts"):
                self.assertIn(key, report)
            self.assertEqual(report["dataset"]["class_names"], main.CIFAR10_CLASSES)
            self.assertEqual(report["dataset"]["test_samples"], 10)
            self.assertEqual(report["training"]["best_epoch"], 1)
            with Path(report["artifacts"]["predictions"]).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            expected_columns = {"sample_index", "true_label", "predicted_label", "is_correct",
                                "confidence", "margin_top1_top2"}
            expected_columns.update(f"top{k}_{field}" for k in range(1, 4)
                                    for field in ("label", "probability"))
            expected_columns.update(f"prob_{name}" for name in main.CIFAR10_CLASSES)
            self.assertEqual(set(rows[0]), expected_columns)
            self.assertEqual([int(row["sample_index"]) for row in rows], list(range(10)))
            reconstructed_cm = [[0] * 10 for _ in range(10)]
            for row in rows:
                probabilities = {name: float(row[f"prob_{name}"]) for name in main.CIFAR10_CLASSES}
                self.assertTrue(all(0 <= p <= 1 for p in probabilities.values()))
                self.assertAlmostEqual(sum(probabilities.values()), 1, places=6)
                ranking = sorted(main.CIFAR10_CLASSES,
                                 key=lambda name: (-probabilities[name], main.CIFAR10_CLASSES.index(name)))
                for k in range(1, 4):
                    self.assertEqual(row[f"top{k}_label"], ranking[k - 1])
                    self.assertAlmostEqual(float(row[f"top{k}_probability"]), probabilities[ranking[k - 1]])
                self.assertEqual(row["predicted_label"], ranking[0])
                self.assertEqual(row["is_correct"], str(row["true_label"] == row["predicted_label"]))
                self.assertAlmostEqual(float(row["confidence"]), probabilities[ranking[0]])
                self.assertAlmostEqual(float(row["margin_top1_top2"]), probabilities[ranking[0]] - probabilities[ranking[1]])
                reconstructed_cm[main.CIFAR10_CLASSES.index(row["true_label"])][main.CIFAR10_CLASSES.index(row["predicted_label"])] += 1
            self.assertEqual(report["evaluation"]["confusion_matrix"], reconstructed_cm)
            correct = sum(row["is_correct"] == "True" for row in rows)
            self.assertEqual(report["evaluation"]["accuracy_pct"], correct * 10)
            self.assertEqual(sum(map(sum, report["evaluation"]["confusion_matrix"])), 10)
            with Path(report["artifacts"]["summary"]).open(encoding="utf-8", newline="") as stream:
                summaries = list(csv.DictReader(stream))
            self.assertEqual(len(summaries), 1)
            summary = summaries[0]
            self.assertEqual(set(summary), {"run_id", "model", "strategy", "optimizer", "lr",
                "batch_size", "epochs", "seed", "best_epoch", "best_val_accuracy_pct",
                "test_accuracy_pct", "test_macro_f1", "avg_epoch_time_s",
                "inference_ms_per_image", "total_params", "trainable_params"})
            self.assertEqual(summary["run_id"], report["run_id"])
            self.assertEqual(float(summary["test_macro_f1"]), report["evaluation"]["macro_f1"])
            self.assertEqual(float(summary["test_accuracy_pct"]), report["evaluation"]["accuracy_pct"])


if __name__ == "__main__":
    unittest.main()

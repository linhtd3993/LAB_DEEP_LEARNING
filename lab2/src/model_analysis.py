"""Checkpoint restoration and Grad-CAM shared by the evaluation notebook.

Grad-CAM follows Selvaraju et al., ICCV 2017 (arXiv:1610.02391).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .config import CIFAR10_CLASSES
from .model_factory import build_model, SUPPORTED_MODELS, SUPPORTED_STRATEGIES


def load_analysis_checkpoint(path: Path, device: torch.device,
                             config_override: Optional[dict] = None):
    """Restore metadata checkpoints and legacy checkpoints named by experiment.

For a legacy checkpoint with no strategy metadata, the caller must supply an
explicit config. Freezing strategy is not encoded in a state_dict.
"""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    config = dict(checkpoint.get("config", {}))
    name = str(checkpoint.get("experiment_name", path.stem.removeprefix("best_")))
    if not config:
        for model_name in SUPPORTED_MODELS:
            for strategy in SUPPORTED_STRATEGIES:
                prefix = f"{model_name}_{strategy}"
                if name == prefix or name.startswith(prefix + "_"):
                    config = {"model": model_name, "strategy": strategy}
    for key, value in (config_override or {}).items():
        if key in config and config[key] != value:
            raise ValueError(f"Config override conflicts with checkpoint metadata: {key}")
        config[key] = value
    if config.get("model") not in SUPPORTED_MODELS or config.get("strategy") not in SUPPORTED_STRATEGIES:
        raise ValueError("Checkpoint needs model/strategy metadata or explicit config_override")
    if config.get("num_classes", 10) != 10:
        raise ValueError("Analysis expects a 10-class CIFAR-10 checkpoint")
    model = build_model(config["model"], config["strategy"], 10, pretrained=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, config


def grad_cam(model: nn.Module, image: torch.Tensor, class_index: int):
    """Return a normalized H×W heatmap and layer name for one image.

    Input gradients keep the graph active for frozen feature extractors.
    Tensor hooks avoid module backward hooks conflicting with inplace ReLU.
    Model modes, parameters, gradients, and hooks are preserved on exit.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects a batch containing one image")
    name = model.lab2_config["model"]
    if name.startswith("resnet"):
        layer_name, layer = "layer4", model.layer4
    elif name == "vgg16_bn":
        index = max(i for i, layer in enumerate(model.features) if isinstance(layer, nn.Conv2d))
        layer_name, layer = f"features.{index}", model.features[index]
    else:
        layer_name, layer = "features.norm5", model.features.norm5
    captured = {}
    tensor_handles = []

    def capture(module, inputs, output):
        captured["activation"] = output.detach().clone()
        tensor_handles.append(output.register_hook(lambda grad: captured.update(gradient=grad.detach())))

    modes = [(module, module.training) for module in model.modules()]
    handle = layer.register_forward_hook(capture)
    try:
        model.eval()
        with torch.enable_grad():
            x = image.detach().clone().requires_grad_(True)
            logits = model(x)
            if not 0 <= class_index < logits.shape[1]:
                raise ValueError("Invalid Grad-CAM target class")
            torch.autograd.grad(logits[0, class_index], x)
            weights = captured["gradient"].mean(dim=(2, 3), keepdim=True)
            heatmap = (weights * captured["activation"]).sum(dim=1, keepdim=True).relu()
            heatmap = F.interpolate(heatmap, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            heatmap -= heatmap.min()
            peak = heatmap.max()
            if peak > 0:
                heatmap /= peak
            return heatmap.cpu().numpy(), layer_name
    finally:
        handle.remove()
        for tensor_handle in tensor_handles:
            tensor_handle.remove()
        for module, mode in modes:
            module.training = mode


def prediction_groups(frame: pd.DataFrame, high: float = .8, low: float = .5, limit: int = 4):
    """Select disjoint confidence groups; empty groups remain empty."""
    if not 0 <= low < high <= 1:
        raise ValueError("Require 0 <= low < high <= 1")
    correct = frame.true_label == frame.predicted_label
    return {
        "Đúng — confidence cao": frame[correct & (frame.confidence >= high)].nlargest(limit, "confidence"),
        "Đúng — confidence thấp": frame[correct & (frame.confidence <= low)].nsmallest(limit, "confidence"),
        "Sai — confidence cao": frame[~correct & (frame.confidence >= high)].nlargest(limit, "confidence"),
        "Sai — confidence thấp": frame[~correct & (frame.confidence <= low)].nsmallest(limit, "confidence"),
    }


def validate_prediction_csv(path: Path, fresh: pd.DataFrame):
    """Validate cached rows against a new inference pass from the chosen checkpoint.

    Validation strategy (two-pass):
      1. sample_index + true_label must match exactly — these are ground-truth
         values that never change between runs.
      2. Per-class probabilities must be numerically close (atol=1e-5) — this
         tolerates the tiny floating-point differences that arise when the same
         checkpoint is re-evaluated on CPU vs GPU or across PyTorch versions.

    predicted_label is intentionally NOT compared directly because it is derived
    from argmax(probabilities). On borderline samples where two classes share
    nearly identical softmax scores, CPU and GPU may produce different argmax
    results while their probabilities still pass np.allclose. Validating the
    raw probabilities is both more robust and more informative.
    """
    cached = pd.read_csv(path)
    columns = ["sample_index", "true_label", "predicted_label", "confidence"]
    columns += [f"prob_{name}" for name in CIFAR10_CLASSES]
    if not set(columns).issubset(cached.columns) or cached.sample_index.duplicated().any():
        raise ValueError("Predictions CSV has missing columns or duplicated sample IDs")
    cached = cached.sort_values("sample_index").reset_index(drop=True)
    fresh = fresh.sort_values("sample_index").reset_index(drop=True)

    # --- Pass 1: exact match on ground-truth identity columns only ---
    # Coerce to stable dtypes before comparing (pandas may read int64 while
    # evaluate_test_set emits Python int; StringDtype variants also differ).
    id_cols = ["sample_index", "true_label"]
    cached_id = cached[id_cols].copy()
    fresh_id = fresh[id_cols].copy()
    cached_id["sample_index"] = cached_id["sample_index"].astype(np.int64)
    fresh_id["sample_index"] = fresh_id["sample_index"].astype(np.int64)
    cached_id["true_label"] = cached_id["true_label"].astype(str)
    fresh_id["true_label"] = fresh_id["true_label"].astype(str)

    if not cached_id.equals(fresh_id):
        mismatch = ~(cached_id == fresh_id).all(axis=1)
        n = int(mismatch.sum())
        raise ValueError(
            f"Predictions CSV sample IDs or ground-truth labels do not match "
            f"this evaluation ({n} row(s) differ). "
            "Ensure PREDICTIONS_CSV was produced by the same dataset split."
        )

    # --- Pass 2: numeric tolerance on probabilities (tolerates CPU/GPU drift) ---
    prob_cols = [f"prob_{name}" for name in CIFAR10_CLASSES]
    if not np.allclose(cached[prob_cols].values, fresh[prob_cols].values, atol=1e-5, rtol=1e-4):
        # Find first diverging sample for a helpful message
        abs_diff = np.abs(cached[prob_cols].values - fresh[prob_cols].values)
        worst_row = int(np.argmax(abs_diff.max(axis=1)))
        worst_col = prob_cols[int(np.argmax(abs_diff[worst_row]))]
        raise ValueError(
            f"Predictions CSV probabilities do not match the selected checkpoint "
            f"(largest diff at row {worst_row}, column '{worst_col}'). "
            "Check that PREDICTIONS_CSV was generated by this exact checkpoint."
        )

    return cached

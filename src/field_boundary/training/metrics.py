from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
import torch


def binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    dims = (1, 2, 3)
    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1 - targets)).sum(dim=dims)
    fn = ((1 - preds) * targets).sum(dim=dims)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean()), "dice": float(dice.mean()), "iou": float(iou.mean())}


def compute_binary_metrics(probs: torch.Tensor, targets: torch.Tensor, threshold: float, eps: float = 1e-7) -> Dict[str, np.ndarray]:
    preds = (probs >= threshold).float()
    dims = (1, 2, 3)
    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1 - targets)).sum(dim=dims)
    fn = ((1 - preds) * targets).sum(dim=dims)
    tn = ((1 - preds) * (1 - targets)).sum(dim=dims)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return {k: v.detach().cpu().numpy() for k, v in {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "dice": dice, "iou": iou}.items()}


def summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())}

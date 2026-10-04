from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().numpy()
    if image_np.shape[0] >= 3:
        rgb = image_np[:3]
    else:
        rgb = np.repeat(image_np[:1], repeats=3, axis=0)
    rgb = np.moveaxis(rgb, 0, -1)
    return np.clip(rgb, 0, 1)


def save_training_curves(metrics: List[Dict[str, float]], output_path: Path) -> None:
    if not metrics:
        return
    epochs = [m["epoch"] for m in metrics]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, [m["train_loss"] for m in metrics], label="train")
    axes[0].plot(epochs, [m["val_loss"] for m in metrics], label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [m["train_dice"] for m in metrics], label="train")
    axes[1].plot(epochs, [m["val_dice"] for m in metrics], label="val")
    axes[1].set_title("Dice"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, [m["train_iou"] for m in metrics], label="train")
    axes[2].plot(epochs, [m["val_iou"] for m in metrics], label="val")
    axes[2].set_title("IoU"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    for ax in axes: ax.set_xlabel("Epoch")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_prediction_panel(rgb, mask, prob, pred, output_path: Path, title: str = "Predição") -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(rgb); axes[0].set_title("Imagem"); axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Máscara real"); axes[1].axis("off")
    axes[2].imshow(prob, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Probabilidade"); axes[2].axis("off")
    axes[3].imshow(pred, cmap="gray"); axes[3].set_title("Predição binária"); axes[3].axis("off")
    axes[4].imshow(rgb); axes[4].imshow(pred, cmap="Reds", alpha=0.35); axes[4].set_title("Sobreposição"); axes[4].axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from field_boundary.data.dataset import AI4BoundariesSegmentationDataset
from field_boundary.models.factory import build_model


def build_official_val_pairs(data_root: Path) -> List[Tuple[Path, Path]]:
    image_dir = data_root / "orthos" / "images" / "val"
    mask_dir = data_root / "orthos" / "masks" / "val"

    pairs = []

    for image_path in sorted(image_dir.glob("*_ortho_1m_512.tif")):
        mask_name = image_path.name.replace(
            "_ortho_1m_512.tif",
            "_ortholabel_1m_512.tif",
        )
        mask_path = mask_dir / mask_name

        if mask_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


def infer_architecture_from_state_dict(state_dict: Dict) -> str | None:
    keys = list(state_dict.keys())

    if any(k.startswith("decoder.aspp") or k.startswith("decoder.block1") for k in keys):
        return "deeplabv3plus"

    if any(k.startswith("decoder.blocks") for k in keys):
        return "unet"

    return None


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    state_dict = checkpoint["model_state_dict"]

    architecture = config.get("architecture") or infer_architecture_from_state_dict(state_dict) or "unet"
    encoder_name = config.get("encoder_name", "resnet34")
    in_channels = int(config.get("in_channels", 3))

    print(f"\nCarregando modelo: {checkpoint_path}")
    print(f"Arquitetura: {architecture}")
    print(f"Encoder: {encoder_name}")
    print(f"In channels: {in_channels}")

    model = build_model(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def compute_binary_metrics(probs: torch.Tensor, targets: torch.Tensor, threshold: float, eps: float = 1e-7):
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

    return {
        "tp": tp.detach().cpu().numpy(),
        "fp": fp.detach().cpu().numpy(),
        "fn": fn.detach().cpu().numpy(),
        "tn": tn.detach().cpu().numpy(),
        "precision": precision.detach().cpu().numpy(),
        "recall": recall.detach().cpu().numpy(),
        "f1": f1.detach().cpu().numpy(),
        "dice": dice.detach().cpu().numpy(),
        "iou": iou.detach().cpu().numpy(),
    }


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


@torch.no_grad()
def evaluate_ensemble(
    unet_model,
    deeplab_model,
    loader,
    device,
    unet_weight: float,
    threshold: float,
):
    deeplab_weight = 1.0 - unet_weight
    rows = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        image_paths = batch["image_path"]
        mask_paths = batch["mask_path"]

        probs_unet = torch.sigmoid(unet_model(images))
        probs_deeplab = torch.sigmoid(deeplab_model(images))

        probs_final = unet_weight * probs_unet + deeplab_weight * probs_deeplab

        metrics = compute_binary_metrics(
            probs=probs_final,
            targets=masks,
            threshold=threshold,
        )

        for i in range(images.shape[0]):
            rows.append(
                {
                    "image_path": image_paths[i],
                    "mask_path": mask_paths[i],
                    "unet_weight": unet_weight,
                    "deeplab_weight": deeplab_weight,
                    "threshold": threshold,
                    "dice": float(metrics["dice"][i]),
                    "iou": float(metrics["iou"][i]),
                    "precision": float(metrics["precision"][i]),
                    "recall": float(metrics["recall"][i]),
                    "f1": float(metrics["f1"][i]),
                    "tp": float(metrics["tp"][i]),
                    "fp": float(metrics["fp"][i]),
                    "fn": float(metrics["fn"][i]),
                    "tn": float(metrics["tn"][i]),
                }
            )

    summary = {
        "dice": summarize([r["dice"] for r in rows]),
        "iou": summarize([r["iou"] for r in rows]),
        "precision": summarize([r["precision"] for r in rows]),
        "recall": summarize([r["recall"] for r in rows]),
        "f1": summarize([r["f1"] for r in rows]),
    }

    return rows, summary


def save_rows_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Avaliação de ensemble U-Net + DeepLabV3+")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ai4boundaries"))
    parser.add_argument("--unet-checkpoint", type=Path, required=True)
    parser.add_argument("--deeplab-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--unet-weight", type=float, default=0.80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    pairs = build_official_val_pairs(args.data_root)
    print(f"Pares oficiais de validação: {len(pairs)}")

    dataset = AI4BoundariesSegmentationDataset(
        pairs=pairs,
        image_size=args.image_size,
        in_channels=args.in_channels,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    unet_model = load_model(args.unet_checkpoint, device)
    deeplab_model = load_model(args.deeplab_checkpoint, device)

    rows, summary = evaluate_ensemble(
        unet_model=unet_model,
        deeplab_model=deeplab_model,
        loader=loader,
        device=device,
        unet_weight=args.unet_weight,
        threshold=args.threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_rows_csv(rows, args.output_dir / "per_sample_metrics.csv")
    save_json(
        {
            "unet_checkpoint": str(args.unet_checkpoint),
            "deeplab_checkpoint": str(args.deeplab_checkpoint),
            "unet_weight": args.unet_weight,
            "deeplab_weight": 1.0 - args.unet_weight,
            "threshold": args.threshold,
            "num_samples": len(dataset),
            "summary": summary,
        },
        args.output_dir / "summary_metrics.json",
    )

    print("\nResumo:")
    for metric_name, metric_summary in summary.items():
        print(metric_name, metric_summary)

    print(f"\nArquivos salvos em: {args.output_dir}")


if __name__ == "__main__":
    main()
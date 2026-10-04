from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from field_boundary.data.dataset import AI4BoundariesSegmentationDataset
from field_boundary.models.factory import build_model


def build_pairs(data_root: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = data_root / "orthos" / "images" / split
    mask_dir = data_root / "orthos" / "masks" / split

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


def normalize_image_percentile(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)

    for c in range(img.shape[0]):
        band = img[c]
        valid = band[np.isfinite(band)]

        if valid.size == 0:
            continue

        p2, p98 = np.percentile(valid, [2, 98])

        if np.isclose(p2, p98):
            out[c] = 0.0
            continue

        band = np.clip(band, p2, p98)
        out[c] = (band - p2) / (p98 - p2 + eps)

    return out


def force_channel_count(img: np.ndarray, in_channels: int) -> np.ndarray:
    channels, _, _ = img.shape

    if channels == in_channels:
        return img

    if channels > in_channels:
        return img[:in_channels]

    missing = in_channels - channels
    last_band = img[-1:, :, :]
    padding = np.repeat(last_band, repeats=missing, axis=0)

    return np.concatenate([img, padding], axis=0)


def prepare_classifier_tensor(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    with rasterio.open(image_path) as src:
        img = src.read()

    img = force_channel_count(img, 3)
    img = normalize_image_percentile(img)

    tensor = torch.from_numpy(img).float().unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    return tensor.to(device)


def build_classifier(model_name: str):
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    if model_name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    raise ValueError(f"Classificador não suportado: {model_name}")


def load_classifier(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = checkpoint.get("model_name", "resnet18")
    image_size = int(checkpoint.get("image_size", 256))

    print(f"\nCarregando classificador: {checkpoint_path}")
    print(f"Modelo: {model_name}")
    print(f"Image size: {image_size}")

    model = build_classifier(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, image_size


def infer_architecture_from_state_dict(state_dict: Dict) -> str | None:
    keys = list(state_dict.keys())

    if any(k.startswith("decoder.aspp") or k.startswith("decoder.block1") for k in keys):
        return "deeplabv3plus"

    if any(k.startswith("decoder.blocks") for k in keys):
        return "unet"

    return None


def load_segmenter(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    state_dict = checkpoint["model_state_dict"]

    architecture = config.get("architecture") or infer_architecture_from_state_dict(state_dict) or "unet"
    encoder_name = config.get("encoder_name", "resnet34")
    in_channels = int(config.get("in_channels", 3))

    print(f"\nCarregando segmentador: {checkpoint_path}")
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


def compute_counts(pred: np.ndarray, target: np.ndarray):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    tn = float(np.logical_and(~pred, ~target).sum())

    return tp, fp, fn, tn


def compute_metrics(tp, fp, fn, tn, eps=1e-7):
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return precision, recall, f1, dice, iou


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def classify_mask_size(ratio: float) -> str:
    if ratio == 0:
        return "empty"
    if ratio < 0.01:
        return "very_small_<1%"
    if ratio < 0.05:
        return "small_1_5%"
    if ratio < 0.25:
        return "medium_5_25%"
    if ratio < 0.75:
        return "large_25_75%"
    return "very_large_>75%"


def save_csv(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Teste 017 — Avaliação final do pipeline no split test")

    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ai4boundaries"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path("outputs/tests/test_014_empty_classifier_resnet18_256_fast/best_empty_classifier.pt"),
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=Path("outputs/tests/test_008_unet_imagenet_1000samples_10epochs/best_unet_baseline.pt"),
    )
    parser.add_argument(
        "--deeplab-checkpoint",
        type=Path,
        default=Path("outputs/tests/test_007_deeplabv3plus_1000samples_10epochs/best_unet_baseline.pt"),
    )

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_test/test_017_pipeline_final"))

    parser.add_argument("--classifier-threshold", type=float, default=0.75)
    parser.add_argument("--segmentation-threshold", type=float, default=0.60)
    parser.add_argument("--unet-weight", type=float, default=0.70)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args.data_root, args.split)

    print(f"Pares encontrados no split {args.split}: {len(pairs)}")

    if len(pairs) == 0:
        raise RuntimeError("Nenhum par imagem/máscara encontrado.")

    classifier, classifier_image_size = load_classifier(args.classifier_checkpoint, device)
    unet = load_segmenter(args.unet_checkpoint, device)
    deeplab = load_segmenter(args.deeplab_checkpoint, device)

    deeplab_weight = 1.0 - args.unet_weight

    dataset = AI4BoundariesSegmentationDataset(
        pairs=pairs,
        image_size=args.image_size,
        in_channels=args.in_channels,
        augment=False,
    )

    rows = []

    for idx in range(len(dataset)):
        if idx % 50 == 0:
            print(f"Processando {idx}/{len(dataset)}...")

        image_path, mask_path = pairs[idx]

        classifier_tensor = prepare_classifier_tensor(
            image_path=image_path,
            image_size=classifier_image_size,
            device=device,
        )

        clf_logit = classifier(classifier_tensor).squeeze(1)
        prob_non_empty = float(torch.sigmoid(clf_logit).detach().cpu().item())

        pass_gate = prob_non_empty >= args.classifier_threshold

        sample = dataset[idx]

        image = sample["image"].unsqueeze(0).to(device)
        target = sample["mask"].detach().cpu().numpy()[0].astype(np.uint8)

        if pass_gate:
            probs_unet = torch.sigmoid(unet(image))
            probs_deeplab = torch.sigmoid(deeplab(image))

            probs_final = args.unet_weight * probs_unet + deeplab_weight * probs_deeplab
            prob_np = probs_final[0, 0].detach().cpu().numpy()

            pred = (prob_np >= args.segmentation_threshold).astype(np.uint8)
        else:
            pred = np.zeros_like(target, dtype=np.uint8)

        tp, fp, fn, tn = compute_counts(pred, target)
        precision, recall, f1, dice, iou = compute_metrics(tp, fp, fn, tn)

        target_pixels = tp + fn
        pred_pixels = tp + fp
        total_pixels = tp + fp + fn + tn

        target_ratio = target_pixels / total_pixels if total_pixels > 0 else 0.0
        pred_ratio = pred_pixels / total_pixels if total_pixels > 0 else 0.0

        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "prob_non_empty": prob_non_empty,
                "pass_gate": pass_gate,

                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,

                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,

                "target_pixels": target_pixels,
                "pred_pixels": pred_pixels,
                "total_pixels": total_pixels,
                "target_ratio": target_ratio,
                "pred_ratio": pred_ratio,
                "mask_size_class": classify_mask_size(target_ratio),
            }
        )

    df = pd.DataFrame(rows)

    per_sample_path = args.output_dir / "per_sample_metrics.csv"
    df.to_csv(per_sample_path, index=False)

    summary = {
        "split": args.split,
        "num_samples": int(len(df)),
        "classifier_threshold": args.classifier_threshold,
        "segmentation_threshold": args.segmentation_threshold,
        "unet_weight": args.unet_weight,
        "deeplab_weight": deeplab_weight,
        "dice": summarize(df["dice"]),
        "iou": summarize(df["iou"]),
        "precision": summarize(df["precision"]),
        "recall": summarize(df["recall"]),
        "f1": summarize(df["f1"]),
    }

    empty = df[df["target_pixels"] == 0]
    non_empty = df[df["target_pixels"] > 0]

    summary["empty_count"] = int(len(empty))
    summary["non_empty_count"] = int(len(non_empty))

    if len(empty) > 0:
        summary["empty_false_positive_rate"] = float((empty["pred_pixels"] > 0).mean())
        summary["empty_pred_pixels_mean"] = float(empty["pred_pixels"].mean())
        summary["empty_pred_pixels_median"] = float(empty["pred_pixels"].median())

    if len(non_empty) > 0:
        summary["non_empty_blocked_rate"] = float((~non_empty["pass_gate"]).mean())
        summary["non_empty_kept_rate"] = float(non_empty["pass_gate"].mean())

    summary_path = args.output_dir / "summary_metrics.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_by_size = (
        df.groupby("mask_size_class")
        .agg(
            count=("dice", "count"),
            dice_mean=("dice", "mean"),
            dice_median=("dice", "median"),
            iou_mean=("iou", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            pred_pixels_mean=("pred_pixels", "mean"),
            target_pixels_mean=("target_pixels", "mean"),
            target_ratio_mean=("target_ratio", "mean"),
            pred_ratio_mean=("pred_ratio", "mean"),
        )
        .reset_index()
    )

    by_size_path = args.output_dir / "summary_by_mask_size.csv"
    summary_by_size.to_csv(by_size_path, index=False)

    print("\nResumo final no TEST:")
    print("dice", summary["dice"])
    print("iou", summary["iou"])
    print("precision", summary["precision"])
    print("recall", summary["recall"])
    print("f1", summary["f1"])

    print("\nResumo operacional:")
    print("empty_count:", summary["empty_count"])
    print("non_empty_count:", summary["non_empty_count"])
    print("empty_false_positive_rate:", summary.get("empty_false_positive_rate"))
    print("empty_pred_pixels_mean:", summary.get("empty_pred_pixels_mean"))
    print("non_empty_blocked_rate:", summary.get("non_empty_blocked_rate"))

    print("\nPor tamanho de máscara:")
    print(summary_by_size)

    print(f"\nArquivos salvos em: {args.output_dir}")
    print(f"- {per_sample_path}")
    print(f"- {summary_path}")
    print(f"- {by_size_path}")


if __name__ == "__main__":
    main()
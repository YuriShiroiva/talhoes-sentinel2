from __future__ import annotations

import argparse
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


def prepare_classifier_tensor(
    image_path: Path,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
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


def compute_metrics(tp: float, fp: float, fn: float, tn: float, eps: float = 1e-7):
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
    }


def summarize(series: pd.Series):
    arr = np.asarray(series, dtype=np.float64)

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


def add_metric_columns(row: Dict, prefix: str, tp: float, fp: float, fn: float, tn: float):
    metrics = compute_metrics(tp, fp, fn, tn)

    row[f"tp_{prefix}"] = tp
    row[f"fp_{prefix}"] = fp
    row[f"fn_{prefix}"] = fn
    row[f"tn_{prefix}"] = tn

    row[f"dice_{prefix}"] = metrics["dice"]
    row[f"iou_{prefix}"] = metrics["iou"]
    row[f"precision_{prefix}"] = metrics["precision"]
    row[f"recall_{prefix}"] = metrics["recall"]
    row[f"f1_{prefix}"] = metrics["f1"]


def make_summary(df: pd.DataFrame, prefix: str, classifier_threshold: float, segmentation_threshold: float):
    summary = {
        "classifier_threshold": classifier_threshold,
        "segmentation_threshold": segmentation_threshold,
        "dice": summarize(df[f"dice_{prefix}"]),
        "iou": summarize(df[f"iou_{prefix}"]),
        "precision": summarize(df[f"precision_{prefix}"]),
        "recall": summarize(df[f"recall_{prefix}"]),
        "f1": summarize(df[f"f1_{prefix}"]),
    }

    empty = df[df["target_pixels"] == 0]
    non_empty = df[df["target_pixels"] > 0]

    summary["empty_count"] = int(len(empty))
    summary["non_empty_count"] = int(len(non_empty))

    if len(empty) > 0:
        summary["empty_false_positive_rate"] = float((empty[f"pred_pixels_{prefix}"] > 0).mean())
        summary["empty_pred_pixels_mean"] = float(empty[f"pred_pixels_{prefix}"].mean())
        summary["empty_pred_pixels_median"] = float(empty[f"pred_pixels_{prefix}"].median())

    if len(non_empty) > 0:
        summary["non_empty_blocked_rate"] = float((~non_empty["pass_gate"]).mean())
        summary["non_empty_kept_rate"] = float(non_empty["pass_gate"].mean())

    return summary


def make_summary_by_size(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return (
        df.groupby("mask_size_class")
        .agg(
            count=(f"dice_{prefix}", "count"),
            dice_mean=(f"dice_{prefix}", "mean"),
            dice_median=(f"dice_{prefix}", "median"),
            iou_mean=(f"iou_{prefix}", "mean"),
            precision_mean=(f"precision_{prefix}", "mean"),
            recall_mean=(f"recall_{prefix}", "mean"),
            f1_mean=(f"f1_{prefix}", "mean"),
            pred_pixels_mean=(f"pred_pixels_{prefix}", "mean"),
            target_pixels_mean=("target_pixels", "mean"),
            target_ratio_mean=("target_ratio", "mean"),
            pred_ratio_mean=(f"pred_ratio_{prefix}", "mean"),
        )
        .reset_index()
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Avalia o pipeline final ResNet18 + U-Net sem DeepLab/ensemble."
    )

    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ai4boundaries"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path("outputs/tests/test_014_empty_classifier_resnet18_256_fast/best_empty_classifier.pt"),
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        type=Path,
        default=Path("outputs/tests/test_019_unet_imagenet_3000samples_20epochs/best_unet_baseline.pt"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_model/test020_unet_pipeline_clean"),
    )

    parser.add_argument("--classifier-threshold", type=float, default=0.75)
    parser.add_argument("--segmentation-threshold", type=float, default=0.65)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args.data_root, args.split)

    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]

    print(f"Pares encontrados no split {args.split}: {len(pairs)}")

    if len(pairs) == 0:
        raise RuntimeError("Nenhum par imagem/máscara encontrado.")

    classifier, classifier_image_size = load_classifier(args.classifier_checkpoint, device)
    segmenter = load_segmenter(args.segmenter_checkpoint, device)

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

        probs = torch.sigmoid(segmenter(image))
        prob_np = probs[0, 0].detach().cpu().numpy()

        pred_before_gate = (prob_np >= args.segmentation_threshold).astype(np.uint8)

        if pass_gate:
            pred_after_gate = pred_before_gate.copy()
        else:
            pred_after_gate = np.zeros_like(pred_before_gate, dtype=np.uint8)

        tp_before, fp_before, fn_before, tn_before = compute_counts(pred_before_gate, target)
        tp_after, fp_after, fn_after, tn_after = compute_counts(pred_after_gate, target)

        target_pixels = float(target.sum())
        total_pixels = float(target.size)
        pred_pixels_before = float(pred_before_gate.sum())
        pred_pixels_after = float(pred_after_gate.sum())

        target_ratio = target_pixels / total_pixels if total_pixels > 0 else 0.0
        pred_ratio_before = pred_pixels_before / total_pixels if total_pixels > 0 else 0.0
        pred_ratio_after = pred_pixels_after / total_pixels if total_pixels > 0 else 0.0

        row = {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "prob_non_empty": prob_non_empty,
            "pass_gate": bool(pass_gate),
            "target_pixels": target_pixels,
            "total_pixels": total_pixels,
            "pred_pixels_before_gate": pred_pixels_before,
            "pred_pixels_after_gate": pred_pixels_after,
            "target_ratio": target_ratio,
            "pred_ratio_before_gate": pred_ratio_before,
            "pred_ratio_after_gate": pred_ratio_after,
            "mask_size_class": classify_mask_size(target_ratio),
        }

        add_metric_columns(
            row,
            prefix="before_gate",
            tp=tp_before,
            fp=fp_before,
            fn=fn_before,
            tn=tn_before,
        )

        add_metric_columns(
            row,
            prefix="after_gate",
            tp=tp_after,
            fp=fp_after,
            fn=fn_after,
            tn=tn_after,
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    per_sample_path = args.output_dir / "per_sample_metrics.csv"
    df.to_csv(per_sample_path, index=False)

    summary = {
        "split": args.split,
        "num_samples": int(len(df)),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "segmenter_checkpoint": str(args.segmenter_checkpoint),
        "classifier_threshold": args.classifier_threshold,
        "segmentation_threshold": args.segmentation_threshold,
        "before_gate": make_summary(
            df,
            prefix="before_gate",
            classifier_threshold=args.classifier_threshold,
            segmentation_threshold=args.segmentation_threshold,
        ),
        "after_gate": make_summary(
            df,
            prefix="after_gate",
            classifier_threshold=args.classifier_threshold,
            segmentation_threshold=args.segmentation_threshold,
        ),
    }

    summary_path = args.output_dir / "summary_metrics.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_by_size_before = make_summary_by_size(df, prefix="before_gate")
    summary_by_size_after = make_summary_by_size(df, prefix="after_gate")

    before_path = args.output_dir / "summary_by_mask_size_before_gate.csv"
    after_path = args.output_dir / "summary_by_mask_size_after_gate.csv"

    summary_by_size_before.to_csv(before_path, index=False)
    summary_by_size_after.to_csv(after_path, index=False)

    print("\nResumo BEFORE GATE:")
    print("dice", summary["before_gate"]["dice"])
    print("iou", summary["before_gate"]["iou"])
    print("precision", summary["before_gate"]["precision"])
    print("recall", summary["before_gate"]["recall"])
    print("f1", summary["before_gate"]["f1"])

    print("\nResumo AFTER GATE:")
    print("dice", summary["after_gate"]["dice"])
    print("iou", summary["after_gate"]["iou"])
    print("precision", summary["after_gate"]["precision"])
    print("recall", summary["after_gate"]["recall"])
    print("f1", summary["after_gate"]["f1"])

    print("\nResumo operacional AFTER GATE:")
    print("empty_count:", summary["after_gate"]["empty_count"])
    print("non_empty_count:", summary["after_gate"]["non_empty_count"])
    print("empty_false_positive_rate:", summary["after_gate"].get("empty_false_positive_rate"))
    print("empty_pred_pixels_mean:", summary["after_gate"].get("empty_pred_pixels_mean"))
    print("empty_pred_pixels_median:", summary["after_gate"].get("empty_pred_pixels_median"))
    print("non_empty_blocked_rate:", summary["after_gate"].get("non_empty_blocked_rate"))
    print("non_empty_kept_rate:", summary["after_gate"].get("non_empty_kept_rate"))

    print("\nArquivos salvos em:", args.output_dir)
    print("-", per_sample_path)
    print("-", summary_path)
    print("-", before_path)
    print("-", after_path)


if __name__ == "__main__":
    main()

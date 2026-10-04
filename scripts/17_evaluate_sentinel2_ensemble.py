from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp


def normalize_percentile(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, [p_low, p_high])
    if np.isclose(lo, hi):
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def build_pairs(root: Path, split: str):
    image_dir = root / "images" / split
    mask_dir = root / "masks" / split
    csv_path = root / "metadata" / f"{split}.csv"

    if csv_path.exists():
        names = pd.read_csv(csv_path)["image"].astype(str).tolist()
    else:
        names = [p.name for p in sorted(image_dir.glob("*.nc"))]

    pairs = []
    for name in names:
        image_path = image_dir / name
        mask_path = mask_dir / name.replace(".nc", ".tif")
        if image_path.exists() and mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


class Sentinel2Dataset(Dataset):
    def __init__(self, pairs, bands, time_mode, time_index, image_size, positive_values=None):
        self.pairs = pairs
        self.bands = bands
        self.time_mode = time_mode
        self.time_index = time_index
        self.image_size = image_size
        self.positive_values = positive_values

    def __len__(self):
        return len(self.pairs)

    def _read_image(self, image_path: Path):
        ds = xr.open_dataset(image_path)
        channels = []
        try:
            for band in self.bands:
                data = ds[band]
                if "time" in data.dims:
                    if self.time_mode == "median":
                        channels.append(normalize_percentile(data.median(dim="time", skipna=True).values))
                    elif self.time_mode == "mean":
                        channels.append(normalize_percentile(data.mean(dim="time", skipna=True).values))
                    elif self.time_mode == "index":
                        channels.append(normalize_percentile(data.isel(time=self.time_index).values))
                    elif self.time_mode == "stack":
                        values = data.values
                        for t in range(values.shape[0]):
                            channels.append(normalize_percentile(values[t]))
                    else:
                        raise ValueError(f"time_mode inválido: {self.time_mode}")
                else:
                    channels.append(normalize_percentile(data.values))
        finally:
            ds.close()
        return np.stack(channels, axis=0).astype(np.float32)

    def _read_mask(self, mask_path: Path):
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        if self.positive_values is None:
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.isin(mask, self.positive_values).astype(np.float32)
        return mask[None, :, :]

    def _resize(self, image, mask):
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image, mask
        image = F.interpolate(image[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask[None], size=(self.image_size, self.image_size), mode="nearest")[0]
        return image, mask

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = torch.from_numpy(self._read_image(image_path)).float()
        mask = torch.from_numpy(self._read_mask(mask_path)).float()
        image, mask = self._resize(image, mask)
        return {"image": image, "mask": mask, "image_path": str(image_path), "mask_path": str(mask_path)}


def build_model(architecture: str, encoder_name: str, in_channels: int) -> nn.Module:
    if architecture == "unet":
        return smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    raise ValueError(f"Arquitetura inválida: {architecture}")


def load_model(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = build_model(
        architecture=config.get("architecture", "unet"),
        encoder_name=config.get("encoder_name", "resnet34"),
        in_channels=int(config.get("in_channels", 30)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def metrics_from_probs(probs, masks, threshold, eps=1e-7):
    preds = (probs >= threshold).float()
    p = preds.flatten(1)
    y = masks.flatten(1)

    tp = (p * y).sum(1)
    fp = (p * (1 - y)).sum(1)
    fn = ((1 - p) * y).sum(1)
    tn = ((1 - p) * (1 - y)).sum(1)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    return {
        "dice": dice.cpu().numpy(),
        "iou": iou.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy(),
        "tp": tp.cpu().numpy(),
        "tn": tn.cpu().numpy(),
        "fp": fp.cpu().numpy(),
        "fn": fn.cpu().numpy(),
        "pred_pixels": p.sum(1).cpu().numpy(),
        "target_pixels": y.sum(1).cpu().numpy(),
        "total_pixels": np.full(len(dice), y.shape[1], dtype=np.float32),
    }


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def mask_size_class(ratio):
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


@torch.no_grad()
def evaluate(model_a, model_b, loader, device, threshold, weight_a, amp, amp_dtype):
    rows = []
    model_a.eval()
    model_b.eval()

    for batch in tqdm(loader, desc="Avaliando ensemble"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            prob_a = torch.sigmoid(model_a(images))
            prob_b = torch.sigmoid(model_b(images))
            probs = weight_a * prob_a + (1.0 - weight_a) * prob_b

        m = metrics_from_probs(probs, masks, threshold)

        for i in range(images.shape[0]):
            target_ratio = float(m["target_pixels"][i] / m["total_pixels"][i])
            rows.append({
                "image_path": batch["image_path"][i],
                "mask_path": batch["mask_path"][i],
                "dice": float(m["dice"][i]),
                "iou": float(m["iou"][i]),
                "precision": float(m["precision"][i]),
                "recall": float(m["recall"][i]),
                "f1": float(m["f1"][i]),
                "tp": float(m["tp"][i]),
                "tn": float(m["tn"][i]),
                "fp": float(m["fp"][i]),
                "fn": float(m["fn"][i]),
                "pred_pixels": float(m["pred_pixels"][i]),
                "target_pixels": float(m["target_pixels"][i]),
                "total_pixels": float(m["total_pixels"][i]),
                "target_ratio": target_ratio,
                "pred_ratio": float(m["pred_pixels"][i] / m["total_pixels"][i]),
                "mask_size_class": mask_size_class(target_ratio),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Avalia ensemble U-Net + DeepLabV3+ no Sentinel-2 stack30.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/ai4boundaries/sentinel2"))
    parser.add_argument("--checkpoint-a", type=Path, required=True, help="Normalmente U-Net.")
    parser.add_argument("--checkpoint-b", type=Path, required=True, help="Normalmente DeepLabV3+.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--weight-a", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, config_a = load_model(args.checkpoint_a, device)
    model_b, config_b = load_model(args.checkpoint_b, device)

    bands_a = config_a.get("bands", ["B2", "B3", "B4", "B8", "NDVI"])
    bands_b = config_b.get("bands", ["B2", "B3", "B4", "B8", "NDVI"])
    time_mode_a = config_a.get("time_mode", "stack")
    time_mode_b = config_b.get("time_mode", "stack")

    if bands_a != bands_b:
        raise ValueError(f"Bandas diferentes: {bands_a} vs {bands_b}")
    if time_mode_a != time_mode_b:
        raise ValueError(f"time_mode diferente: {time_mode_a} vs {time_mode_b}")

    pairs = build_pairs(args.root, args.split)
    if args.max_samples is not None:
        pairs = pairs[:args.max_samples]

    dataset = Sentinel2Dataset(
        pairs=pairs,
        bands=bands_a,
        time_mode=time_mode_a,
        time_index=int(config_a.get("time_index", -1)),
        image_size=int(config_a.get("image_size", 256)),
        positive_values=config_a.get("positive_values", None),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    print("Configuração ensemble:")
    print(f"Split: {args.split}")
    print(f"Checkpoint A: {args.checkpoint_a}")
    print(f"Checkpoint B: {args.checkpoint_b}")
    print(f"Peso A: {args.weight_a:.2f}")
    print(f"Peso B: {1.0 - args.weight_a:.2f}")
    print(f"Threshold: {args.threshold}")
    print(f"Pares: {len(pairs)}")
    print(f"Bands: {bands_a}")
    print(f"Time mode: {time_mode_a}")

    df = evaluate(
        model_a=model_a,
        model_b=model_b,
        loader=loader,
        device=device,
        threshold=args.threshold,
        weight_a=args.weight_a,
        amp=args.amp,
        amp_dtype=amp_dtype,
    )

    per_sample_path = args.output_dir / "per_sample_metrics.csv"
    df.to_csv(per_sample_path, index=False)

    summary = {
        "split": args.split,
        "checkpoint_a": str(args.checkpoint_a),
        "checkpoint_b": str(args.checkpoint_b),
        "threshold": args.threshold,
        "weight_a": args.weight_a,
        "weight_b": 1.0 - args.weight_a,
        "num_samples": int(len(df)),
        "dice": summarize(df["dice"].values),
        "iou": summarize(df["iou"].values),
        "precision": summarize(df["precision"].values),
        "recall": summarize(df["recall"].values),
        "f1": summarize(df["f1"].values),
    }

    summary_path = args.output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    by_size = (
        df.groupby("mask_size_class")
        .agg(
            count=("dice", "size"),
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
    by_size.to_csv(by_size_path, index=False)

    print("\nResumo final:")
    for metric in ["dice", "iou", "precision", "recall", "f1"]:
        print(metric, summary[metric])

    print("\nPor tamanho de máscara:")
    print(by_size.to_string(index=False))

    print("\nArquivos salvos:")
    print("-", per_sample_path)
    print("-", summary_path)
    print("-", by_size_path)


if __name__ == "__main__":
    main()

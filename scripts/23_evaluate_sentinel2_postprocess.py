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
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp


def normalize_percentile(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, [2, 98])
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

    return [
        (image_dir / name, mask_dir / name.replace(".nc", ".tif"))
        for name in names
        if (image_dir / name).exists() and (mask_dir / name.replace(".nc", ".tif")).exists()
    ]


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
                    if self.time_mode == "stack":
                        values = data.values
                        for t in range(values.shape[0]):
                            channels.append(normalize_percentile(values[t]))
                    elif self.time_mode == "median":
                        channels.append(normalize_percentile(data.median(dim="time", skipna=True).values))
                    elif self.time_mode == "mean":
                        channels.append(normalize_percentile(data.mean(dim="time", skipna=True).values))
                    elif self.time_mode == "index":
                        channels.append(normalize_percentile(data.isel(time=self.time_index).values))
                    else:
                        raise ValueError(self.time_mode)
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



def build_model(architecture: str, encoder_name: str, encoder_weights=None, in_channels=None) -> nn.Module:
    """
    Função compatível com scripts que chamam:
    - build_model(architecture, encoder_name, in_channels)
    - build_model(architecture, encoder_name, encoder_weights, in_channels)
    """

    # Compatibilidade com chamadas antigas:
    # build_model(architecture, encoder_name, in_channels)
    if in_channels is None:
        in_channels = encoder_weights
        encoder_weights = None

    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None

    architecture = architecture.lower()

    if architecture == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,
        )

    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,
        )

    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,
        )

    if architecture == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,
        )

    if architecture == "pan":
        return smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,
        )

    raise ValueError(f"Arquitetura inválida: {architecture}")


def load_model(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = build_model(
        config.get("architecture", "unet"),
        config.get("encoder_name", "resnet34"),
        int(config.get("in_channels", 30)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def parse_weights(text: str, n_models: int):
    weights = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(weights) != n_models:
        raise ValueError(f"--weights precisa ter {n_models} valores. Recebido: {weights}")
    total = sum(weights)
    return [w / total for w in weights]


def apply_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "id":
        return x
    if name == "h":
        return torch.flip(x, dims=[3])
    if name == "v":
        return torch.flip(x, dims=[2])
    if name == "hv":
        return torch.flip(x, dims=[2, 3])
    raise ValueError(name)


def invert_transform(y: torch.Tensor, name: str) -> torch.Tensor:
    if name == "id":
        return y
    if name == "h":
        return torch.flip(y, dims=[3])
    if name == "v":
        return torch.flip(y, dims=[2])
    if name == "hv":
        return torch.flip(y, dims=[2, 3])
    raise ValueError(name)


def get_tta_list(mode: str):
    if mode == "none":
        return ["id"]
    if mode == "flips":
        return ["id", "h", "v", "hv"]
    raise ValueError("Este script simples aceita --tta-mode none ou flips.")


def remove_small_components(mask: np.ndarray, min_area: int, connectivity: int = 8) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)

    structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )

    labels, n_labels = ndimage.label(mask.astype(bool), structure=structure)

    if n_labels == 0:
        return mask.astype(np.uint8)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False

    return keep[labels].astype(np.uint8)


def maybe_fill_holes(mask: np.ndarray, fill_holes: bool) -> np.ndarray:
    if not fill_holes:
        return mask.astype(np.uint8)
    return ndimage.binary_fill_holes(mask.astype(bool)).astype(np.uint8)


def metrics_from_binary(preds_np: np.ndarray, masks_np: np.ndarray, eps: float = 1e-7):
    p = preds_np.reshape(preds_np.shape[0], -1).astype(np.float32)
    y = masks_np.reshape(masks_np.shape[0], -1).astype(np.float32)

    tp = (p * y).sum(axis=1)
    fp = (p * (1 - y)).sum(axis=1)
    fn = ((1 - p) * y).sum(axis=1)
    tn = ((1 - p) * (1 - y)).sum(axis=1)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_pixels": p.sum(axis=1),
        "target_pixels": y.sum(axis=1),
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


def make_summary(df: pd.DataFrame, extra: dict):
    summary = dict(extra)
    summary["num_samples"] = int(len(df))
    for metric in ["dice", "iou", "precision", "recall", "f1"]:
        summary[metric] = summarize(df[metric].values)

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

    return summary, by_size


@torch.no_grad()
def evaluate(models, weights, loader, device, threshold, amp, amp_dtype, tta_list, min_area_values, fill_holes, connectivity):
    for model in models:
        model.eval()

    rows_by_min_area = {min_area: [] for min_area in min_area_values}

    for batch in tqdm(loader, desc=f"Avaliando min_area {min_area_values}"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        probs_sum = None

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            for tta in tta_list:
                x_t = apply_transform(images, tta)

                probs_t = None
                for weight, model in zip(weights, models):
                    p = torch.sigmoid(model(x_t))
                    probs_t = weight * p if probs_t is None else probs_t + weight * p

                probs_t = invert_transform(probs_t, tta)
                probs_sum = probs_t if probs_sum is None else probs_sum + probs_t

            probs = probs_sum / float(len(tta_list))

        raw_preds = (probs >= threshold).float().detach().cpu().numpy().astype(np.uint8)
        masks_np = masks.detach().cpu().numpy().astype(np.uint8)

        for min_area in min_area_values:
            processed = np.zeros_like(raw_preds, dtype=np.uint8)

            for i in range(raw_preds.shape[0]):
                m = raw_preds[i, 0]
                m = remove_small_components(m, min_area=min_area, connectivity=connectivity)
                m = maybe_fill_holes(m, fill_holes=fill_holes)
                processed[i, 0] = m

            metric = metrics_from_binary(processed, masks_np)

            for i in range(images.shape[0]):
                target_ratio = float(metric["target_pixels"][i] / metric["total_pixels"][i])
                rows_by_min_area[min_area].append(
                    {
                        "image_path": batch["image_path"][i],
                        "mask_path": batch["mask_path"][i],
                        "dice": float(metric["dice"][i]),
                        "iou": float(metric["iou"][i]),
                        "precision": float(metric["precision"][i]),
                        "recall": float(metric["recall"][i]),
                        "f1": float(metric["f1"][i]),
                        "tp": float(metric["tp"][i]),
                        "tn": float(metric["tn"][i]),
                        "fp": float(metric["fp"][i]),
                        "fn": float(metric["fn"][i]),
                        "pred_pixels": float(metric["pred_pixels"][i]),
                        "target_pixels": float(metric["target_pixels"][i]),
                        "total_pixels": float(metric["total_pixels"][i]),
                        "target_ratio": target_ratio,
                        "pred_ratio": float(metric["pred_pixels"][i] / metric["total_pixels"][i]),
                        "mask_size_class": mask_size_class(target_ratio),
                    }
                )

    return {min_area: pd.DataFrame(rows) for min_area, rows in rows_by_min_area.items()}


def main():
    parser = argparse.ArgumentParser(description="Avalia ensemble + TTA + remoção de componentes pequenos.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/ai4boundaries/sentinel2"))
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--tta-mode", type=str, default="flips", choices=["none", "flips"])
    parser.add_argument("--min-area-values", type=str, default="0,3,5,10,20,30,50,75,100")
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, configs = [], []

    for ckpt_path in args.checkpoints:
        model, config = load_model(ckpt_path, device)
        models.append(model)
        configs.append(config)

    weights = parse_weights(args.weights, len(models))
    tta_list = get_tta_list(args.tta_mode)
    min_area_values = [int(x.strip()) for x in args.min_area_values.split(",") if x.strip()]

    bands = configs[0].get("bands", ["B2", "B3", "B4", "B8", "NDVI"])
    time_mode = configs[0].get("time_mode", "stack")
    image_size = int(configs[0].get("image_size", 256))
    time_index = int(configs[0].get("time_index", -1))
    positive_values = configs[0].get("positive_values", None)

    pairs = build_pairs(args.root, args.split)
    dataset = Sentinel2Dataset(pairs, bands, time_mode, time_index, image_size, positive_values)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    print("Configuração pós-processamento:")
    print({
        "n_models": len(models),
        "weights": weights,
        "threshold": args.threshold,
        "tta": tta_list,
        "min_area_values": min_area_values,
        "fill_holes": args.fill_holes,
        "connectivity": args.connectivity,
        "n_samples": len(dataset),
    })

    dfs = evaluate(
        models=models,
        weights=weights,
        loader=loader,
        device=device,
        threshold=args.threshold,
        amp=args.amp,
        amp_dtype=amp_dtype,
        tta_list=tta_list,
        min_area_values=min_area_values,
        fill_holes=args.fill_holes,
        connectivity=args.connectivity,
    )

    ranking_rows = []

    for min_area, df in dfs.items():
        run_dir = args.output_dir / f"min_area_{min_area}"
        run_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(run_dir / "per_sample_metrics.csv", index=False)

        summary, by_size = make_summary(
            df,
            {
                "split": args.split,
                "weights": weights,
                "threshold": args.threshold,
                "tta_mode": args.tta_mode,
                "tta_list": tta_list,
                "min_area": min_area,
                "fill_holes": args.fill_holes,
                "connectivity": args.connectivity,
            },
        )

        (run_dir / "summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        by_size.to_csv(run_dir / "summary_by_mask_size.csv", index=False)

        ranking_rows.append(
            {
                "min_area": min_area,
                "dice": summary["dice"]["mean"],
                "iou": summary["iou"]["mean"],
                "precision": summary["precision"]["mean"],
                "recall": summary["recall"]["mean"],
                "f1": summary["f1"]["mean"],
                "run_dir": str(run_dir),
            }
        )

    ranking = pd.DataFrame(ranking_rows).sort_values(["dice", "f1", "iou"], ascending=False)
    ranking.to_csv(args.output_dir / "postprocess_ranking.csv", index=False)

    print("\nRanking pós-processamento:")
    print(ranking.to_string(index=False))
    print("\nArquivos salvos em:", args.output_dir)


if __name__ == "__main__":
    main()

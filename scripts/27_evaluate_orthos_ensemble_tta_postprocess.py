from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import segmentation_models_pytorch as smp


def image_to_mask_path(image_path: Path, root: Path, split: str) -> Path:
    mask_name = image_path.name.replace("_ortho_", "_ortholabel_")
    return root / "masks" / split / mask_name


def build_pairs(root: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = root / "images" / split
    pairs: List[Tuple[Path, Path]] = []

    for image_path in sorted(image_dir.glob("*.tif")):
        mask_path = image_to_mask_path(image_path, root, split)
        if image_path.exists() and mask_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


def size_class_from_ratio(ratio: float) -> str:
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


class OrthoDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        image_size: int = 512,
        mask_band: int = 1,
        mask_mode: str = "band",
        normalization: str = "imagenet",
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.mask_band = mask_band
        self.mask_mode = mask_mode
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_image(self, image_path: Path) -> np.ndarray:
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3]).astype(np.float32)

        if image.max() > 2.0:
            image = image / 255.0

        if self.normalization == "imagenet":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            image = (image - mean) / std
        elif self.normalization == "none":
            pass
        else:
            raise ValueError(f"normalization inválida: {self.normalization}")

        return image.astype(np.float32)

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        with rasterio.open(mask_path) as src:
            if self.mask_mode == "band":
                mask = src.read(self.mask_band)
                mask = (mask > 0).astype(np.float32)
            elif self.mask_mode == "any":
                arr = src.read()
                mask = (arr > 0).any(axis=0).astype(np.float32)
            else:
                raise ValueError(f"mask_mode inválido: {self.mask_mode}")

        return mask[None, :, :].astype(np.float32)

    def _resize(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image, mask

        image = F.interpolate(
            image[None],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0]

        mask = F.interpolate(
            mask[None],
            size=(self.image_size, self.image_size),
            mode="nearest",
        )[0]

        return image, mask

    def __getitem__(self, idx: int) -> Dict:
        image_path, mask_path = self.pairs[idx]

        image = torch.from_numpy(self._read_image(image_path)).float()
        mask = torch.from_numpy(self._read_mask(mask_path)).float()
        image, mask = self._resize(image, mask)

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def build_model(architecture: str, encoder_name: str, in_channels: int) -> nn.Module:
    architecture = str(architecture).lower()

    if architecture == "unet":
        return smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    if architecture == "fpn":
        return smp.FPN(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)
    if architecture == "pan":
        return smp.PAN(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1)

    raise ValueError(f"Arquitetura inválida: {architecture}")


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = build_model(
        architecture=config.get("architecture", "unet"),
        encoder_name=config.get("encoder_name", "resnet34"),
        in_channels=int(config.get("in_channels", 3)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config


def parse_weights(text: str, n_models: int) -> List[float]:
    weights = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(weights) != n_models:
        raise ValueError(f"Esperado {n_models} pesos, recebido: {weights}")
    total = sum(weights)
    if total <= 0:
        raise ValueError("A soma dos pesos precisa ser positiva.")
    return [w / total for w in weights]


def get_tta_list(mode: str) -> List[str]:
    if mode == "none":
        return ["id"]
    if mode == "flips":
        return ["id", "h", "v", "hv"]
    if mode == "d4":
        return ["id", "h", "v", "hv", "r90", "r180", "r270"]
    raise ValueError(f"tta-mode inválido: {mode}")


def apply_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "id": return x
    if name == "h": return torch.flip(x, dims=[3])
    if name == "v": return torch.flip(x, dims=[2])
    if name == "hv": return torch.flip(x, dims=[2, 3])
    if name == "r90": return torch.rot90(x, k=1, dims=[2, 3])
    if name == "r180": return torch.rot90(x, k=2, dims=[2, 3])
    if name == "r270": return torch.rot90(x, k=3, dims=[2, 3])
    raise ValueError(f"TTA inválido: {name}")


def invert_transform(y: torch.Tensor, name: str) -> torch.Tensor:
    if name == "id": return y
    if name == "h": return torch.flip(y, dims=[3])
    if name == "v": return torch.flip(y, dims=[2])
    if name == "hv": return torch.flip(y, dims=[2, 3])
    if name == "r90": return torch.rot90(y, k=3, dims=[2, 3])
    if name == "r180": return torch.rot90(y, k=2, dims=[2, 3])
    if name == "r270": return torch.rot90(y, k=1, dims=[2, 3])
    raise ValueError(f"TTA inválido: {name}")


def remove_small_components(mask: np.ndarray, min_area: int, connectivity: int = 8) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.uint8)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    labels, n_labels = ndimage.label(mask.astype(bool), structure=structure)
    if n_labels == 0:
        return mask.astype(np.uint8)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8)


def metrics_from_binary(preds_np: np.ndarray, masks_np: np.ndarray, eps: float = 1e-7) -> Dict[str, np.ndarray]:
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
        "dice": dice, "iou": iou, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "pred_pixels": p.sum(axis=1), "target_pixels": y.sum(axis=1),
        "total_pixels": np.full(len(dice), y.shape[1], dtype=np.float32),
    }


def summarize(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}


@torch.no_grad()
def evaluate_ensemble(models, weights, loader, device, threshold, amp, amp_dtype, tta_list, min_area_values, connectivity):
    for model in models:
        model.eval()

    rows_by_area = {area: [] for area in min_area_values}

    for batch in tqdm(loader, desc="Avaliando ensemble TTA/min_area"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        probs_sum = None

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            for tta in tta_list:
                images_t = apply_transform(images, tta)

                probs_t = None
                for weight, model in zip(weights, models):
                    p = torch.sigmoid(model(images_t))
                    probs_t = weight * p if probs_t is None else probs_t + weight * p

                probs_t = invert_transform(probs_t, tta)
                probs_sum = probs_t if probs_sum is None else probs_sum + probs_t

            probs = probs_sum / float(len(tta_list))

        raw_preds = (probs >= threshold).detach().cpu().numpy().astype(np.uint8)
        masks_np = masks.detach().cpu().numpy().astype(np.uint8)

        for min_area in min_area_values:
            processed = np.zeros_like(raw_preds, dtype=np.uint8)
            for i in range(raw_preds.shape[0]):
                processed[i, 0] = remove_small_components(raw_preds[i, 0], min_area=min_area, connectivity=connectivity)

            metrics = metrics_from_binary(processed, masks_np)

            for i in range(images.shape[0]):
                target_ratio = float(metrics["target_pixels"][i] / metrics["total_pixels"][i])
                rows_by_area[min_area].append({
                    "image_path": batch["image_path"][i],
                    "mask_path": batch["mask_path"][i],
                    "dice": float(metrics["dice"][i]),
                    "iou": float(metrics["iou"][i]),
                    "precision": float(metrics["precision"][i]),
                    "recall": float(metrics["recall"][i]),
                    "f1": float(metrics["f1"][i]),
                    "tp": float(metrics["tp"][i]),
                    "tn": float(metrics["tn"][i]),
                    "fp": float(metrics["fp"][i]),
                    "fn": float(metrics["fn"][i]),
                    "pred_pixels": float(metrics["pred_pixels"][i]),
                    "target_pixels": float(metrics["target_pixels"][i]),
                    "total_pixels": float(metrics["total_pixels"][i]),
                    "target_ratio": target_ratio,
                    "pred_ratio": float(metrics["pred_pixels"][i] / metrics["total_pixels"][i]),
                    "mask_size_class": size_class_from_ratio(target_ratio),
                })

    return {area: pd.DataFrame(rows) for area, rows in rows_by_area.items()}


def main():
    parser = argparse.ArgumentParser(description="Avalia ensemble Ortho com TTA e pós-processamento.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/ai4boundaries/orthos"))
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.66)
    parser.add_argument("--tta-mode", type=str, default="flips", choices=["none", "flips", "d4"])
    parser.add_argument("--min-area-values", type=str, default="900,1000,1200,1500,1800,2000")
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    configs = []
    for checkpoint in args.checkpoints:
        model, config = load_model(checkpoint, device)
        models.append(model)
        configs.append(config)

    weights = parse_weights(args.weights, len(models))

    config0 = configs[0]
    image_size = int(config0.get("image_size", 512))
    mask_band = int(config0.get("mask_band", 1))
    mask_mode = config0.get("mask_mode", "band")
    normalization = config0.get("normalization", "imagenet")

    pairs = build_pairs(args.root, args.split)
    dataset = OrthoDataset(pairs, image_size=image_size, mask_band=mask_band, mask_mode=mask_mode, normalization=normalization)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    tta_list = get_tta_list(args.tta_mode)
    min_area_values = [int(x.strip()) for x in args.min_area_values.split(",") if x.strip()]
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    print("Configuração ensemble orthos:")
    print({
        "checkpoints": [str(p) for p in args.checkpoints],
        "weights": weights,
        "threshold": args.threshold,
        "tta": tta_list,
        "min_area_values": min_area_values,
        "connectivity": args.connectivity,
        "image_size": image_size,
        "mask_band": mask_band,
        "mask_mode": mask_mode,
        "normalization": normalization,
        "pairs": len(pairs),
        "num_workers": args.num_workers,
    })

    dfs = evaluate_ensemble(models, weights, loader, device, args.threshold, args.amp, amp_dtype, tta_list, min_area_values, args.connectivity)

    ranking_rows = []
    for min_area, df in dfs.items():
        run_dir = args.output_dir / f"min_area_{min_area}"
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(run_dir / "per_sample_metrics.csv", index=False)

        summary = {
            "split": args.split,
            "checkpoints": [str(p) for p in args.checkpoints],
            "weights": weights,
            "threshold": args.threshold,
            "tta_mode": args.tta_mode,
            "tta_list": tta_list,
            "min_area": min_area,
            "connectivity": args.connectivity,
            "num_samples": int(len(df)),
        }

        for metric in ["dice", "iou", "precision", "recall", "f1"]:
            summary[metric] = summarize(df[metric].values)

        with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        by_size = df.groupby("mask_size_class").agg(
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
        ).reset_index()
        by_size.to_csv(run_dir / "summary_by_mask_size.csv", index=False)

        ranking_rows.append({
            "min_area": min_area,
            "dice": summary["dice"]["mean"],
            "iou": summary["iou"]["mean"],
            "precision": summary["precision"]["mean"],
            "recall": summary["recall"]["mean"],
            "f1": summary["f1"]["mean"],
            "run_dir": str(run_dir),
        })

    ranking = pd.DataFrame(ranking_rows).sort_values(["dice", "f1", "iou"], ascending=False)
    ranking.to_csv(args.output_dir / "postprocess_ranking.csv", index=False)

    print("\nRanking pós-processamento:")
    print(ranking.to_string(index=False))
    print("\nArquivos salvos em:", args.output_dir)


if __name__ == "__main__":
    main()

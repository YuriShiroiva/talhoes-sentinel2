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
    pairs = []

    for image_path in sorted(image_dir.glob("*.tif")):
        mask_path = image_to_mask_path(image_path, root, split)
        if image_path.exists() and mask_path.exists():
            pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"Nenhum par encontrado para split={split} em {image_dir}")

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
    if ratio < 0.90:
        return "very_large_75_90%"
    return "almost_full_90_100%"


class OrthoBoundaryEvalDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        image_size: int = 512,
        area_band: int = 1,
        boundary_band: int = 2,
        normalization: str = "imagenet",
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.area_band = area_band
        self.boundary_band = boundary_band
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_image(self, image_path: Path) -> np.ndarray:
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3]).astype(np.float32)

        if image.max() > 2.0:
            image /= 255.0

        image = np.clip(image, 0.0, 1.0)

        if self.normalization == "imagenet":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            image = (image - mean) / std
        elif self.normalization == "none":
            pass
        else:
            raise ValueError(f"normalization inválida: {self.normalization}")

        return image.astype(np.float32)

    def _read_targets(self, mask_path: Path) -> np.ndarray:
        with rasterio.open(mask_path) as src:
            area = (src.read(self.area_band) > 0).astype(np.float32)
            boundary = (src.read(self.boundary_band) > 0).astype(np.float32)

        return np.stack([area, boundary], axis=0).astype(np.float32)

    def _resize(self, image: torch.Tensor, target: torch.Tensor):
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image, target

        image = F.interpolate(
            image[None],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0]

        target = F.interpolate(
            target[None],
            size=(self.image_size, self.image_size),
            mode="nearest",
        )[0]

        return image, target

    def __getitem__(self, idx: int) -> Dict:
        image_path, mask_path = self.pairs[idx]

        image = torch.from_numpy(self._read_image(image_path)).float()
        target = torch.from_numpy(self._read_targets(mask_path)).float()
        image, target = self._resize(image, target)

        return {
            "image": image,
            "target": target,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def build_model(architecture: str, encoder_name: str, in_channels: int, classes: int) -> nn.Module:
    architecture = architecture.lower()

    if architecture == "unet":
        return smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)

    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)

    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)

    if architecture == "fpn":
        return smp.FPN(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)

    raise ValueError(f"Arquitetura inválida: {architecture}")


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = build_model(
        architecture=cfg.get("architecture", "unetplusplus"),
        encoder_name=cfg.get("encoder_name", "resnet50"),
        in_channels=int(cfg.get("in_channels", 3)),
        classes=int(cfg.get("classes", 2)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, cfg, ckpt


def get_tta_list(mode: str) -> List[str]:
    if mode == "none":
        return ["id"]
    if mode == "flips":
        return ["id", "h", "v", "hv"]
    if mode == "d4":
        return ["id", "h", "v", "hv", "r90", "r180", "r270"]
    raise ValueError(f"tta-mode inválido: {mode}")


def apply_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "id":
        return x
    if name == "h":
        return torch.flip(x, dims=[3])
    if name == "v":
        return torch.flip(x, dims=[2])
    if name == "hv":
        return torch.flip(x, dims=[2, 3])
    if name == "r90":
        return torch.rot90(x, k=1, dims=[2, 3])
    if name == "r180":
        return torch.rot90(x, k=2, dims=[2, 3])
    if name == "r270":
        return torch.rot90(x, k=3, dims=[2, 3])
    raise ValueError(f"TTA inválido: {name}")


def invert_transform(y: torch.Tensor, name: str) -> torch.Tensor:
    if name == "id":
        return y
    if name == "h":
        return torch.flip(y, dims=[3])
    if name == "v":
        return torch.flip(y, dims=[2])
    if name == "hv":
        return torch.flip(y, dims=[2, 3])
    if name == "r90":
        return torch.rot90(y, k=3, dims=[2, 3])
    if name == "r180":
        return torch.rot90(y, k=2, dims=[2, 3])
    if name == "r270":
        return torch.rot90(y, k=1, dims=[2, 3])
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


def binary_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> Dict[str, float]:
    p = pred.reshape(-1).astype(np.float32)
    y = target.reshape(-1).astype(np.float32)

    tp = float((p * y).sum())
    fp = float((p * (1 - y)).sum())
    fn = float(((1 - p) * y).sum())
    tn = float(((1 - p) * (1 - y)).sum())

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    total_pixels = float(p.size)
    pred_pixels = float(p.sum())
    target_pixels = float(y.sum())

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_pixels": pred_pixels,
        "target_pixels": target_pixels,
        "total_pixels": total_pixels,
        "target_ratio": target_pixels / total_pixels,
        "pred_ratio": pred_pixels / total_pixels,
    }


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, amp_dtype: torch.dtype, tta_list: List[str]):
    rows = []

    for batch in tqdm(loader, desc="Inferindo B01"):
        images = batch["image"].to(device, non_blocking=True)
        prob_sum = None

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            for tta in tta_list:
                x_t = apply_transform(images, tta)
                logits_t = model(x_t)
                probs_t = torch.sigmoid(logits_t)
                probs_t = invert_transform(probs_t, tta)
                prob_sum = probs_t if prob_sum is None else prob_sum + probs_t

            probs = prob_sum / float(len(tta_list))

        probs_np = probs.float().detach().cpu().numpy().astype(np.float32)
        targets_np = batch["target"].numpy().astype(np.uint8)

        for i in range(probs_np.shape[0]):
            rows.append({
                "image_path": batch["image_path"][i],
                "mask_path": batch["mask_path"][i],
                "area_prob": probs_np[i, 0],
                "boundary_prob": probs_np[i, 1],
                "area_target": targets_np[i, 0],
                "boundary_target": targets_np[i, 1],
            })

    return rows


def evaluate_predictions(payload: List[Dict], area_threshold: float, boundary_threshold: float, min_area: int, connectivity: int) -> tuple[pd.DataFrame, Dict]:
    rows = []

    for item in payload:
        area_raw = (item["area_prob"] >= area_threshold).astype(np.uint8)
        area_pred = remove_small_components(area_raw, min_area=min_area, connectivity=connectivity)
        boundary_pred = (item["boundary_prob"] >= boundary_threshold).astype(np.uint8)

        area_metric = binary_metrics(area_pred, item["area_target"])
        boundary_metric = binary_metrics(boundary_pred, item["boundary_target"])

        target_ratio = area_metric["target_ratio"]

        rows.append({
            "image_path": item["image_path"],
            "mask_path": item["mask_path"],
            "area_threshold": area_threshold,
            "boundary_threshold": boundary_threshold,
            "min_area": min_area,
            "area_dice": area_metric["dice"],
            "area_iou": area_metric["iou"],
            "area_precision": area_metric["precision"],
            "area_recall": area_metric["recall"],
            "area_f1": area_metric["f1"],
            "boundary_dice": boundary_metric["dice"],
            "boundary_iou": boundary_metric["iou"],
            "boundary_precision": boundary_metric["precision"],
            "boundary_recall": boundary_metric["recall"],
            "boundary_f1": boundary_metric["f1"],
            "target_pixels": area_metric["target_pixels"],
            "pred_pixels": area_metric["pred_pixels"],
            "total_pixels": area_metric["total_pixels"],
            "target_ratio": target_ratio,
            "pred_ratio": area_metric["pred_ratio"],
            "mask_size_class": size_class_from_ratio(target_ratio),
        })

    df = pd.DataFrame(rows)
    summary = summarize_df(df)
    return df, summary


def summarize_subset(sub: pd.DataFrame) -> Dict:
    row = {"count": int(len(sub))}
    metrics = [
        "area_dice",
        "area_iou",
        "area_precision",
        "area_recall",
        "area_f1",
        "boundary_dice",
        "boundary_iou",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
    ]

    for metric in metrics:
        if len(sub) == 0:
            row[f"{metric}_mean"] = float("nan")
            row[f"{metric}_median"] = float("nan")
        else:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_median"] = float(sub[metric].median())

    return row


def summarize_df(df: pd.DataFrame) -> Dict:
    subsets = {
        "all": df,
        "non_empty": df[df["target_ratio"] > 0],
        "useful_5_75": df[(df["target_ratio"] >= 0.05) & (df["target_ratio"] < 0.75)],
        "nontrivial_1_90": df[(df["target_ratio"] >= 0.01) & (df["target_ratio"] < 0.90)],
        "small_1_5": df[(df["target_ratio"] >= 0.01) & (df["target_ratio"] < 0.05)],
        "very_small_0_1": df[(df["target_ratio"] > 0) & (df["target_ratio"] < 0.01)],
        "empty": df[df["target_ratio"] == 0],
        "almost_full_90_100": df[df["target_ratio"] >= 0.90],
    }

    return {name: summarize_subset(sub) for name, sub in subsets.items()}


def flatten_summary(summary: Dict, prefix: str = "") -> Dict:
    out = {}
    for subset, vals in summary.items():
        for k, v in vals.items():
            out[f"{prefix}{subset}_{k}"] = v
    return out


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_score(summary: Dict) -> float:
    useful_dice = summary["useful_5_75"]["area_dice_mean"]
    useful_boundary_f1 = summary["useful_5_75"]["boundary_f1_mean"]
    small_dice = summary["small_1_5"]["area_dice_mean"]

    vals = []
    vals.append(0.0 if pd.isna(useful_dice) else useful_dice)
    vals.append(0.20 * (0.0 if pd.isna(useful_boundary_f1) else useful_boundary_f1))
    vals.append(0.10 * (0.0 if pd.isna(small_dice) else small_dice))
    return float(sum(vals))


def main():
    parser = argparse.ArgumentParser(description="Avalia modelo B01 boundary-aware multitask.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/ai4boundaries/orthos"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--area-thresholds", type=str, default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--boundary-thresholds", type=str, default="0.40,0.50,0.60")
    parser.add_argument("--min-area-values", type=str, default="0,200,500,800,1000,1200")
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])

    parser.add_argument("--tta-mode", type=str, default="flips", choices=["none", "flips", "d4"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--area-band", type=int, default=None)
    parser.add_argument("--boundary-band", type=int, default=None)
    parser.add_argument("--normalization", type=str, default=None, choices=[None, "imagenet", "none"])

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    model, cfg, ckpt = load_model(args.checkpoint, device)

    image_size = args.image_size if args.image_size is not None else int(cfg.get("image_size", 512))
    area_band = args.area_band if args.area_band is not None else int(cfg.get("area_band", 1))
    boundary_band = args.boundary_band if args.boundary_band is not None else int(cfg.get("boundary_band", 2))
    normalization = args.normalization if args.normalization is not None else cfg.get("normalization", "imagenet")

    pairs = build_pairs(args.root, args.split)
    dataset = OrthoBoundaryEvalDataset(
        pairs=pairs,
        image_size=image_size,
        area_band=area_band,
        boundary_band=boundary_band,
        normalization=normalization,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    tta_list = get_tta_list(args.tta_mode)

    config_out = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "root": str(args.root),
        "output_dir": str(args.output_dir),
        "image_size": image_size,
        "area_band": area_band,
        "boundary_band": boundary_band,
        "normalization": normalization,
        "tta_mode": args.tta_mode,
        "tta_list": tta_list,
        "n_samples": len(dataset),
        "training_config": cfg,
    }

    (args.output_dir / "eval_config.json").write_text(
        json.dumps(config_out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    payload = predict_probs(
        model=model,
        loader=loader,
        device=device,
        amp=args.amp,
        amp_dtype=amp_dtype,
        tta_list=tta_list,
    )

    area_thresholds = parse_float_list(args.area_thresholds)
    boundary_thresholds = parse_float_list(args.boundary_thresholds)
    min_area_values = parse_int_list(args.min_area_values)

    rank_rows = []

    for area_thr in area_thresholds:
        for boundary_thr in boundary_thresholds:
            for min_area in min_area_values:
                df, summary = evaluate_predictions(
                    payload=payload,
                    area_threshold=area_thr,
                    boundary_threshold=boundary_thr,
                    min_area=min_area,
                    connectivity=args.connectivity,
                )

                run_name = f"area_thr{area_thr:.4f}_boundary_thr{boundary_thr:.4f}_minarea{min_area}"
                run_dir = args.output_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                df.to_csv(run_dir / "per_sample_metrics.csv", index=False)

                summary_file = {
                    "area_threshold": area_thr,
                    "boundary_threshold": boundary_thr,
                    "min_area": min_area,
                    "summary": summary,
                }

                (run_dir / "summary_metrics.json").write_text(
                    json.dumps(summary_file, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                flat = flatten_summary(summary)
                rank_rows.append({
                    "run_dir": str(run_dir),
                    "area_threshold": area_thr,
                    "boundary_threshold": boundary_thr,
                    "min_area": min_area,
                    "selection_score": safe_score(summary),
                    "all_area_dice": summary["all"]["area_dice_mean"],
                    "all_area_iou": summary["all"]["area_iou_mean"],
                    "all_boundary_f1": summary["all"]["boundary_f1_mean"],
                    "useful_area_dice": summary["useful_5_75"]["area_dice_mean"],
                    "useful_area_iou": summary["useful_5_75"]["area_iou_mean"],
                    "useful_boundary_f1": summary["useful_5_75"]["boundary_f1_mean"],
                    "small_area_dice": summary["small_1_5"]["area_dice_mean"],
                    "very_small_area_dice": summary["very_small_0_1"]["area_dice_mean"],
                    "non_empty_area_dice": summary["non_empty"]["area_dice_mean"],
                    **flat,
                })

    ranking = pd.DataFrame(rank_rows).sort_values(
        ["selection_score", "useful_area_dice", "all_area_dice"],
        ascending=False,
    )
    ranking.to_csv(args.output_dir / "ranking.csv", index=False)

    print("\nRanking B01:")
    cols = [
        "run_dir",
        "area_threshold",
        "boundary_threshold",
        "min_area",
        "selection_score",
        "all_area_dice",
        "useful_area_dice",
        "small_area_dice",
        "all_boundary_f1",
        "useful_boundary_f1",
    ]
    print(ranking[cols].head(30).to_string(index=False))

    print("\nArquivos salvos em:", args.output_dir)
    print("- ranking.csv")
    print("- subpastas com per_sample_metrics.csv e summary_metrics.json")


if __name__ == "__main__":
    main()

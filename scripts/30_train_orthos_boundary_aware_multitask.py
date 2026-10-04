from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import segmentation_models_pytorch as smp


# ============================================================
# Script 30
# Treino boundary-aware multitask para orthos AI4Boundaries
#
# Ideia:
#   entrada RGB ortho
#   saída canal 0: área/interior do talhão (mask band 1)
#   saída canal 1: borda/limite do talhão (mask band 2)
#
# Loss:
#   loss_total = area_weight * loss_area + boundary_weight * loss_boundary
#   loss_area = BCEWithLogits + DiceLoss
#   loss_boundary = BCEWithLogits(pos_weight dinâmico) + DiceLoss
#
# Também inclui sampler balanceado por proporção de máscara de área,
# para reduzir o domínio dos casos vazios/quase cheios.
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    if not pairs:
        raise RuntimeError(f"Nenhum par imagem/máscara encontrado em {image_dir}")

    return pairs


def ratio_class(ratio: float) -> str:
    if ratio == 0:
        return "empty"
    if ratio < 0.01:
        return "very_small_0_1"
    if ratio < 0.05:
        return "small_1_5"
    if ratio < 0.25:
        return "medium_5_25"
    if ratio < 0.75:
        return "large_25_75"
    if ratio < 0.90:
        return "very_large_75_90"
    return "almost_full_90_100"


def compute_area_ratio(mask_path: Path, area_band: int) -> float:
    with rasterio.open(mask_path) as src:
        mask = src.read(area_band)
    return float((mask > 0).mean())


def compute_pair_stats(pairs: List[Tuple[Path, Path]], area_band: int, cache_csv: Path | None = None) -> pd.DataFrame:
    if cache_csv is not None and cache_csv.exists():
        return pd.read_csv(cache_csv)

    rows = []
    for image_path, mask_path in tqdm(pairs, desc="Calculando proporções das máscaras"):
        r = compute_area_ratio(mask_path, area_band=area_band)
        rows.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "target_ratio": r,
            "ratio_class": ratio_class(r),
        })

    df = pd.DataFrame(rows)

    if cache_csv is not None:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)

    return df


def build_balanced_sampler(stats: pd.DataFrame, strategy: str = "balanced") -> WeightedRandomSampler | None:
    if strategy == "none":
        return None

    # Pesos por classe: força mais amostras de pequenos/médios e reduz quase cheios/vazios.
    class_weights = {
        "empty": 0.50,
        "very_small_0_1": 3.00,
        "small_1_5": 3.50,
        "medium_5_25": 2.50,
        "large_25_75": 1.00,
        "very_large_75_90": 0.65,
        "almost_full_90_100": 0.45,
    }

    if strategy == "inverse_freq":
        counts = stats["ratio_class"].value_counts().to_dict()
        weights = [1.0 / max(counts[c], 1) for c in stats["ratio_class"].values]
    elif strategy == "balanced":
        weights = [class_weights.get(c, 1.0) for c in stats["ratio_class"].values]
    else:
        raise ValueError(f"sampler_strategy inválida: {strategy}")

    weights = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


class OrthoBoundaryDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        image_size: int = 512,
        area_band: int = 1,
        boundary_band: int = 2,
        normalization: str = "imagenet",
        augment: bool = False,
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.area_band = area_band
        self.boundary_band = boundary_band
        self.normalization = normalization
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_image(self, image_path: Path) -> np.ndarray:
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3]).astype(np.float32)

        if image.max() > 2.0:
            image = image / 255.0

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

        target = np.stack([area, boundary], axis=0).astype(np.float32)
        return target

    def _resize(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _augment(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flips.
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            target = torch.flip(target, dims=[2])

        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            target = torch.flip(target, dims=[1])

        # Rotação 0/90/180/270.
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k=k, dims=[1, 2])
            target = torch.rot90(target, k=k, dims=[1, 2])

        # Ruído leve no espaço já normalizado. Mantido pequeno.
        if random.random() < 0.25:
            image = image + torch.randn_like(image) * 0.015

        return image, target

    def __getitem__(self, idx: int) -> Dict:
        image_path, mask_path = self.pairs[idx]

        image = torch.from_numpy(self._read_image(image_path)).float()
        target = torch.from_numpy(self._read_targets(mask_path)).float()
        image, target = self._resize(image, target)

        if self.augment:
            image, target = self._augment(image, target)

        area_ratio = float(target[0].mean().item())
        boundary_ratio = float(target[1].mean().item())

        return {
            "image": image,
            "target": target,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "area_ratio": area_ratio,
            "boundary_ratio": boundary_ratio,
        }


def build_model(architecture: str, encoder_name: str, encoder_weights: str | None, in_channels: int, classes: int) -> nn.Module:
    if encoder_weights is not None and str(encoder_weights).lower() == "none":
        encoder_weights = None

    architecture = architecture.lower()

    if architecture == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    if architecture == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    raise ValueError(f"Arquitetura inválida: {architecture}")


def soft_dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)

    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def tversky_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.35,
    beta: float = 0.65,
    eps: float = 1e-7,
) -> torch.Tensor:
    # alpha penaliza FP, beta penaliza FN.
    # beta maior ajuda recall, útil para objetos pequenos.
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)

    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1 - targets)).sum(dim=dims)
    fn = ((1 - probs) * targets).sum(dim=dims)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky.mean()


def dynamic_pos_weight(targets: torch.Tensor, max_weight: float = 50.0) -> torch.Tensor:
    # targets: B,1,H,W
    pos = targets.sum()
    neg = targets.numel() - pos
    if float(pos.item()) <= 0:
        return torch.tensor(1.0, device=targets.device)
    w = neg / (pos + 1e-7)
    return torch.clamp(w, min=1.0, max=max_weight)


def multitask_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    area_weight: float = 1.0,
    boundary_weight: float = 0.50,
    bce_area_weight: float = 0.50,
    dice_area_weight: float = 0.50,
    bce_boundary_weight: float = 0.50,
    dice_boundary_weight: float = 0.50,
    boundary_pos_weight_max: float = 80.0,
    use_tversky_area: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    area_logits = logits[:, 0:1]
    boundary_logits = logits[:, 1:2]

    area_target = targets[:, 0:1]
    boundary_target = targets[:, 1:2]

    area_bce = F.binary_cross_entropy_with_logits(area_logits, area_target)

    if use_tversky_area:
        area_region = tversky_loss_with_logits(area_logits, area_target, alpha=0.35, beta=0.65)
    else:
        area_region = soft_dice_loss_with_logits(area_logits, area_target)

    boundary_pw = dynamic_pos_weight(boundary_target, max_weight=boundary_pos_weight_max)
    boundary_bce = F.binary_cross_entropy_with_logits(
        boundary_logits,
        boundary_target,
        pos_weight=boundary_pw,
    )
    boundary_dice = soft_dice_loss_with_logits(boundary_logits, boundary_target)

    area_loss = bce_area_weight * area_bce + dice_area_weight * area_region
    boundary_loss = bce_boundary_weight * boundary_bce + dice_boundary_weight * boundary_dice

    total = area_weight * area_loss + boundary_weight * boundary_loss

    parts = {
        "total": float(total.detach().item()),
        "area_loss": float(area_loss.detach().item()),
        "boundary_loss": float(boundary_loss.detach().item()),
        "area_bce": float(area_bce.detach().item()),
        "area_region": float(area_region.detach().item()),
        "boundary_bce": float(boundary_bce.detach().item()),
        "boundary_dice": float(boundary_dice.detach().item()),
        "boundary_pos_weight": float(boundary_pw.detach().item()),
    }

    return total, parts


def binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    p = preds.reshape(preds.shape[0], -1)
    y = targets.reshape(targets.shape[0], -1)

    tp = (p * y).sum(dim=1)
    fp = (p * (1 - y)).sum(dim=1)
    fn = ((1 - p) * y).sum(dim=1)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "f1": float(f1.mean().item()),
    }


def useful_area_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    min_ratio: float = 0.05,
    max_ratio: float = 0.75,
    eps: float = 1e-7,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    p = preds.reshape(preds.shape[0], -1)
    y = targets.reshape(targets.shape[0], -1)

    ratios = y.mean(dim=1)
    keep = (ratios >= min_ratio) & (ratios < max_ratio)

    if keep.sum().item() == 0:
        return {
            "dice": float("nan"),
            "iou": float("nan"),
            "count": 0,
        }

    p = p[keep]
    y = y[keep]

    tp = (p * y).sum(dim=1)
    fp = (p * (1 - y)).sum(dim=1)
    fn = ((1 - p) * y).sum(dim=1)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "count": int(keep.sum().item()),
    }


def boundary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    return binary_metrics_from_logits(logits, targets, threshold=threshold)


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}

    keys = dicts[0].keys()
    out = {}

    for key in keys:
        vals = [d[key] for d in dicts if key in d and not math.isnan(float(d[key]))]
        if vals:
            out[key] = float(np.mean(vals))
        else:
            out[key] = float("nan")

    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    args,
) -> Dict[str, float]:
    model.train()

    loss_rows = []
    area_metric_rows = []
    boundary_metric_rows = []
    useful_rows = []

    pbar = tqdm(loader, desc="Treino", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            logits = model(images)
            loss, parts = multitask_loss(
                logits=logits,
                targets=targets,
                area_weight=args.area_weight,
                boundary_weight=args.boundary_weight,
                bce_area_weight=args.bce_area_weight,
                dice_area_weight=args.dice_area_weight,
                bce_boundary_weight=args.bce_boundary_weight,
                dice_boundary_weight=args.dice_boundary_weight,
                boundary_pos_weight_max=args.boundary_pos_weight_max,
                use_tversky_area=args.use_tversky_area,
            )

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        loss_rows.append(parts)

        with torch.no_grad():
            area_metrics = binary_metrics_from_logits(logits[:, 0:1], targets[:, 0:1], threshold=args.metric_threshold)
            boundary_metrics = boundary_metrics_from_logits(logits[:, 1:2], targets[:, 1:2], threshold=args.metric_threshold)
            useful_metrics = useful_area_metrics_from_logits(
                logits[:, 0:1],
                targets[:, 0:1],
                threshold=args.metric_threshold,
                min_ratio=0.05,
                max_ratio=0.75,
            )

        area_metric_rows.append(area_metrics)
        boundary_metric_rows.append(boundary_metrics)
        useful_rows.append(useful_metrics)

        pbar.set_postfix(
            loss=parts["total"],
            area_dice=area_metrics["dice"],
            bnd_dice=boundary_metrics["dice"],
        )

    loss_mean = mean_dict(loss_rows)
    area_mean = mean_dict(area_metric_rows)
    boundary_mean = mean_dict(boundary_metric_rows)
    useful_mean = mean_dict(useful_rows)

    return {
        **{f"loss_{k}": v for k, v in loss_mean.items()},
        **{f"area_{k}": v for k, v in area_mean.items()},
        **{f"boundary_{k}": v for k, v in boundary_mean.items()},
        **{f"useful_{k}": v for k, v in useful_mean.items()},
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    args,
) -> Dict[str, float]:
    model.eval()

    loss_rows = []
    area_metric_rows = []
    boundary_metric_rows = []
    useful_rows = []

    for batch in tqdm(loader, desc="Validação", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            logits = model(images)
            loss, parts = multitask_loss(
                logits=logits,
                targets=targets,
                area_weight=args.area_weight,
                boundary_weight=args.boundary_weight,
                bce_area_weight=args.bce_area_weight,
                dice_area_weight=args.dice_area_weight,
                bce_boundary_weight=args.bce_boundary_weight,
                dice_boundary_weight=args.dice_boundary_weight,
                boundary_pos_weight_max=args.boundary_pos_weight_max,
                use_tversky_area=args.use_tversky_area,
            )

        loss_rows.append(parts)

        area_metrics = binary_metrics_from_logits(logits[:, 0:1], targets[:, 0:1], threshold=args.metric_threshold)
        boundary_metrics = boundary_metrics_from_logits(logits[:, 1:2], targets[:, 1:2], threshold=args.metric_threshold)
        useful_metrics = useful_area_metrics_from_logits(
            logits[:, 0:1],
            targets[:, 0:1],
            threshold=args.metric_threshold,
            min_ratio=0.05,
            max_ratio=0.75,
        )

        area_metric_rows.append(area_metrics)
        boundary_metric_rows.append(boundary_metrics)
        useful_rows.append(useful_metrics)

    loss_mean = mean_dict(loss_rows)
    area_mean = mean_dict(area_metric_rows)
    boundary_mean = mean_dict(boundary_metric_rows)
    useful_mean = mean_dict(useful_rows)

    return {
        **{f"loss_{k}": v for k, v in loss_mean.items()},
        **{f"area_{k}": v for k, v in area_mean.items()},
        **{f"boundary_{k}": v for k, v in boundary_mean.items()},
        **{f"useful_{k}": v for k, v in useful_mean.items()},
    }


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    config: Dict,
    filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        output_dir / filename,
    )


def main():
    parser = argparse.ArgumentParser(description="Treina modelo Ortho boundary-aware multitask.")

    parser.add_argument("--root", type=Path, default=Path("data/raw/ai4boundaries/orthos"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", type=str, default="unetplusplus", choices=["unet", "unetplusplus", "deeplabv3plus", "fpn"])
    parser.add_argument("--encoder-name", type=str, default="resnet50")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=512)

    parser.add_argument("--area-band", type=int, default=1)
    parser.add_argument("--boundary-band", type=int, default=2)
    parser.add_argument("--normalization", type=str, default="imagenet", choices=["imagenet", "none"])

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampler-strategy", type=str, default="balanced", choices=["none", "balanced", "inverse_freq"])
    parser.add_argument("--no-augment", action="store_true")

    parser.add_argument("--area-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=0.50)
    parser.add_argument("--bce-area-weight", type=float, default=0.50)
    parser.add_argument("--dice-area-weight", type=float, default=0.50)
    parser.add_argument("--bce-boundary-weight", type=float, default=0.50)
    parser.add_argument("--dice-boundary-weight", type=float, default=0.50)
    parser.add_argument("--boundary-pos-weight-max", type=float, default=80.0)
    parser.add_argument("--use-tversky-area", action="store_true")

    parser.add_argument("--metric-threshold", type=float, default=0.5)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    args = parser.parse_args()

    # Windows + DataLoader com workers/persistent_workers pode travar (deadlock) apos varias epocas.
    # Forca num_workers=0 para evitar o problema (carregamento no processo principal; GPU e o gargalo aqui).
    import sys as _sys
    if _sys.platform.startswith("win") and args.num_workers > 0:
        print(f"[aviso] Windows: num_workers={args.num_workers} pode travar o treino (deadlock). Usando num_workers=0.")
        args.num_workers = 0

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    train_pairs = build_pairs(args.root, "train")
    val_pairs = build_pairs(args.root, "val")

    train_stats = compute_pair_stats(
        train_pairs,
        area_band=args.area_band,
        cache_csv=args.output_dir / "train_mask_ratio_stats.csv",
    )

    val_stats = compute_pair_stats(
        val_pairs,
        area_band=args.area_band,
        cache_csv=args.output_dir / "val_mask_ratio_stats.csv",
    )

    print("\nDistribuição treino:")
    print(train_stats["ratio_class"].value_counts().to_string())
    print("\nDistribuição val:")
    print(val_stats["ratio_class"].value_counts().to_string())

    train_dataset = OrthoBoundaryDataset(
        pairs=train_pairs,
        image_size=args.image_size,
        area_band=args.area_band,
        boundary_band=args.boundary_band,
        normalization=args.normalization,
        augment=not args.no_augment,
    )

    val_dataset = OrthoBoundaryDataset(
        pairs=val_pairs,
        image_size=args.image_size,
        area_band=args.area_band,
        boundary_band=args.boundary_band,
        normalization=args.normalization,
        augment=False,
    )

    sampler = build_balanced_sampler(train_stats, strategy=args.sampler_strategy)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler leve. Evita overfitting tardio.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.05,
    )

    scaler = GradScaler("cuda", enabled=args.amp and args.amp_dtype == "fp16")

    config = vars(args).copy()
    config.update({
        "task": "ortho_boundary_aware_multitask",
        "in_channels": 3,
        "classes": 2,
        "output_channels": {
            "0": "area_mask_band_1",
            "1": "boundary_mask_band_2",
        },
        "model_selection_score": "val_useful_dice + 0.25 * val_boundary_f1",
    })

    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    history = []
    best_score = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=args.amp,
            amp_dtype=amp_dtype,
            args=args,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            amp=args.amp,
            amp_dtype=amp_dtype,
            args=args,
        )

        scheduler.step()

        # Score foca casos úteis e borda. Área global fica registrada, mas não guia tudo.
        useful_dice = val_metrics.get("useful_dice", float("nan"))
        boundary_f1 = val_metrics.get("boundary_f1", float("nan"))
        area_dice = val_metrics.get("area_dice", float("nan"))

        if math.isnan(useful_dice):
            score = area_dice
        else:
            score = useful_dice + 0.25 * (0.0 if math.isnan(boundary_f1) else boundary_f1)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "selection_score": score,
        }
        history.append(row)

        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics.get('loss_total', float('nan')):.4f} | "
            f"val_loss={val_metrics.get('loss_total', float('nan')):.4f} | "
            f"val_area_dice={val_metrics.get('area_dice', float('nan')):.4f} | "
            f"val_useful_dice={val_metrics.get('useful_dice', float('nan')):.4f} | "
            f"val_boundary_f1={val_metrics.get('boundary_f1', float('nan')):.4f} | "
            f"score={score:.4f}"
        )

        save_checkpoint(
            output_dir=args.output_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_score=best_score,
            config=config,
            filename="last_boundary_model.pt",
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                output_dir=args.output_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_score,
                config=config,
                filename="best_boundary_model.pt",
            )
            print(
                f"Novo melhor modelo: {args.output_dir / 'best_boundary_model.pt'} | "
                f"best_score={best_score:.4f} | epoch={best_epoch}"
            )

    final_summary = {
        "best_score": best_score,
        "best_epoch": best_epoch,
        "output_dir": str(args.output_dir),
    }

    (args.output_dir / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nTreino finalizado.")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

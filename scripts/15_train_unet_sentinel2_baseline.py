from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def build_pairs(root: Path, split: str, metadata_csv: Path | None = None) -> List[Tuple[Path, Path]]:
    image_dir = root / "images" / split
    mask_dir = root / "masks" / split

    if metadata_csv is not None and metadata_csv.exists():
        df = pd.read_csv(metadata_csv)

        if "image" not in df.columns:
            raise ValueError(f"O CSV precisa ter a coluna 'image'. Colunas encontradas: {list(df.columns)}")

        names = df["image"].astype(str).tolist()
    else:
        names = [p.name for p in sorted(image_dir.glob("*.nc"))]

    pairs = []

    for image_name in names:
        image_path = image_dir / image_name
        mask_path = mask_dir / image_name.replace(".nc", ".tif")

        if image_path.exists() and mask_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


class AI4BoundariesSentinel2Dataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        bands: List[str],
        time_mode: str = "median",
        time_index: int = -1,
        image_size: int = 256,
        positive_values: List[int] | None = None,
        augment: bool = False,
        p_low: float = 2.0,
        p_high: float = 98.0,
    ):
        self.pairs = pairs
        self.bands = bands
        self.time_mode = time_mode
        self.time_index = time_index
        self.image_size = image_size
        self.positive_values = positive_values
        self.augment = augment
        self.p_low = p_low
        self.p_high = p_high

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_sentinel2(self, image_path: Path) -> np.ndarray:
        ds = xr.open_dataset(image_path)

        channels = []

        try:
            for band in self.bands:
                if band not in ds:
                    raise ValueError(f"Banda ausente em {image_path.name}: {band}")

                data = ds[band]

                if "time" in data.dims:
                    if self.time_mode == "median":
                        arr = data.median(dim="time", skipna=True).values
                        channels.append(normalize_percentile(arr, self.p_low, self.p_high))
                    elif self.time_mode == "mean":
                        arr = data.mean(dim="time", skipna=True).values
                        channels.append(normalize_percentile(arr, self.p_low, self.p_high))
                    elif self.time_mode == "index":
                        arr = data.isel(time=self.time_index).values
                        channels.append(normalize_percentile(arr, self.p_low, self.p_high))
                    elif self.time_mode == "stack":
                        values = data.values
                        for t in range(values.shape[0]):
                            channels.append(normalize_percentile(values[t], self.p_low, self.p_high))
                    else:
                        raise ValueError(f"time_mode inválido: {self.time_mode}")
                else:
                    arr = data.values
                    channels.append(normalize_percentile(arr, self.p_low, self.p_high))

            image = np.stack(channels, axis=0).astype(np.float32)
            return image
        finally:
            ds.close()

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        if self.positive_values is None:
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.isin(mask, self.positive_values).astype(np.float32)

        return mask[None, :, :]

    def _maybe_resize(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.shape[-1] == self.image_size and image.shape[-2] == self.image_size:
            return image, mask

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)

        return image, mask

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])

        k = random.randint(0, 3)
        if k:
            image = torch.rot90(image, k=k, dims=[1, 2])
            mask = torch.rot90(mask, k=k, dims=[1, 2])

        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        image_path, mask_path = self.pairs[idx]

        image = self._read_sentinel2(image_path)
        mask = self._read_mask(mask_path)

        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).float()

        image_t, mask_t = self._maybe_resize(image_t, mask_t)

        if self.augment:
            image_t, mask_t = self._augment(image_t, mask_t)

        return {
            "image": image_t,
            "mask": mask_t,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_f = preds.view(preds.shape[0], -1)
    targets_f = targets.view(targets.shape[0], -1)

    tp = (preds_f * targets_f).sum(dim=1)
    fp = (preds_f * (1.0 - targets_f)).sum(dim=1)
    fn = ((1.0 - preds_f) * targets_f).sum(dim=1)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)

    return {
        "dice": float(dice.mean().detach().cpu()),
        "iou": float(iou.mean().detach().cpu()),
        "precision": float(precision.mean().detach().cpu()),
        "recall": float(recall.mean().detach().cpu()),
        "f1": float(f1.mean().detach().cpu()),
    }


def build_model(
    architecture: str,
    encoder_name: str,
    encoder_weights: str | None,
    in_channels: int,
) -> nn.Module:
    if encoder_weights is not None and encoder_weights.lower() in {"", "none", "null"}:
        encoder_weights = None

    if architecture == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

    if architecture == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

    if architecture == "pan":
        return smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

    raise ValueError(f"Arquitetura inválida: {architecture}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bce_loss: nn.Module,
    amp: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler | None,
    threshold: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc="Treino", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            logits = model(images)
            loss_bce = bce_loss(logits, masks)
            loss_dice = soft_dice_loss(logits, masks)
            loss = loss_bce + loss_dice

        if amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_metrics = compute_batch_metrics(logits.detach(), masks, threshold=threshold)

        total_loss += float(loss.detach().cpu())
        for key in metrics_sum:
            metrics_sum[key] += batch_metrics[key]

        n_batches += 1

        pbar.set_postfix(
            {
                "loss": total_loss / n_batches,
                "dice": metrics_sum["dice"] / n_batches,
            }
        )

    out = {"loss": total_loss / max(n_batches, 1)}
    out.update({key: value / max(n_batches, 1) for key, value in metrics_sum.items()})

    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bce_loss: nn.Module,
    amp: bool,
    amp_dtype: torch.dtype,
    threshold: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc="Validação", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            logits = model(images)
            loss_bce = bce_loss(logits, masks)
            loss_dice = soft_dice_loss(logits, masks)
            loss = loss_bce + loss_dice

        batch_metrics = compute_batch_metrics(logits, masks, threshold=threshold)

        total_loss += float(loss.detach().cpu())
        for key in metrics_sum:
            metrics_sum[key] += batch_metrics[key]

        n_batches += 1

        pbar.set_postfix(
            {
                "loss": total_loss / n_batches,
                "dice": metrics_sum["dice"] / n_batches,
            }
        )

    out = {"loss": total_loss / max(n_batches, 1)}
    out.update({key: value / max(n_batches, 1) for key, value in metrics_sum.items()})

    return out


def main():
    parser = argparse.ArgumentParser(description="Treina baseline U-Net/DeepLabV3+ com Sentinel-2 AI4Boundaries.")

    parser.add_argument("--root", type=Path, default=Path("data/raw/ai4boundaries/sentinel2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sentinel2/test_024_unet_s2_median5_a100"))

    parser.add_argument("--architecture", type=str, default="unet", choices=["unet", "deeplabv3plus", "unetplusplus", "fpn", "pan"])
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--bands", type=str, default="B2,B3,B4,B8,NDVI")
    parser.add_argument("--time-mode", type=str, default="median", choices=["median", "mean", "index", "stack"])
    parser.add_argument("--time-index", type=int, default=-1)

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    parser.add_argument("--positive-values", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    args = parser.parse_args()

    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]

    positive_values = None
    if args.positive_values.strip():
        positive_values = [int(v.strip()) for v in args.positive_values.split(",") if v.strip()]

    train_pairs = build_pairs(args.root, "train", args.root / "metadata" / "train.csv")
    val_pairs = build_pairs(args.root, "val", args.root / "metadata" / "val.csv")

    if args.max_train_samples is not None:
        train_pairs = train_pairs[: args.max_train_samples]

    if args.max_val_samples is not None:
        val_pairs = val_pairs[: args.max_val_samples]

    if not train_pairs:
        raise RuntimeError("Nenhum par de treino encontrado.")
    if not val_pairs:
        raise RuntimeError("Nenhum par de validação encontrado.")

    if args.time_mode == "stack":
        sample_ds = xr.open_dataset(train_pairs[0][0])
        time_count = int(sample_ds.sizes.get("time", 1))
        sample_ds.close()
        in_channels = len(bands) * time_count
    else:
        in_channels = len(bands)

    print("Configuração:")
    print(f"Root: {args.root}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Arquitetura: {args.architecture}")
    print(f"Encoder: {args.encoder_name}")
    print(f"Encoder weights: {args.encoder_weights}")
    print(f"Bands: {bands}")
    print(f"Time mode: {args.time_mode}")
    print(f"In channels: {in_channels}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"AMP: {args.amp} ({args.amp_dtype})")
    print(f"Positive values: {positive_values if positive_values is not None else 'mask > 0'}")

    train_dataset = AI4BoundariesSentinel2Dataset(
        pairs=train_pairs,
        bands=bands,
        time_mode=args.time_mode,
        time_index=args.time_index,
        image_size=args.image_size,
        positive_values=positive_values,
        augment=True,
    )

    val_dataset = AI4BoundariesSentinel2Dataset(
        pairs=val_pairs,
        bands=bands,
        time_mode=args.time_mode,
        time_index=args.time_index,
        image_size=args.image_size,
        positive_values=positive_values,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights if args.encoder_weights.lower() != "none" else None,
        in_channels=in_channels,
    ).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.amp_dtype == "fp16")

    best_val_dice = -1.0
    history = []

    config = vars(args).copy()
    config["bands"] = bands
    config["in_channels"] = in_channels
    config["positive_values"] = positive_values

    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            bce_loss=bce_loss,
            amp=args.amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            threshold=args.threshold,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            bce_loss=bce_loss,
            amp=args.amp,
            amp_dtype=amp_dtype,
            threshold=args.threshold,
        )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(args.output_dir / "training_history.csv", index=False)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]

            ckpt_path = args.output_dir / "best_sentinel2_unet.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                    "best_val_dice": best_val_dice,
                },
                ckpt_path,
            )

            print(f"Novo melhor modelo: {ckpt_path} | val_dice={best_val_dice:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_val_dice": best_val_dice,
        },
        args.output_dir / "last_sentinel2_unet.pt",
    )

    print(f"Treino finalizado. Melhor val_dice={best_val_dice:.4f}")
    print(f"Arquivos salvos em: {args.output_dir}")


if __name__ == "__main__":
    main()

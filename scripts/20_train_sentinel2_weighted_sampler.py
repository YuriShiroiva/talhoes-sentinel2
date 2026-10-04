from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


def build_items(root: Path, split: str):
    image_dir = root / "images" / split
    mask_dir = root / "masks" / split
    csv_path = root / "metadata" / f"{split}.csv"
    names = pd.read_csv(csv_path)["image"].astype(str).tolist() if csv_path.exists() else [p.name for p in sorted(image_dir.glob("*.nc"))]
    items = []
    for image_name in tqdm(names, desc=f"Indexando {split}"):
        image_path = image_dir / image_name
        mask_path = mask_dir / image_name.replace(".nc", ".tif")
        if not image_path.exists() or not mask_path.exists():
            continue
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        ratio = float((mask > 0).mean())
        items.append({
            "image_path": image_path,
            "mask_path": mask_path,
            "image_name": image_name,
            "mask_name": mask_path.name,
            "positive_ratio": ratio,
            "size_class": size_class_from_ratio(ratio),
        })
    return items


class Sentinel2SegDataset(Dataset):
    def __init__(self, items, bands, time_mode="stack", time_index=-1, image_size=256, augment=False):
        self.items = items
        self.bands = bands
        self.time_mode = time_mode
        self.time_index = time_index
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def _read_image(self, image_path: Path) -> np.ndarray:
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

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        return (mask > 0).astype(np.float32)[None, :, :]

    def _resize(self, image, mask):
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image, mask
        image = F.interpolate(image[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask[None], size=(self.image_size, self.image_size), mode="nearest")[0]
        return image, mask

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2]); mask = torch.flip(mask, dims=[2])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1]); mask = torch.flip(mask, dims=[1])
        k = random.randint(0, 3)
        if k:
            image = torch.rot90(image, k=k, dims=[1, 2]); mask = torch.rot90(mask, k=k, dims=[1, 2])
        return image, mask

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = torch.from_numpy(self._read_image(item["image_path"])).float()
        mask = torch.from_numpy(self._read_mask(item["mask_path"])).float()
        image, mask = self._resize(image, mask)
        if self.augment:
            image, mask = self._augment(image, mask)
        return {
            "image": image,
            "mask": mask,
            "image_path": str(item["image_path"]),
            "mask_path": str(item["mask_path"]),
            "size_class": item["size_class"],
        }


def parse_class_weights(text: str):
    out = {}
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        key, value = chunk.split(":")
        out[key.strip()] = float(value.strip())
    return out


def make_sampler(items, class_weights, multiplier=1.0):
    sample_weights = [class_weights.get(item["size_class"], 1.0) for item in items]
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), int(len(items) * multiplier), replacement=True)


def dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(1)
    den = probs.sum(1) + targets.sum(1)
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)
    at = alpha * targets + (1 - alpha) * (1 - targets)
    return (at * (1 - pt).pow(gamma) * bce).mean()


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, eps=1e-7):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    tp = (probs * targets).sum(1)
    fp = (probs * (1 - targets)).sum(1)
    fn = ((1 - probs) * targets).sum(1)
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - score.mean()


def compute_loss(logits, masks, mode):
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    if mode == "bce_dice":
        return bce + dice_loss(logits, masks)
    if mode == "bce_dice_focal":
        return bce + dice_loss(logits, masks) + 0.5 * focal_loss(logits, masks)
    if mode == "bce_tversky":
        return bce + tversky_loss(logits, masks)
    if mode == "bce_dice_tversky":
        return bce + dice_loss(logits, masks) + tversky_loss(logits, masks)
    raise ValueError(mode)


def batch_metrics(logits, targets, threshold=0.5, eps=1e-7):
    preds = (torch.sigmoid(logits) >= threshold).float().flatten(1)
    y = targets.flatten(1)
    tp = (preds * y).sum(1)
    fp = (preds * (1 - y)).sum(1)
    fn = ((1 - preds) * y).sum(1)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return {"dice": float(dice.mean().cpu()), "iou": float(iou.mean().cpu()), "precision": float(precision.mean().cpu()), "recall": float(recall.mean().cpu()), "f1": float(f1.mean().cpu())}


def build_model(architecture, encoder_name, encoder_weights, in_channels):
    kwargs = dict(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=1)
    if architecture == "unet":
        return smp.Unet(**kwargs)
    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(**kwargs)
    raise ValueError(architecture)


def run_epoch(model, loader, optimizer, device, amp, amp_dtype, scaler, threshold, loss_mode):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    desc = "Treino" if train else "Validação"
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
                logits = model(images)
                loss = compute_loss(logits, masks, loss_mode)
            if train:
                if amp and scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
        m = batch_metrics(logits.detach(), masks, threshold)
        total_loss += float(loss.detach().cpu())
        for k in sums:
            sums[k] += m[k]
        n += 1
    out = {"loss": total_loss / max(n, 1)}
    out.update({k: v / max(n, 1) for k, v in sums.items()})
    return out


def main():
    parser = argparse.ArgumentParser(description="Treino Sentinel-2 com oversampling por tamanho de máscara.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/ai4boundaries/sentinel2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", default="unet", choices=["unet", "deeplabv3plus"])
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--bands", default="B2,B3,B4,B8,NDVI")
    parser.add_argument("--time-mode", default="stack", choices=["median", "mean", "index", "stack"])
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--loss-mode", default="bce_dice", choices=["bce_dice", "bce_dice_focal", "bce_tversky", "bce_dice_tversky"])
    parser.add_argument("--class-weights", default="empty:0.7,very_small_<1%:8,small_1_5%:5,medium_5_25%:2.5,large_25_75%:1,very_large_>75%:1")
    parser.add_argument("--sampler-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    class_weights = parse_class_weights(args.class_weights)

    train_items = build_items(args.root, "train")
    val_items = build_items(args.root, "val")
    pd.DataFrame(train_items).to_csv(args.output_dir / "train_items_index.csv", index=False)
    pd.DataFrame(val_items).to_csv(args.output_dir / "val_items_index.csv", index=False)

    print("\nDistribuição train:")
    print(pd.DataFrame(train_items)["size_class"].value_counts().to_string())
    print("\nPesos usados:", class_weights)

    if args.time_mode == "stack":
        ds = xr.open_dataset(train_items[0]["image_path"])
        in_channels = len(bands) * int(ds.sizes.get("time", 1))
        ds.close()
    else:
        in_channels = len(bands)

    train_ds = Sentinel2SegDataset(train_items, bands, args.time_mode, args.time_index, args.image_size, augment=True)
    val_ds = Sentinel2SegDataset(val_items, bands, args.time_mode, args.time_index, args.image_size, augment=False)
    sampler = make_sampler(train_items, class_weights, args.sampler_multiplier)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.architecture, args.encoder_name, None if args.encoder_weights.lower() == "none" else args.encoder_weights, in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.amp_dtype == "fp16")

    config = vars(args).copy()
    config.update({"bands": bands, "in_channels": in_channels, "class_weights_parsed": class_weights})
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    best_val_dice = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, device, args.amp, amp_dtype, scaler, args.threshold, args.loss_mode)
        with torch.no_grad():
            val_m = run_epoch(model, val_loader, None, device, args.amp, amp_dtype, None, args.threshold, args.loss_mode)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_m['loss']:.4f} | train_dice={train_m['dice']:.4f} | val_loss={val_m['loss']:.4f} | val_dice={val_m['dice']:.4f} | val_iou={val_m['iou']:.4f} | val_f1={val_m['f1']:.4f}")
        if val_m["dice"] > best_val_dice:
            best_val_dice = val_m["dice"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_metrics": val_m, "config": config, "best_val_dice": best_val_dice}, args.output_dir / "best_sentinel2_unet.pt")
            print(f"Novo melhor modelo: {args.output_dir / 'best_sentinel2_unet.pt'} | val_dice={best_val_dice:.4f}")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "best_val_dice": best_val_dice}, args.output_dir / "last_sentinel2_unet.pt")
    print(f"Treino finalizado. Melhor val_dice={best_val_dice:.4f}")
    print(f"Arquivos salvos em: {args.output_dir}")


if __name__ == "__main__":
    main()

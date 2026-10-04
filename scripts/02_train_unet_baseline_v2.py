from __future__ import annotations

import argparse
from email import parser
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from field_boundary.data.dataset import (
    AI4BoundariesSegmentationDataset,
    find_image_mask_pairs,
    save_pairs_csv,
    split_pairs,
    balance_pairs_by_mask_size,
)
from field_boundary.models import build_model
from field_boundary.models import build_model
from field_boundary.training.losses import build_loss
from field_boundary.training.metrics import binary_metrics_from_logits
from field_boundary.utils.io import save_json, save_rows_csv
from field_boundary.utils.seed import get_device, set_seed
from field_boundary.visualization.plots import save_training_curves, save_prediction_panel, tensor_to_rgb


def run_epoch(model, loader, optimizer, loss_fn, device, threshold, train: bool, amp: bool, scaler, grad_clip: Optional[float]) -> Dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n_batches = 0

    iterator = tqdm(loader, desc="Treino" if train else "Validação", leave=False)
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                loss = loss_fn(logits, masks)

            if train:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()

        metrics = binary_metrics_from_logits(logits.detach(), masks, threshold=threshold)
        totals["loss"] += float(loss.item())
        for key in ["dice", "iou", "precision", "recall", "f1"]:
            totals[key] += metrics[key]
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, config: Dict, epoch: int, best_val_dice: float, history: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "best_val_dice": best_val_dice,
            "metrics_history": history,
        },
        path,
    )


@torch.no_grad()
def save_sample_prediction(model, dataset, device, output_path: Path, threshold: float) -> None:
    model.eval()
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].cpu().numpy()[0]
    logits = model(image)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred = (prob >= threshold).astype("float32")
    rgb = tensor_to_rgb(sample["image"])
    save_prediction_panel(rgb, mask, prob, pred, output_path, title="U-Net baseline — validação")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino U-Net baseline v2")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ai4boundaries"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/unet_baseline"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument(
    "--architecture",
    type=str,
    default="unet",
    choices=["unet", "deeplabv3plus", "deeplabv3+"],)
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument(
    "--loss-name",
    type=str,
    default="bce_dice",
    choices=["bce_dice", "tversky", "focal_tversky", "bce_focal_tversky"],)
    parser.add_argument("--focal-tversky-weight", type=float, default=0.7)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--tversky-gamma", type=float, default=0.75)
    args = parser.parse_args()

    encoder_weights = args.encoder_weights
    if encoder_weights.lower() in {"none", "null", "false"}:
        encoder_weights = None
    if args.in_channels != 3 and encoder_weights is not None:
        print("Aviso: in_channels != 3. encoder_weights será definido como None.")
        encoder_weights = None

    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["output_dir"] = str(args.output_dir)
    config["encoder_weights"] = encoder_weights
    config["architecture"] = args.architecture

    set_seed(args.seed)
    device = get_device()
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(f"Dispositivo: {device} | AMP: {amp_enabled}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, args.output_dir / "config.json")

    pairs = find_image_mask_pairs(args.data_root)
    print(f"Pares encontrados: {len(pairs)}")
    if not pairs:
        raise RuntimeError("Nenhum par imagem/máscara encontrado. Rode scripts/01_inspect_dataset.py.")
    if args.max_samples:
        if args.balanced_sampling:
            pairs = balance_pairs_by_mask_size(
                pairs=pairs,
                max_samples=args.max_samples,
                seed=args.seed,
            )
        else:
            rng = random.Random(args.seed)
            rng.shuffle(pairs)
            pairs = pairs[: args.max_samples]

    train_pairs, val_pairs = split_pairs(pairs, args.val_fraction, args.seed)
    save_pairs_csv(train_pairs, args.output_dir / "splits" / "train_pairs.csv")
    save_pairs_csv(val_pairs, args.output_dir / "splits" / "val_pairs.csv")

    train_ds = AI4BoundariesSegmentationDataset(train_pairs, args.image_size, args.in_channels, augment=True)
    val_ds = AI4BoundariesSegmentationDataset(val_pairs, args.image_size, args.in_channels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = build_model(
    args.architecture,
    args.encoder_name,
    encoder_weights,
    args.in_channels,).to(device)
    loss_fn = build_loss(
    loss_name=args.loss_name,
    bce_weight=args.bce_weight,
    dice_weight=args.dice_weight,
    focal_tversky_weight=args.focal_tversky_weight,
    alpha=args.tversky_alpha,
    beta=args.tversky_beta,
    gamma=args.tversky_gamma,)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 2))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_dice = -1.0
    no_improve = 0
    history: List[Dict] = []
    best_path = args.output_dir / "best_unet_baseline.pt"
    last_path = args.output_dir / "last_unet_baseline.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, loss_fn, device, args.threshold, True, amp_enabled, scaler, args.grad_clip)
        val_metrics = run_epoch(model, val_loader, optimizer, loss_fn, device, args.threshold, False, amp_enabled, scaler, None)
        scheduler.step(val_metrics["dice"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        save_rows_csv(history, args.output_dir / "metrics.csv")
        save_training_curves(history, args.output_dir / "training_curves.png")
        save_checkpoint(last_path, model, optimizer, scheduler, config, epoch, best_val_dice, history)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train_dice={row['train_dice']:.4f} | val_dice={row['val_dice']:.4f} | val_iou={row['val_iou']:.4f}")

        if val_metrics["dice"] > best_val_dice + args.min_delta:
            best_val_dice = val_metrics["dice"]
            no_improve = 0
            save_checkpoint(best_path, model, optimizer, scheduler, config, epoch, best_val_dice, history)
            print(f"Novo melhor modelo: {best_path} | val_dice={best_val_dice:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping na epoch {epoch}.")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_sample_prediction(model, val_ds, device, args.output_dir / "sample_prediction_best.png", args.threshold)
    print(f"Treino finalizado. Melhor val_dice={best_val_dice:.4f}")


if __name__ == "__main__":
    main()

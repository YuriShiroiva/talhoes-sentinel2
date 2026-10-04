from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from field_boundary.data.dataset import AI4BoundariesSegmentationDataset, read_pairs_csv
from field_boundary.models.factory import build_model
from field_boundary.training.metrics import compute_binary_metrics, summarize
from field_boundary.utils.io import load_json, save_json, save_rows_csv
from field_boundary.utils.seed import get_device, set_seed
from field_boundary.visualization.plots import save_prediction_panel, tensor_to_rgb


def get_split_pairs(checkpoint: Path, split: str):
    splits_dir = checkpoint.parent / "splits"
    train_csv = splits_dir / "train_pairs.csv"
    val_csv = splits_dir / "val_pairs.csv"
    if split == "train":
        return read_pairs_csv(train_csv)
    if split == "val":
        return read_pairs_csv(val_csv)
    if split == "all":
        return read_pairs_csv(train_csv) + read_pairs_csv(val_csv)
    raise ValueError("split deve ser train, val ou all")


def load_model_from_checkpoint(checkpoint_path: Path, config: Dict, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})

    architecture = checkpoint_config.get(
        "architecture",
        config.get("architecture", "unet"),
    )

    encoder_name = checkpoint_config.get(
        "encoder_name",
        config.get("encoder_name", "resnet34"),
    )

    in_channels = int(
        checkpoint_config.get(
            "in_channels",
            config.get("in_channels", 3),
        )
    )

    print(f"Carregando arquitetura: {architecture}")
    print(f"Encoder: {encoder_name}")
    print(f"In channels: {in_channels}")

    model = build_model(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def evaluate(model, loader, device, threshold: float):
    rows: List[Dict] = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        metrics = compute_binary_metrics(probs, masks, threshold=threshold)
        for i in range(images.shape[0]):
            rows.append({
                "image_path": batch["image_path"][i],
                "mask_path": batch["mask_path"][i],
                "threshold": threshold,
                "dice": float(metrics["dice"][i]),
                "iou": float(metrics["iou"][i]),
                "precision": float(metrics["precision"][i]),
                "recall": float(metrics["recall"][i]),
                "f1": float(metrics["f1"][i]),
                "tp": float(metrics["tp"][i]),
                "fp": float(metrics["fp"][i]),
                "fn": float(metrics["fn"][i]),
                "tn": float(metrics["tn"][i]),
            })
    summary = {m: summarize([float(r[m]) for r in rows]) for m in ["dice", "iou", "precision", "recall", "f1"]}
    return rows, summary


@torch.no_grad()
def threshold_sweep(model, loader, device, thresholds):
    all_probs, all_masks = [], []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        probs = torch.sigmoid(model(images))
        all_probs.append(probs.cpu())
        all_masks.append(masks.cpu())
    probs = torch.cat(all_probs, dim=0)
    masks = torch.cat(all_masks, dim=0)
    rows = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(probs, masks, threshold=float(threshold))
        rows.append({
            "threshold": float(threshold),
            "dice_mean": float(np.mean(metrics["dice"])),
            "iou_mean": float(np.mean(metrics["iou"])),
            "precision_mean": float(np.mean(metrics["precision"])),
            "recall_mean": float(np.mean(metrics["recall"])),
            "f1_mean": float(np.mean(metrics["f1"])),
        })
    return rows


def save_threshold_plot(rows: List[Dict], output_path: Path):
    plt.figure(figsize=(10, 6))
    x = [r["threshold"] for r in rows]
    for metric in ["dice_mean", "iou_mean", "precision_mean", "recall_mean"]:
        plt.plot(x, [r[metric] for r in rows], label=metric.replace("_mean", ""))
    plt.xlabel("Threshold"); plt.ylabel("Métrica média"); plt.title("Varredura de threshold")
    plt.legend(); plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def save_ranked_figures(model, dataset, rows, device, threshold, output_dir: Path, num_figures: int):
    if not rows:
        return
    ranked = sorted(enumerate(rows), key=lambda item: float(item[1]["dice"]))
    groups = {"worst": ranked[:num_figures], "best": ranked[-num_figures:][::-1]}
    for group_name, items in groups.items():
        for rank, (idx, row) in enumerate(items, start=1):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].cpu().numpy()[0]
            prob = torch.sigmoid(model(image))[0, 0].cpu().numpy()
            pred = (prob >= threshold).astype("float32")
            rgb = tensor_to_rgb(sample["image"])
            title = f"{group_name} {rank} | Dice={row['dice']:.3f} | IoU={row['iou']:.3f}"
            save_prediction_panel(rgb, mask, prob, pred, output_dir / "figures" / group_name / f"{group_name}_{rank:02d}_idx_{idx}.png", title)


def main():
    parser = argparse.ArgumentParser(description="Avaliação U-Net baseline v2")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/unet_baseline/best_unet_baseline.pt"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-figures", type=int, default=5)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {}) or load_json(args.checkpoint.parent / "config.json")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    threshold = args.threshold if args.threshold is not None else float(config.get("threshold", 0.5))
    output_dir = args.output_dir or (args.checkpoint.parent / f"eval_{args.split}")
    output_dir = Path(output_dir)

    pairs = get_split_pairs(args.checkpoint, args.split)
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    dataset = AI4BoundariesSegmentationDataset(
        pairs,
        image_size=int(config.get("image_size", 256)),
        in_channels=int(config.get("in_channels", 3)),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    device = get_device()
    model = load_model_from_checkpoint(args.checkpoint, config, device)

    rows, summary = evaluate(model, loader, device, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(rows, output_dir / "per_sample_metrics.csv")
    save_json({"split": args.split, "num_samples": len(dataset), "threshold": threshold, "summary": summary}, output_dir / "summary_metrics.json")

    thresholds = np.linspace(0.05, 0.95, 19)
    threshold_rows = threshold_sweep(model, loader, device, thresholds)
    save_rows_csv(threshold_rows, output_dir / "threshold_sweep.csv")
    save_threshold_plot(threshold_rows, output_dir / "threshold_sweep.png")
    save_json({"best_by_dice": max(threshold_rows, key=lambda r: r["dice_mean"]), "best_by_iou": max(threshold_rows, key=lambda r: r["iou_mean"])}, output_dir / "threshold_summary.json")
    save_ranked_figures(model, dataset, rows, device, threshold, output_dir, args.num_figures)

    print("Resumo:")
    for key, value in summary.items():
        print(key, value)
    print(f"Arquivos salvos em: {output_dir}")


if __name__ == "__main__":
    main()

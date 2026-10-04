from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import models


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


def force_3_channels(img: np.ndarray) -> np.ndarray:
    channels, _, _ = img.shape

    if channels == 3:
        return img

    if channels > 3:
        return img[:3]

    missing = 3 - channels
    last_band = img[-1:, :, :]
    padding = np.repeat(last_band, repeats=missing, axis=0)

    return np.concatenate([img, padding], axis=0)


class EmptyFieldDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], image_size: int):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        with rasterio.open(image_path) as src:
            img = src.read()

        img = force_3_channels(img)
        img = normalize_image_percentile(img)

        image = torch.from_numpy(img).float().unsqueeze(0)
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # label = 1 significa "tem talhão"; label = 0 significa "vazio"
        label = 1.0 if (mask > 0).sum() > 0 else 0.0

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def build_classifier(name: str, pretrained: bool):
    name = name.lower()

    if name == "resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)

        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    if name == "resnet34":
        if pretrained:
            weights = models.ResNet34_Weights.DEFAULT
            model = models.resnet34(weights=weights)
        else:
            model = models.resnet34(weights=None)

        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    raise ValueError(f"Modelo não suportado: {name}")


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int32)
    labels = labels.astype(np.int32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    eps = 1e-7

    accuracy = (tp + tn) / max(len(labels), 1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    empty_false_positive_rate = fp / (fp + tn + eps)
    empty_recall = tn / (tn + fp + eps)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision_non_empty": precision,
        "recall_non_empty": recall,
        "f1_non_empty": f1,
        "empty_false_positive_rate": empty_false_positive_rate,
        "empty_recall": empty_recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold: float):
    model.eval()

    all_probs = []
    all_labels = []
    rows = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy()

        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

        for image_path, mask_path, prob, label in zip(
            batch["image_path"],
            batch["mask_path"],
            probs,
            labels,
        ):
            rows.append({
                "image_path": image_path,
                "mask_path": mask_path,
                "prob_non_empty": float(prob),
                "label_non_empty": float(label),
            })

    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)

    metrics = compute_metrics(all_probs, all_labels, threshold)

    return metrics, rows, all_probs, all_labels


def save_csv(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Teste 014 — Classificador vazio vs não vazio")

    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ai4boundaries"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = build_pairs(args.data_root, "train")
    val_pairs = build_pairs(args.data_root, "val")

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")

    train_dataset = EmptyFieldDataset(train_pairs, image_size=args.image_size)
    val_dataset = EmptyFieldDataset(val_pairs, image_size=args.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_classifier(args.model_name, pretrained=args.pretrained).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_f1 = -1.0
    best_path = args.output_dir / "best_empty_classifier.pt"

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_losses = []

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        val_metrics, val_rows, val_probs, val_labels = evaluate(
            model,
            val_loader,
            device,
            threshold=args.threshold,
        )

        train_loss = float(np.mean(train_losses))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"loss={train_loss:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"f1={val_metrics['f1_non_empty']:.4f} | "
            f"empty_fp_rate={val_metrics['empty_false_positive_rate']:.4f}"
        )

        if val_metrics["f1_non_empty"] > best_f1:
            best_f1 = val_metrics["f1_non_empty"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "pretrained": args.pretrained,
                    "image_size": args.image_size,
                    "threshold": args.threshold,
                },
                best_path,
            )

            save_csv(val_rows, args.output_dir / "val_predictions_best.csv")

            print(f"Novo melhor modelo salvo em: {best_path}")

    save_csv(history, args.output_dir / "training_history.csv")

    # Recarrega melhor modelo para sweep final
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_rows, val_probs, val_labels = evaluate(
        model,
        val_loader,
        device,
        threshold=args.threshold,
    )

    sweep_rows = []

    for thr in np.arange(0.1, 0.96, 0.05):
        sweep_rows.append(compute_metrics(val_probs, val_labels, float(thr)))

    save_csv(sweep_rows, args.output_dir / "threshold_sweep.csv")
    save_csv(val_rows, args.output_dir / "val_predictions.csv")

    summary = {
        "best_f1_non_empty": best_f1,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nArquivos salvos em:", args.output_dir)
    print("- best_empty_classifier.pt")
    print("- training_history.csv")
    print("- threshold_sweep.csv")
    print("- val_predictions.csv")


if __name__ == "__main__":
    main()
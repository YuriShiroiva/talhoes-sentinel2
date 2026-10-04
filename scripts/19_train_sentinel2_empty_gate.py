from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_pct(a, p_low=2, p_high=98):
    a = a.astype(np.float32)
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.percentile(v, [p_low, p_high])
    if np.isclose(lo, hi):
        return np.zeros_like(a, dtype=np.float32)
    a = np.clip(a, lo, hi)
    a = (a - lo) / (hi - lo)
    return np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def size_class(r):
    if r == 0:
        return "empty"
    if r < 0.01:
        return "very_small_<1%"
    if r < 0.05:
        return "small_1_5%"
    if r < 0.25:
        return "medium_5_25%"
    if r < 0.75:
        return "large_25_75%"
    return "very_large_>75%"


def index_split(root: Path, split: str):
    img_dir = root / "images" / split
    msk_dir = root / "masks" / split
    csv_path = root / "metadata" / f"{split}.csv"

    if csv_path.exists():
        names = pd.read_csv(csv_path)["image"].astype(str).tolist()
    else:
        names = [p.name for p in sorted(img_dir.glob("*.nc"))]

    rows = []
    for name in tqdm(names, desc=f"Indexando {split}"):
        img = img_dir / name
        msk = msk_dir / name.replace(".nc", ".tif")
        if not img.exists() or not msk.exists():
            continue
        with rasterio.open(msk) as src:
            m = src.read(1)
        ratio = float((m > 0).mean())
        rows.append({
            "image_path": str(img),
            "mask_path": str(msk),
            "image_name": name,
            "positive_ratio": ratio,
            "is_empty": int(ratio == 0),
            "size_class": size_class(ratio),
        })
    return pd.DataFrame(rows)


class S2GateDataset(Dataset):
    def __init__(self, df, bands, time_mode="stack", time_index=-1, image_size=256, augment=False):
        self.df = df.reset_index(drop=True)
        self.bands = bands
        self.time_mode = time_mode
        self.time_index = time_index
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def read_img(self, path):
        ds = xr.open_dataset(path)
        ch = []
        try:
            for b in self.bands:
                d = ds[b]
                if "time" in d.dims:
                    if self.time_mode == "stack":
                        vals = d.values
                        for t in range(vals.shape[0]):
                            ch.append(norm_pct(vals[t]))
                    elif self.time_mode == "median":
                        ch.append(norm_pct(d.median(dim="time", skipna=True).values))
                    elif self.time_mode == "mean":
                        ch.append(norm_pct(d.mean(dim="time", skipna=True).values))
                    elif self.time_mode == "index":
                        ch.append(norm_pct(d.isel(time=self.time_index).values))
                    else:
                        raise ValueError(self.time_mode)
                else:
                    ch.append(norm_pct(d.values))
        finally:
            ds.close()
        return np.stack(ch, axis=0).astype(np.float32)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = torch.from_numpy(self.read_img(r.image_path)).float()
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            if random.random() < 0.5:
                x = torch.flip(x, dims=[1])
            k = random.randint(0, 3)
            if k:
                x = torch.rot90(x, k=k, dims=[1, 2])
        y = torch.tensor([float(r.is_empty)], dtype=torch.float32)
        return {
            "image": x,
            "label_empty": y,
            "image_path": r.image_path,
            "mask_path": r.mask_path,
            "positive_ratio": torch.tensor([float(r.positive_ratio)], dtype=torch.float32),
            "size_class": r.size_class,
        }


class GateCNN(nn.Module):
    def __init__(self, in_ch, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, 2, 1, bias=False), nn.BatchNorm2d(48), nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, 3, 2, 1, bias=False), nn.BatchNorm2d(96), nn.SiLU(inplace=True),
            nn.Conv2d(96, 160, 3, 2, 1, bias=False), nn.BatchNorm2d(160), nn.SiLU(inplace=True),
            nn.Conv2d(160, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU(inplace=True),
            nn.Conv2d(256, 384, 3, 2, 1, bias=False), nn.BatchNorm2d(384), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(384, 1),
        )

    def forward(self, x):
        return self.net(x)


def threshold_metrics(prob, label, thr):
    pred = (prob >= thr).astype(np.int64)
    label = label.astype(np.int64)
    tp = int(((pred == 1) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())  # falso vazio: zera imagem com talhão
    fn = int(((pred == 0) & (label == 1)).sum())  # perdeu vazio
    acc = (tp + tn) / max(len(label), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    false_empty_rate = fp / max((label == 0).sum(), 1)
    missed_empty_rate = fn / max((label == 1).sum(), 1)
    return {
        "threshold": float(thr),
        "total": int(len(label)),
        "empty_count": int((label == 1).sum()),
        "non_empty_count": int((label == 0).sum()),
        "tp_empty": tp,
        "tn_non_empty": tn,
        "fp_false_empty": fp,
        "fn_missed_empty": fn,
        "accuracy": float(acc),
        "precision_empty": float(prec),
        "recall_empty": float(rec),
        "f1_empty": float(f1),
        "false_empty_rate_nonempty": float(false_empty_rate),
        "missed_empty_rate": float(missed_empty_rate),
    }


@torch.no_grad()
def predict(model, loader, device, amp, amp_dtype):
    model.eval()
    rows = []
    for b in tqdm(loader, desc="Predizendo"):
        x = b["image"].to(device, non_blocking=True)
        y = b["label_empty"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            p = torch.sigmoid(model(x))
        p = p.float().squeeze(1).detach().cpu().numpy()
        y = y.squeeze(1).detach().cpu().numpy()
        ratios = b["positive_ratio"].squeeze(1).numpy()
        for i in range(len(p)):
            rows.append({
                "image_path": b["image_path"][i],
                "mask_path": b["mask_path"][i],
                "prob_empty": float(p[i]),
                "label_empty": int(y[i]),
                "positive_ratio": float(ratios[i]),
                "size_class": b["size_class"][i],
            })
    return pd.DataFrame(rows)


def train_epoch(model, loader, opt, crit, device, amp, amp_dtype, scaler):
    model.train()
    total = 0.0
    n = 0
    for b in tqdm(loader, desc="Treino", leave=False):
        x = b["image"].to(device, non_blocking=True)
        y = b["label_empty"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            loss = crit(model(x), y)
        if amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Gate vazio/não vazio Sentinel-2. label_empty=1 significa máscara vazia.")
    ap.add_argument("--root", type=Path, default=Path("data/raw/ai4boundaries/sentinel2"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/sentinel2/test_031_gate_empty_classifier_stack30"))
    ap.add_argument("--bands", default="B2,B3,B4,B8,NDVI")
    ap.add_argument("--time-mode", default="stack", choices=["stack", "median", "mean", "index"])
    ap.add_argument("--time-index", type=int, default=-1)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--use-balanced-sampler", action="store_true")
    ap.add_argument("--empty-pos-weight", type=float, default=0.0, help="0 calcula non_empty/empty.")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bands = [x.strip() for x in args.bands.split(",") if x.strip()]
    in_ch = len(bands) * 6 if args.time_mode == "stack" else len(bands)

    train_df = index_split(args.root, "train")
    val_df = index_split(args.root, "val")
    test_df = index_split(args.root, "test")

    idx_dir = args.output_dir / "indices"
    idx_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(idx_dir / "train_items.csv", index=False)
    val_df.to_csv(idx_dir / "val_items.csv", index=False)
    test_df.to_csv(idx_dir / "test_items.csv", index=False)

    n_empty = int(train_df.is_empty.sum())
    n_non = len(train_df) - n_empty
    pos_weight = args.empty_pos_weight if args.empty_pos_weight > 0 else n_non / max(n_empty, 1)

    print("\nDistribuição")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        e = int(df.is_empty.sum())
        print(f"{name}: total={len(df)} empty={e} non_empty={len(df)-e}")
    print(f"in_channels={in_ch}")
    print(f"pos_weight_empty={pos_weight:.4f}")

    train_ds = S2GateDataset(train_df, bands, args.time_mode, args.time_index, args.image_size, augment=True)
    val_ds = S2GateDataset(val_df, bands, args.time_mode, args.time_index, args.image_size)
    test_ds = S2GateDataset(test_df, bands, args.time_mode, args.time_index, args.image_size)

    sampler = None
    shuffle = True
    if args.use_balanced_sampler:
        labels = train_df.is_empty.values.astype(int)
        counts = np.bincount(labels, minlength=2)
        weights = 1.0 / np.maximum(counts, 1)
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights[labels]), len(labels), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GateCNN(in_ch, dropout=args.dropout).to(device)

    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.amp_dtype == "fp16")

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
    best_score = -1e9
    history = []

    config = vars(args).copy()
    config.update({"bands": bands, "in_channels": in_ch, "pos_weight_empty": pos_weight})
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, crit, device, args.amp, amp_dtype, scaler)
        val_pred = predict(model, val_loader, device, args.amp, amp_dtype)

        rows = [threshold_metrics(val_pred.prob_empty.values, val_pred.label_empty.values, t) for t in thresholds]
        met = pd.DataFrame(rows)
        # Gate conservador: recompensa recall de vazios, penaliza muito falsos vazios.
        met["gate_score"] = met["recall_empty"] - 8.0 * met["false_empty_rate_nonempty"]
        best = met.sort_values("gate_score", ascending=False).iloc[0].to_dict()

        hist_row = {
            "epoch": ep, "train_loss": loss,
            "best_threshold_val": best["threshold"],
            "best_gate_score_val": best["gate_score"],
            "best_recall_empty_val": best["recall_empty"],
            "best_false_empty_rate_nonempty_val": best["false_empty_rate_nonempty"],
            "best_precision_empty_val": best["precision_empty"],
            "best_f1_empty_val": best["f1_empty"],
        }
        history.append(hist_row)

        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
        met.to_csv(args.output_dir / f"val_threshold_metrics_epoch_{ep:03d}.csv", index=False)

        print(
            f"Epoch {ep:03d}/{args.epochs:03d} | loss={loss:.4f} | "
            f"thr={best['threshold']:.2f} | recall_empty={best['recall_empty']:.4f} | "
            f"false_empty_rate_nonempty={best['false_empty_rate_nonempty']:.4f} | "
            f"precision_empty={best['precision_empty']:.4f} | score={best['gate_score']:.4f}"
        )

        if best["gate_score"] > best_score:
            best_score = best["gate_score"]
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "config": config,
                "best_val_row": best,
                "best_score": best_score,
            }, args.output_dir / "best_empty_gate.pt")
            print(f"Novo melhor gate salvo. epoch={ep} score={best_score:.4f}")

    ckpt = torch.load(args.output_dir / "best_empty_gate.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        pred = predict(model, loader, device, args.amp, amp_dtype)
        pred.to_csv(args.output_dir / f"{split_name}_empty_gate_predictions.csv", index=False)
        met = pd.DataFrame([threshold_metrics(pred.prob_empty.values, pred.label_empty.values, t) for t in thresholds])
        met.to_csv(args.output_dir / f"{split_name}_threshold_metrics.csv", index=False)
        print(f"\n{split_name.upper()} threshold metrics:")
        print(met.to_string(index=False))

    print("\nTreino finalizado.")
    print(f"Arquivos salvos em: {args.output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import torch  # importar torch ANTES de rasterio/GDAL (evita conflito de DLL no Windows)
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_boundary.data.dataset import normalize_image_percentile


def _load_pred_module():
    path = PROJECT_ROOT / "scripts" / "44_predict_sentinel2_fields_end_to_end.py"
    spec = importlib.util.spec_from_file_location("s2pred", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_stack(wa, wb):
    """FTW (2 janelas, 4 bandas B,G,R,NIR) -> 30 canais com a MESMA preprocessing do treino:
    ordem band_time (cada banda, suas 6 datas) + percentil 2-98 por canal."""
    def five_raw(w):
        b2, b3, b4, b8 = [w[i].astype(np.float32) for i in range(4)]  # B,G,R,NIR cruas
        ndvi = ((b8 - b4) / (b8 + b4 + 1e-6)).astype(np.float32)
        return [b2, b3, b4, b8, ndvi]
    fa, fb = five_raw(wa), five_raw(wb)
    per_t = [fa, fa, fa, fb, fb, fb]  # 6 datas (janelas A,A,A,B,B,B)
    ch = []
    for b in range(5):          # band_time: para cada banda...
        for t in range(6):      # ...as 6 datas
            ch.append(per_t[t][b])
    x = np.nan_to_num(np.stack(ch, 0).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_image_percentile(x)


class FTWDataset(Dataset):
    def __init__(self, root: Path, aoi_ids):
        self.root = root
        self.ids = aoi_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        a = self.ids[i]
        with rasterio.open(self.root / "s2_images" / "window_a" / f"{a}.tif") as s:
            wa = s.read()
        with rasterio.open(self.root / "s2_images" / "window_b" / f"{a}.tif") as s:
            wb = s.read()
        with rasterio.open(self.root / "label_masks" / "semantic_2class" / f"{a}.tif") as s:
            m = s.read(1)
        x = torch.from_numpy(build_stack(wa, wb)).float()
        target = torch.from_numpy((m == 1).astype(np.float32))[None]
        valid = torch.from_numpy((m != 0).astype(np.float32))[None]
        return {"x": x, "target": target, "valid": valid}


def dice_bce_loss(logits, target, valid):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    y = target * valid
    inter = (p * y).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
    return bce + dice


@torch.no_grad()
def eval_split(model, loader, device, thr=0.5):
    model.eval()
    tp = fp = fn = 0.0
    dices = []
    for b in loader:
        x = b["x"].to(device); y = b["target"].to(device); v = b["valid"].to(device)
        logits = model(x)
        if logits.shape[1] > 1:
            logits = logits[:, 0:1]
        pred = ((torch.sigmoid(logits) >= thr).float()) * v
        yv = y * v
        for k in range(pred.shape[0]):
            pk, yk = pred[k], yv[k]
            i = (pk * yk).sum().item(); pp = pk.sum().item(); yy = yk.sum().item()
            d = (2 * i + 1) / (pp + yy + 1)
            dices.append(d)
            tp += i; fp += pp - i; fn += yy - i
    prec = tp / (tp + fp + 1e-7); rec = tp / (tp + fn + 1e-7)
    return {"dice": float(np.mean(dices)), "iou": tp / (tp + fp + fn + 1e-7),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec + 1e-7)}


def main():
    ap = argparse.ArgumentParser(description="Fine-tuna modelo Sentinel-2 pre-treinado no FTW-Brasil.")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "ftw" / "brazil")
    ap.add_argument("--zip", type=Path, default=PROJECT_ROOT / "data" / "ftw" / "brazil.zip")
    ap.add_argument("--pretrained", type=Path,
                    default=PROJECT_ROOT / "outputs/sentinel2/test_025_unet_s2_stack30_a100/best_sentinel2_unet.pt")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sentinel2" / "ftw_brazil_finetune")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None)
    args = ap.parse_args()
    if sys.platform.startswith("win"):
        args.num_workers = 0

    P = _load_pred_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # split do parquet
    z = zipfile.ZipFile(args.zip)
    df = pd.read_parquet(io.BytesIO(z.read("chips_brazil.parquet")))
    splits = {s: df[df["split"] == s]["aoi_id"].tolist() for s in ["train", "val", "test"]}
    if args.max_train:
        splits["train"] = splits["train"][: args.max_train]
    print({k: len(v) for k, v in splits.items()})

    model, cfg = P.load_model(args.pretrained, device)
    print(f"Pretrained: {cfg.get('architecture')}/{cfg.get('encoder_name')} in_ch={cfg.get('in_channels')}")

    tl = DataLoader(FTWDataset(args.root, splits["train"]), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True)
    vl = DataLoader(FTWDataset(args.root, splits["val"]), batch_size=args.batch_size, num_workers=args.num_workers)
    te = DataLoader(FTWDataset(args.root, splits["test"]), batch_size=args.batch_size, num_workers=args.num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    base = eval_split(model, vl, device)
    print(f"val ANTES do fine-tune: Dice={base['dice']:.3f}")
    best = -1.0; hist = []
    from tqdm import tqdm
    for ep in range(1, args.epochs + 1):
        model.train(); tot = n = 0
        for b in tqdm(tl, desc=f"ft {ep}", leave=False):
            x = b["x"].to(device); y = b["target"].to(device); v = b["valid"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if logits.shape[1] > 1:
                logits = logits[:, 0:1]
            loss = dice_bce_loss(logits, y, v)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss); n += 1
        sched.step()
        vm = eval_split(model, vl, device)
        hist.append({"epoch": ep, "train_loss": tot / n, **{f"val_{k}": v for k, v in vm.items()}})
        pd.DataFrame(hist).to_csv(args.output_dir / "history.csv", index=False)
        print(f"ep {ep:02d}/{args.epochs} loss={tot/n:.3f} val_dice={vm['dice']:.3f} val_iou={vm['iou']:.3f}")
        if vm["dice"] > best:
            best = vm["dice"]
            torch.save({"model_state_dict": model.state_dict(), "config": cfg, "val_dice": best},
                       args.output_dir / "best_ftw_finetuned.pt")
    # avaliacao final no TEST
    ck = torch.load(args.output_dir / "best_ftw_finetuned.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    tm = eval_split(model, te, device)
    summary = {"dataset": "FTW-brazil (fine-tuned)", "val_dice_antes": base["dice"], "best_val_dice": best,
               "test": tm, "n_test": len(splits["test"]), "epochs": args.epochs, "lr": args.lr}
    (args.output_dir / "summary_finetune.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== TEST (FTW-Brasil) apos fine-tune ===")
    for k in ["dice", "iou", "precision", "recall", "f1"]:
        print(f"  {k:10s} {tm[k]:.3f}")
    print(f"Salvo: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

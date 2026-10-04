from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import segmentation_models_pytorch as smp

# ==========================================================================
# Script 56 - Treino RGB "Deep Watershed": area + ENERGIA (dist. normalizada).
#   canal 0: area do talhao (BCE + Tversky)
#   canal 1: energia [0,1] (regressao L1) -> picos = centros -> 1 semente/talhao
# Melhorias para RGB: augmentation FOTOMETRICA (brilho/contraste/gamma/canal)
#   e MULTI-ESCALA (recupera robustez a escala perdida no GSD fixo).
# ==========================================================================


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_pairs(root: Path, split: str):
    pairs = []
    for img in sorted((root / "images" / split).glob("*.tif")):
        m = root / "masks" / split / img.name.replace("_ortho_", "_ortholabel_")
        if m.exists():
            pairs.append((img, m))
    if not pairs:
        raise RuntimeError(f"Sem pares em {root/'images'/split}")
    return pairs


def ratio_class(r):
    if r == 0: return "empty"
    if r < 0.05: return "small"
    if r < 0.25: return "medium"
    if r < 0.75: return "large"
    return "almost_full"


class EnergyDataset(Dataset):
    def __init__(self, pairs, image_size=512, augment=False, scale_jitter=True):
        self.pairs = pairs; self.image_size = image_size; self.augment = augment; self.scale_jitter = scale_jitter

    def __len__(self): return len(self.pairs)

    def _read(self, img_p, mask_p):
        with rasterio.open(img_p) as s:
            rgb = s.read([1, 2, 3]).astype(np.float32)
        if rgb.max() > 2: rgb /= 255.0
        rgb = np.clip(rgb, 0, 1)
        with rasterio.open(mask_p) as s:
            area = (s.read(1) > 0).astype(np.float32)
            energy = s.read(2).astype(np.float32) / 255.0
        return rgb, np.stack([area, energy], 0).astype(np.float32)

    def _photometric(self, x):  # x: 3,H,W em [0,1]
        # brilho/contraste/gamma
        if random.random() < 0.5: x = x + random.uniform(-0.18, 0.18)
        if random.random() < 0.5: x = (x - 0.5) * random.uniform(0.7, 1.3) + 0.5
        if random.random() < 0.5:
            x = torch.clamp(x, 0, 1) ** random.uniform(0.7, 1.4)
        # COR AGRESSIVA: forca aprender textura/forma, nao "verde" -> generaliza p/ talhao NU
        if random.random() < 0.7:  # dessatura (mistura com cinza), as vezes quase remove a cor
            gray = x.mean(0, keepdim=True); s = random.uniform(0.1, 1.0); x = s * x + (1 - s) * gray
        if random.random() < 0.5:  # peso por canal (quebra dependencia do verde)
            x = x * torch.tensor([random.uniform(0.8, 1.2) for _ in range(3)]).view(3, 1, 1)
        if random.random() < 0.2:  # cinza total
            x = x.mean(0, keepdim=True).repeat(3, 1, 1)
        if random.random() < 0.3:
            x = x + torch.randn_like(x) * 0.025
        return torch.clamp(x, 0, 1)

    def _scale_jitter(self, x, y):  # zoom in/out e recorta/preenche para image_size
        s = random.uniform(0.75, 1.3)
        H = self.image_size
        nh = max(8, int(round(H * s)))
        x = F.interpolate(x[None], size=(nh, nh), mode="bilinear", align_corners=False)[0]
        ya = F.interpolate(y[0:1][None], size=(nh, nh), mode="nearest")[0]
        ye = F.interpolate(y[1:2][None], size=(nh, nh), mode="bilinear", align_corners=False)[0]
        y = torch.cat([ya, ye], 0)
        if nh >= H:  # crop aleatorio
            top = random.randint(0, nh - H); left = random.randint(0, nh - H)
            return x[:, top:top + H, left:left + H], y[:, top:top + H, left:left + H]
        pad = H - nh; t = random.randint(0, pad); l = random.randint(0, pad)  # pad para H
        x = F.pad(x, (l, pad - l, t, pad - t), mode="reflect")
        y = F.pad(y, (l, pad - l, t, pad - t), mode="constant", value=0)
        return x, y

    def __getitem__(self, i):
        img_p, mask_p = self.pairs[i]
        rgb, tgt = self._read(img_p, mask_p)
        x = torch.from_numpy(rgb); y = torch.from_numpy(tgt)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
            ya = F.interpolate(y[0:1][None], size=(self.image_size, self.image_size), mode="nearest")[0]
            ye = F.interpolate(y[1:2][None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
            y = torch.cat([ya, ye], 0)
        if self.augment:
            if self.scale_jitter and random.random() < 0.5:
                x, y = self._scale_jitter(x, y)
            if random.random() < 0.5: x = torch.flip(x, [2]); y = torch.flip(y, [2])
            if random.random() < 0.5: x = torch.flip(x, [1]); y = torch.flip(y, [1])
            k = random.randint(0, 3)
            if k: x = torch.rot90(x, k, [1, 2]); y = torch.rot90(y, k, [1, 2])
            x = self._photometric(x)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        return {"image": x.float(), "target": y.float(), "area_ratio": float((y[0] > 0).float().mean())}


def tversky(logits, t, a=0.35, b=0.65, eps=1e-7):
    p = torch.sigmoid(logits).flatten(1); y = t.flatten(1)
    tp = (p * y).sum(1); fp = (p * (1 - y)).sum(1); fn = ((1 - p) * y).sum(1)
    return (1 - (tp + eps) / (tp + a * fp + b * fn + eps)).mean()


def loss_fn(logits, target, energy_weight):
    area_l, en_l = logits[:, 0:1], logits[:, 1:2]
    area_t, en_t = target[:, 0:1], target[:, 1:2]
    area = F.binary_cross_entropy_with_logits(area_l, area_t) + tversky(area_l, area_t)
    energy = F.l1_loss(torch.sigmoid(en_l), en_t)
    return area + energy_weight * energy, float(area.item()), float(energy.item())


def area_dice(logits, target, thr=0.5, eps=1e-7):
    p = (torch.sigmoid(logits[:, 0:1]) >= thr).float().flatten(1); y = target[:, 0:1].flatten(1)
    tp = (p * y).sum(1); d = (2 * tp + eps) / (2 * tp + (p * (1 - y)).sum(1) + ((1 - p) * y).sum(1) + eps)
    return float(d.mean().item())


def energy_mae(logits, target):
    return float(F.l1_loss(torch.sigmoid(logits[:, 1:2]), target[:, 1:2]).item())


def main():
    ap = argparse.ArgumentParser(description="Treino RGB area+energia (Deep Watershed) com aug fotometrica/multi-escala.")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--encoder-name", default="resnet34")
    ap.add_argument("--architecture", default="unet")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--energy-weight", type=float, default=1.5)
    ap.add_argument("--no-scale-jitter", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if sys.platform.startswith("win") and args.num_workers > 0:
        print(f"[aviso] Windows: num_workers={args.num_workers} pode travar. Usando 0.")
        args.num_workers = 0
    set_seed(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = build_pairs(args.root, "train"); va = build_pairs(args.root, "val")
    print(f"train={len(tr)} val={len(va)} | device={device}")

    # sampler balanceado por classe de area
    cls = []
    for _, mp in tqdm(tr, desc="ratios", leave=False):
        with rasterio.open(mp) as s: r = float((s.read(1) > 0).mean())
        cls.append(ratio_class(r))
    cw = {"empty": 0.6, "small": 3.0, "medium": 2.5, "large": 1.0, "almost_full": 0.5}
    weights = torch.tensor([cw.get(c, 1.0) for c in cls], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    tl = DataLoader(EnergyDataset(tr, args.image_size, True, not args.no_scale_jitter),
                    batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                    pin_memory=True, persistent_workers=args.num_workers > 0)
    vl = DataLoader(EnergyDataset(va, args.image_size, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    model = smp.create_model(arch={"unet": "Unet", "unetplusplus": "UnetPlusPlus"}.get(args.architecture, "Unet"),
                             encoder_name=args.encoder_name, encoder_weights="imagenet",
                             in_channels=3, classes=2, activation=None).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    cfg = {"task": "rgb_area_energy", "architecture": args.architecture, "encoder_name": args.encoder_name,
           "in_channels": 3, "classes": 2, "output_channels": {"0": "area", "1": "energy"},
           "energy_weight": args.energy_weight, "image_size": args.image_size}
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    hist = []; best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train(); tl_loss = ta = te = n = 0
        for b in tqdm(tl, desc=f"train {ep}", leave=False):
            x = b["image"].to(device); y = b["target"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x); loss, la, le = loss_fn(logits, y, args.energy_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl_loss += float(loss.item()); ta += la; te += le; n += 1
        sched.step()
        model.eval(); vd = vm = vn = 0
        with torch.no_grad():
            for b in tqdm(vl, desc=f"val {ep}", leave=False):
                x = b["image"].to(device); y = b["target"].to(device)
                logits = model(x); vd += area_dice(logits, y); vm += energy_mae(logits, y); vn += 1
        vd /= vn; vm /= vn; score = vd + max(0.0, 1.0 - 2.0 * vm)
        hist.append({"epoch": ep, "train_loss": tl_loss / n, "train_area": ta / n, "train_energy": te / n,
                     "val_area_dice": vd, "val_energy_mae": vm, "score": score, "lr": opt.param_groups[0]["lr"]})
        pd.DataFrame(hist).to_csv(args.output_dir / "history.csv", index=False)
        print(f"ep {ep:02d}/{args.epochs} | loss={tl_loss/n:.3f} | val_area_dice={vd:.3f} | val_energy_mae={vm:.3f} | score={score:.3f}")
        torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "config": cfg}, args.output_dir / "last_energy_model.pt")
        if score > best:
            best = score
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "config": cfg, "score": best},
                       args.output_dir / "best_energy_model.pt")
            print(f"  novo melhor: score={best:.3f} (ep {ep})")
    print(f"Treino done. melhor score={best:.3f} | {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

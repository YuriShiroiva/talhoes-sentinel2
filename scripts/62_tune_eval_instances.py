"""Fase 2 (parte 2) — calibra o pos-processamento da separacao de instancias no VAL,
avalia no TEST e gera figura qualitativa (baseline semantico vs instancias separadas).

Roda apos scripts/61 ter salvo best_boundary_3class.pt.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import torch  # antes de rasterio

import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S59 = _load("s59", PROJECT_ROOT / "scripts/59_finetune_on_ftw.py")
S60 = _load("s60", PROJECT_ROOT / "scripts/60_object_level_metrics_ftw.py")


def instances_tunable(prob3, seed_min, field_thr, min_area):
    p_int, p_bnd, p_bg = prob3
    field = p_bg < field_thr
    interior = np.argmax(prob3, 0) == 0
    seeds, n = ndimage.label(interior, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(field.shape, np.int32)
    sizes = np.bincount(seeds.ravel())
    for s in np.where(sizes < seed_min)[0]:
        if s != 0:
            seeds[seeds == s] = 0
    if seeds.max() == 0:
        return np.zeros(field.shape, np.int32)
    idx = ndimage.distance_transform_edt(seeds == 0, return_indices=True)[1]
    grown = seeds[tuple(idx)]
    grown[~field] = 0
    out = np.zeros_like(grown); new = 1
    for i in [v for v in np.unique(grown) if v > 0]:
        m = grown == i
        if int(m.sum()) >= min_area:
            out[m] = new; new += 1
    return out.astype(np.int32)


def load_3class(ckpt, dev):
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=30, classes=3).to(dev)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


@torch.no_grad()
def probs3(model, root, a, dev):
    with rasterio.open(root / "s2_images/window_a" / f"{a}.tif") as s: wa = s.read()
    with rasterio.open(root / "s2_images/window_b" / f"{a}.tif") as s: wb = s.read()
    x = torch.from_numpy(S59.build_stack(wa, wb)).unsqueeze(0).float().to(dev)
    return torch.softmax(model(x), 1)[0].cpu().numpy()


def eval_cfg(model, ids, root, z, dev, seed_min, field_thr, min_area):
    agg = dict(tp=0, fp=0, fn=0, sum_iou=0.0, n_gt=0, n_pred=0, under=0, over=0)
    for a in ids:
        pr = probs3(model, root, a, dev)
        pred = instances_tunable(pr, seed_min, field_thr, min_area)
        with rasterio.open(io.BytesIO(z.read(f"label_masks/instance/{a}.tif"))) as s:
            inst = s.read(1)
        gt = S60.instances_from_label(inst, min_area)[0]
        m = S60.object_metrics(pred, gt)
        for k in agg: agg[k] += m[k]
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    pq = agg["sum_iou"] / (tp + 0.5 * fp + 0.5 * fn + 1e-9)
    return {"obj_f1": f1, "precision": prec, "recall": rec, "pq": pq,
            "n_gt": agg["n_gt"], "n_pred": agg["n_pred"], "under": agg["under"], "over": agg["over"]}


def colorize(lab, seed=0):
    rng = np.random.RandomState(seed)
    out = np.zeros((*lab.shape, 3), np.float32)
    for i in [v for v in np.unique(lab) if v > 0]:
        out[lab == i] = rng.rand(3) * 0.7 + 0.3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=PROJECT_ROOT / "outputs/sentinel2/ftw_boundary_instances/best_boundary_3class.pt")
    ap.add_argument("--field-ckpt", type=Path, default=PROJECT_ROOT / "outputs/sentinel2/ftw_brazil_finetune/best_ftw_finetuned.pt")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/ftw/brazil")
    ap.add_argument("--zip", type=Path, default=PROJECT_ROOT / "data/ftw/brazil.zip")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/sentinel2/ftw_object_metrics")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = zipfile.ZipFile(args.zip)
    df = pd.read_parquet(io.BytesIO(z.read("chips_brazil.parquet")))
    val = df[df["split"] == "val"]["aoi_id"].tolist()
    test = df[df["split"] == "test"]["aoi_id"].tolist()
    model = load_3class(args.ckpt, dev)

    # 1) grid search no VAL
    grid = [(sm, ft, ma) for sm in [30, 80, 150, 250] for ft in [0.4, 0.5] for ma in [50, 120]]
    print(f"grid search no val ({len(val)} tiles, {len(grid)} configs)...")
    best = None
    for sm, ft, ma in grid:
        r = eval_cfg(model, val, args.root, z, dev, sm, ft, ma)
        tag = f"seed_min={sm} field_thr={ft} min_area={ma}"
        print(f"  {tag:42s} obj_f1={r['obj_f1']:.3f} prec={r['precision']:.3f} rec={r['recall']:.3f}")
        if best is None or r["obj_f1"] > best[0]["obj_f1"]:
            best = (r, (sm, ft, ma))
    rbest, (sm, ft, ma) = best
    print(f"\nMELHOR no val: seed_min={sm} field_thr={ft} min_area={ma} -> obj_f1={rbest['obj_f1']:.3f}")

    # 2) avalia no TEST com a melhor config
    rtest = eval_cfg(model, test, args.root, z, dev, sm, ft, ma)
    summary = {"best_cfg": {"seed_min": sm, "field_thr": ft, "min_area": ma},
               "val": rbest, "test": rtest}
    (args.output_dir / "summary_instances_tuned.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== TEST objeto (instancias separadas, config calibrada) ===")
    for k in ["obj_f1", "precision", "recall", "pq"]:
        print(f"  {k:10s} {rtest[k]:.3f}")

    # 3) figura qualitativa: escolhe tiles do test com mais instancias GT
    field_model, _ = _load("p", PROJECT_ROOT / "scripts/44_predict_sentinel2_fields_end_to_end.py").load_model(args.field_ckpt, dev)
    counts = []
    for a in test:
        with rasterio.open(io.BytesIO(z.read(f"label_masks/instance/{a}.tif"))) as s:
            inst = s.read(1)
        counts.append((len(np.unique(inst)) - 1, a))
    counts.sort(reverse=True)
    sel = [a for _, a in counts[:4]]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(sel), 4, figsize=(11, 2.7 * len(sel)))
    for i, a in enumerate(sel):
        with rasterio.open(args.root / "s2_images/window_a" / f"{a}.tif") as s: wa = s.read()
        rgb = np.clip(np.stack([wa[2], wa[1], wa[0]], -1).astype(np.float32) / 10000 * 3.2, 0, 1) ** 0.85
        with rasterio.open(io.BytesIO(z.read(f"label_masks/instance/{a}.tif"))) as s: inst = s.read(1)
        gt = S60.instances_from_label(inst, ma)[0]
        # baseline semantico (modelo de campo -> componentes conexos)
        with rasterio.open(args.root / "s2_images/window_b" / f"{a}.tif") as s: wb = s.read()
        with torch.no_grad():
            xb = torch.from_numpy(S59.build_stack(wa, wb)).unsqueeze(0).float().to(dev)
            lg = field_model(xb); lg = lg[:, 0:1] if lg.shape[1] > 1 else lg
            fld = (torch.sigmoid(lg)[0, 0].cpu().numpy() >= 0.49)
        sem = S60.relabel(fld, ma)[0]
        pr = probs3(model, args.root, a, dev)
        ins = instances_tunable(pr, sm, ft, ma)
        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(colorize(gt))
        axes[i, 2].imshow(colorize(sem))
        axes[i, 3].imshow(colorize(ins))
        axes[i, 2].set_ylabel(f"{len(np.unique(sem))-1} inst", fontsize=8)
        axes[i, 3].set_ylabel(f"{len(np.unique(ins))-1} inst", fontsize=8)
        for j in range(4): axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    for j, t in enumerate(["Sentinel-2 RGB", "Gabarito (instancias)", "Baseline semantico", "Borda+separacao"]):
        axes[0, j].set_title(t, fontsize=10)
    fig.suptitle("Separacao de talhoes: baseline funde vizinhos; cabeca de borda separa instancias", fontsize=11, y=0.995)
    fig.tight_layout(); fig.savefig(PROJECT_ROOT / "outputs/sentinel2/figura_tese_instancias.png", dpi=150, bbox_inches="tight")
    print("figura: outputs/sentinel2/figura_tese_instancias.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

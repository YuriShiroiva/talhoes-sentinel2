"""Metricas de OBJETO (instancia) no FTW-Brasil: F1@IoU>=0.5, Panoptic Quality,
sub/super-segmentacao. Compara o modelo atual (semantico -> componentes conexos) com
o gabarito de instancia (label_masks/instance).

Uso:
    python scripts/60_object_level_metrics_ftw.py --mode semantic --split test
    python scripts/60_object_level_metrics_ftw.py --mode watershed --split test  (apos a Fase 2)
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import torch  # antes de rasterio (DLL no Windows)

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_boundary.data.dataset import normalize_image_percentile


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def relabel(mask_bool, min_area):
    """Componentes conexos -> rotulos de instancia, removendo pequenos."""
    lab, n = ndimage.label(mask_bool, structure=np.ones((3, 3)))
    if n == 0:
        return lab, 0
    sizes = np.bincount(lab.ravel())
    keep = np.where(sizes >= min_area)[0]
    keep = keep[keep != 0]
    out = np.zeros_like(lab)
    for new_id, old in enumerate(keep, start=1):
        out[lab == old] = new_id
    return out, len(keep)


def instances_from_label(inst_gt, min_area):
    """Gabarito de instancia (parcel-id) -> rotulos compactos, removendo pequenos."""
    ids = np.unique(inst_gt)
    ids = ids[ids > 0]
    out = np.zeros(inst_gt.shape, dtype=np.int32)
    new = 1
    for i in ids:
        m = inst_gt == i
        if int(m.sum()) >= min_area:
            out[m] = new
            new += 1
    return out, new - 1


def object_metrics(pred_lab, gt_lab, iou_thr=0.5):
    """F1@IoU, PQ, e contagem de sub/super-segmentacao por tile."""
    pred_ids = [i for i in np.unique(pred_lab) if i > 0]
    gt_ids = [i for i in np.unique(gt_lab) if i > 0]
    if not gt_ids and not pred_ids:
        return dict(tp=0, fp=0, fn=0, sum_iou=0.0, n_gt=0, n_pred=0, under=0, over=0)
    # areas
    pred_area = {i: int((pred_lab == i).sum()) for i in pred_ids}
    gt_area = {i: int((gt_lab == i).sum()) for i in gt_ids}
    # intersecoes via histograma 2D dos pares (pred,gt)
    inter = {}
    pf = pred_lab.ravel(); gf = gt_lab.ravel()
    mask = (pf > 0) & (gf > 0)
    if mask.any():
        keys = pf[mask].astype(np.int64) * (max(gt_ids) + 1) + gf[mask].astype(np.int64)
        u, c = np.unique(keys, return_counts=True)
        for k, cnt in zip(u, c):
            p = int(k // (max(gt_ids) + 1)); g = int(k % (max(gt_ids) + 1))
            inter[(p, g)] = int(cnt)
    # IoU por par
    tp = 0; sum_iou = 0.0
    matched_gt = set(); matched_pred = set()
    pairs = []
    for (p, g), i in inter.items():
        union = pred_area[p] + gt_area[g] - i
        iou = i / union if union > 0 else 0.0
        if iou >= iou_thr:
            pairs.append((iou, p, g))
    pairs.sort(reverse=True)
    for iou, p, g in pairs:
        if p in matched_pred or g in matched_gt:
            continue
        matched_pred.add(p); matched_gt.add(g); tp += 1; sum_iou += iou
    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    # sub-segmentacao: 1 pred cobre >=2 GT (cada com overlap >=50% da GT)
    under = 0
    for p in pred_ids:
        covered = sum(1 for g in gt_ids if inter.get((p, g), 0) >= 0.5 * gt_area[g])
        if covered >= 2:
            under += 1
    # super-segmentacao: 1 GT coberta por >=2 preds (cada com overlap >=25% do pred)
    over = 0
    for g in gt_ids:
        splits = sum(1 for p in pred_ids if inter.get((p, g), 0) >= 0.25 * pred_area[p])
        if splits >= 2:
            over += 1
    return dict(tp=tp, fp=fp, fn=fn, sum_iou=sum_iou, n_gt=len(gt_ids),
                n_pred=len(pred_ids), under=under, over=over)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["semantic", "watershed"], default="semantic")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", type=Path,
                    default=PROJECT_ROOT / "outputs/sentinel2/ftw_brazil_finetune/best_ftw_finetuned.pt")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/ftw/brazil")
    ap.add_argument("--zip", type=Path, default=PROJECT_ROOT / "data/ftw/brazil.zip")
    ap.add_argument("--threshold", type=float, default=0.49)
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/sentinel2/ftw_object_metrics")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    P = _load("p", PROJECT_ROOT / "scripts/44_predict_sentinel2_fields_end_to_end.py")
    F = _load("f", PROJECT_ROOT / "scripts/59_finetune_on_ftw.py")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = P.load_model(args.checkpoint, dev)
    z = zipfile.ZipFile(args.zip)
    df = pd.read_parquet(io.BytesIO(z.read("chips_brazil.parquet")))
    ids = df[df["split"] == args.split]["aoi_id"].tolist()
    print(f"mode={args.mode} split={args.split} tiles={len(ids)} ckpt={args.checkpoint.name}")

    @torch.no_grad()
    def predict_field(a):
        with rasterio.open(args.root / "s2_images/window_a" / f"{a}.tif") as s: wa = s.read()
        with rasterio.open(args.root / "s2_images/window_b" / f"{a}.tif") as s: wb = s.read()
        x = torch.from_numpy(F.build_stack(wa, wb)).unsqueeze(0).float().to(dev)
        lg = model(x); lg = lg[:, 0:1] if lg.shape[1] > 1 else lg
        return torch.sigmoid(lg)[0, 0].cpu().numpy()

    agg = dict(tp=0, fp=0, fn=0, sum_iou=0.0, n_gt=0, n_pred=0, under=0, over=0)
    rows = []
    for a in ids:
        prob = predict_field(a)
        field = prob >= args.threshold
        if args.mode == "semantic":
            pred_lab, _ = relabel(field, args.min_area)
        else:
            from skimage.segmentation import watershed as _ws  # noqa
            dist = ndimage.distance_transform_edt(field)
            from scipy.ndimage import maximum_filter
            peaks = (dist == maximum_filter(dist, size=9)) & (dist > 3)
            markers, _ = ndimage.label(peaks)
            pred_lab = _ws(-dist, markers, mask=field)
            pred_lab, _ = relabel(pred_lab > 0, args.min_area) if markers.max() == 0 else (pred_lab, 0)
        with rasterio.open(io.BytesIO(z.read(f"label_masks/instance/{a}.tif"))) as s:
            inst = s.read(1)
        gt_lab, _ = instances_from_label(inst, args.min_area)
        m = object_metrics(pred_lab, gt_lab)
        for k in agg:
            agg[k] += m[k]
        rows.append({"tile": a, **m})

    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    pq = agg["sum_iou"] / (tp + 0.5 * fp + 0.5 * fn + 1e-9)
    sq = agg["sum_iou"] / (tp + 1e-9)
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-9)
    summary = {
        "mode": args.mode, "split": args.split, "n_tiles": len(ids),
        "objeto_F1@IoU0.5": f1, "precisao": prec, "recall": rec,
        "PQ": pq, "SQ": sq, "RQ": rq,
        "tp": tp, "fp": fp, "fn": fn,
        "n_gt_total": agg["n_gt"], "n_pred_total": agg["n_pred"],
        "razao_pred_gt": agg["n_pred"] / max(1, agg["n_gt"]),
        "tiles_com_subsegmentacao": agg["under"], "tiles_com_supersegmentacao": agg["over"],
    }
    out = args.output_dir / f"summary_object_{args.mode}_{args.split}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_dir / f"per_tile_object_{args.mode}_{args.split}.csv", index=False)
    print("\n=== METRICAS DE OBJETO (FTW-Brasil " + args.split + ") ===")
    for k in ["objeto_F1@IoU0.5", "precisao", "recall", "PQ", "SQ", "RQ", "razao_pred_gt"]:
        print(f"  {k:22s} {summary[k]:.3f}")
    print(f"  instancias GT={summary['n_gt_total']}  preditas={summary['n_pred_total']}  "
          f"(sub-seg em {summary['tiles_com_subsegmentacao']} tiles)")
    print(f"Salvo: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

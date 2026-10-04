from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.strtree import STRtree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

METERS_PER_DEG_LAT = 111_320.0


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gt_instances(geoms_in_tile, transform, h, w, min_area_px):
    """Rotula cada talhao do gabarito como uma instancia (label 1..N)."""
    gt = np.zeros((h, w), dtype=np.int32)
    nid = 0
    for g in geoms_in_tile:
        single = rasterize([(g, 1)], out_shape=(h, w), fill=0, transform=transform,
                           all_touched=False, dtype="uint8").astype(bool)
        if single.sum() >= min_area_px:
            nid += 1
            gt[single] = nid  # sobrescrita em overlap raro e aceitavel
    return gt, nid


def match_instances(pred, n_pred, gt, n_gt, iou_thr):
    """Casamento guloso pred<->gt por IoU. Retorna (TP, FP, FN, ious_casados)."""
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0, []
    pred_area = np.bincount(pred.ravel(), minlength=n_pred + 1)
    gt_area = np.bincount(gt.ravel(), minlength=n_gt + 1)
    pairs = []
    for p in range(1, n_pred + 1):
        region = pred == p
        if not region.any():
            continue
        for g in np.unique(gt[region]):
            if g == 0:
                continue
            inter = int(np.logical_and(region, gt == g).sum())
            union = int(pred_area[p] + gt_area[g] - inter)
            if union > 0:
                pairs.append((inter / union, p, int(g)))
    pairs.sort(reverse=True)
    used_p, used_g, matched = set(), set(), []
    for iou, p, g in pairs:
        if iou < iou_thr or p in used_p or g in used_g:
            continue
        used_p.add(p); used_g.add(g); matched.append(iou)
    tp = len(matched)
    return tp, n_pred - tp, n_gt - tp, matched


def main():
    ap = argparse.ArgumentParser(description="Metricas de OBJETO (F1@IoU>=0.5) + holdout espacial.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="Dataset com images/<split> e metadata/tiles_index.csv.")
    ap.add_argument("--fields-csv", type=Path, default=PROJECT_ROOT / "fields" / "reais" / "fields_yuri.csv")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--area-thr", type=float, default=0.5)
    ap.add_argument("--bnd-thr", type=float, default=0.5)
    ap.add_argument("--min-seed-dist", type=int, default=20)
    ap.add_argument("--min-area-px", type=int, default=80)
    ap.add_argument("--energy", action="store_true", help="Modelo area+energia (separa por picos de energia).")
    ap.add_argument("--energy-seed-thr", type=float, default=0.35)
    ap.add_argument("--min-dist-to-train-m", type=float, default=2000.0,
                    help="Tile val a >= esta distancia de qualquer tile de treino conta como holdout espacial.")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    P = _load("p54", PROJECT_ROOT / "scripts" / "54_predict_rgb_instances_and_panels.py")
    B = _load("b51", PROJECT_ROOT / "scripts" / "51_build_rgb_dataset_from_fields.py")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, cfg = P.load_model(args.checkpoint, device)
    print(f"Modelo: {cfg.get('architecture')}/{cfg.get('encoder_name')} | {device}")

    print("Lendo fields ...")
    geoms = B.load_fields(args.fields_csv, 20000, None)
    tree = STRtree(geoms)

    # centros dos tiles de treino (para holdout espacial)
    idx_csv = args.root / "metadata" / "tiles_index.csv"
    train_centers = []
    if idx_csv.exists():
        for r in csv.DictReader(idx_csv.open(encoding="utf-8")):
            if r.get("split") == "train":
                train_centers.append(((float(r["min_lon"]) + float(r["max_lon"])) / 2.0,
                                      (float(r["min_lat"]) + float(r["max_lat"])) / 2.0))
    train_centers = np.array(train_centers) if train_centers else np.zeros((0, 2))

    def is_holdout(clon, clat):
        if len(train_centers) == 0:
            return True
        dlat = (train_centers[:, 1] - clat) * METERS_PER_DEG_LAT
        dlon = (train_centers[:, 0] - clon) * METERS_PER_DEG_LAT * np.cos(np.radians(clat))
        return float(np.min(np.hypot(dlat, dlon))) >= args.min_dist_to_train_m

    images = sorted((args.root / "images" / args.split).glob("*.tif"))
    agg = {"all": [0, 0, 0, []], "holdout": [0, 0, 0, []]}  # tp, fp, fn, ious
    n_holdout = 0
    for img in images:
        with rasterio.open(img) as s:
            rgb = np.transpose(s.read([1, 2, 3]), (1, 2, 0)).astype(np.uint8)
            transform = s.transform
            h, w = s.height, s.width
            b = s.bounds
        clip = box(b.left, b.bottom, b.right, b.top)
        members = [geoms[int(j)].intersection(clip) for j in tree.query(clip) if geoms[int(j)].intersects(clip)]
        members = [g for g in members if not g.is_empty]
        gt, n_gt = gt_instances(members, transform, h, w, args.min_area_px)

        area_prob, ch1 = P.predict_area_boundary(model, rgb, device, 512, tta=args.tta)
        if args.energy:
            pred, n_pred = P.watershed_from_energy(area_prob, ch1, args.area_thr, args.min_seed_dist,
                                                   args.energy_seed_thr, args.min_area_px)
        else:
            pred, n_pred = P.watershed_instances(area_prob, ch1, args.area_thr, args.bnd_thr,
                                                 args.min_seed_dist, args.min_area_px)
        tp, fp, fn, ious = match_instances(pred, n_pred, gt, n_gt, args.iou_thr)
        agg["all"][0] += tp; agg["all"][1] += fp; agg["all"][2] += fn; agg["all"][3] += ious
        clon, clat = (b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0
        if is_holdout(clon, clat):
            n_holdout += 1
            agg["holdout"][0] += tp; agg["holdout"][1] += fp; agg["holdout"][2] += fn; agg["holdout"][3] += ious

    def report(tag, tp, fp, fn, ious):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        miou = float(np.mean(ious)) if ious else 0.0
        print(f"  {tag:18s} TP={tp:4d} FP={fp:4d} FN={fn:4d} | precision={prec:.3f} recall={rec:.3f} "
              f"F1@{args.iou_thr:.2f}={f1:.3f} | IoU medio dos casados={miou:.3f}")
        return {"tag": tag, "tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4),
                "recall": round(rec, 4), "f1": round(f1, 4), "mean_matched_iou": round(miou, 4)}

    print(f"\nMetricas de OBJETO (IoU>={args.iou_thr}) | {len(images)} tiles val ({n_holdout} em holdout espacial >= {args.min_dist_to_train_m:.0f} m do treino):")
    rows = [report("todos val", *agg["all"]), report("holdout espacial", *agg["holdout"])]
    out = args.checkpoint.parent / "object_metrics.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"\nSalvo: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

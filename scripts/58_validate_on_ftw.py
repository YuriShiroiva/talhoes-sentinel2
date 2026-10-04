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

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_boundary.data.dataset import normalize_image_percentile


def _load_predictor_module():
    """Importa funcoes do script 44 (load_model, predict_ensemble) — nome comeca com digito."""
    path = PROJECT_ROOT / "scripts" / "44_predict_sentinel2_fields_end_to_end.py"
    spec = importlib.util.spec_from_file_location("s2pred", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_stack(win_a: np.ndarray, win_b: np.ndarray) -> np.ndarray:
    """FTW 2 janelas (4 bandas B,G,R,NIR) -> 30 canais com a MESMA preprocessing do treino:
    ordem band_time (cada banda, suas 6 datas) + percentil 2-98 por canal.

    Limitacao honesta: o modelo foi treinado com 6 datas reais; aqui replicamos as 2
    janelas do FTW ([A,A,A,B,B,B]), entao a informacao temporal e parcial.
    """
    def five_raw(win):
        b2, b3, b4, b8 = [win[i].astype(np.float32) for i in range(4)]  # B,G,R,NIR cruas
        ndvi = ((b8 - b4) / (b8 + b4 + 1e-6)).astype(np.float32)
        return [b2, b3, b4, b8, ndvi]
    fa, fb = five_raw(win_a), five_raw(win_b)
    per_time = [fa, fa, fa, fb, fb, fb]  # 6 datas
    channels = []
    for b in range(5):          # band_time: para cada banda...
        for t in range(6):      # ...as 6 datas
            channels.append(per_time[t][b])
    x = np.nan_to_num(np.stack(channels, axis=0).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_image_percentile(x)


def metrics(pred: np.ndarray, gt_field: np.ndarray, valid: np.ndarray, eps: float = 1e-7) -> dict:
    p = (pred > 0) & valid
    y = (gt_field > 0) & valid
    tp = float((p & y).sum())
    fp = float((p & ~y).sum())
    fn = float((~p & y).sum())
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    f1 = (2 * prec * rec + eps) / (prec + rec + eps)
    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec, "f1": f1, "target_ratio": float(y.sum()) / max(1.0, float(valid.sum()))}


def read_tif_from_zip(z: zipfile.ZipFile, name: str) -> np.ndarray:
    with rasterio.open(io.BytesIO(z.read(name))) as s:
        return s.read()


def main() -> None:
    ap = argparse.ArgumentParser(description="Valida o ensemble Sentinel-2 (test_058) no benchmark Fields of The World.")
    ap.add_argument("--zip", type=Path, default=PROJECT_ROOT / "data" / "ftw" / "brazil.zip")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sentinel2" / "ftw_brazil_eval")
    ap.add_argument("--threshold", type=float, default=0.49)
    ap.add_argument("--min-area", type=int, default=90)
    ap.add_argument("--tta-mode", choices=["none", "flips"], default="none")
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    P = _load_predictor_module()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print("Carregando ensemble (6 checkpoints)...")
    models = [P.load_model(ck, device)[0] for ck in P.DEFAULT_CHECKPOINTS]
    weights = P.parse_weights(P.DEFAULT_WEIGHTS, len(models))

    z = zipfile.ZipFile(args.zip)
    import os
    tiles = sorted(os.path.basename(n) for n in z.namelist()
                   if n.startswith("s2_images/window_a/") and n.endswith(".tif"))
    if args.max_tiles:
        tiles = tiles[: args.max_tiles]
    print(f"Tiles FTW Brasil: {len(tiles)} | threshold {args.threshold} | TTA {args.tta_mode}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, name in enumerate(tiles):
        try:
            wa = read_tif_from_zip(z, f"s2_images/window_a/{name}")
            wb = read_tif_from_zip(z, f"s2_images/window_b/{name}")
            mask = read_tif_from_zip(z, f"label_masks/semantic_2class/{name}")[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[skip {name}] {exc}")
            continue
        x = build_stack(wa, wb)
        prob, pred = P.predict_ensemble(
            x_np=x, models=models, weights=weights, device=device,
            threshold=args.threshold, min_area=args.min_area, connectivity=8, tta_mode=args.tta_mode,
        )
        gt_field = (mask == 1)
        valid = mask != 0  # exclui pixels de borda/nodata (valor 0)
        m = metrics(pred, gt_field, valid)
        m["tile"] = name
        rows.append(m)
        if (i + 1) % 100 == 0:
            md = np.mean([r["dice"] for r in rows])
            print(f"  {i+1}/{len(tiles)} | Dice medio parcial={md:.3f}")

    keys = ["tile", "dice", "iou", "precision", "recall", "f1", "target_ratio"]
    with (args.output_dir / "per_tile_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    def agg(field):
        v = np.array([r[field] for r in rows], dtype=np.float64)
        return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std())}
    summary = {"dataset": "Fields of The World - brazil", "n_tiles": len(rows),
               "threshold": args.threshold, "tta_mode": args.tta_mode,
               "note": "modelo treinado em 6 datas reais; FTW so tem 2 janelas (replicadas) -> piso, nao comparacao justa",
               **{k: agg(k) for k in ["dice", "iou", "precision", "recall", "f1"]}}
    (args.output_dir / "summary_ftw_brazil.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Resultado FTW Brasil ===")
    for k in ["dice", "iou", "precision", "recall", "f1"]:
        print(f"  {k:10s} media={summary[k]['mean']:.3f}  mediana={summary[k]['median']:.3f}")
    print(f"Salvo: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

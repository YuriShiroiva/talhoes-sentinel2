from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy import ndimage
from shapely.geometry import box, shape

import segmentation_models_pytorch as smp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from field_boundary.geospatial.polygonize import polygonize_mask, write_geojson  # noqa: E402

METERS_PER_DEG_LAT = 111_320.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def _load_builder():
    """Importa fetch_esri_rgb do script 51 (nome de modulo comeca com digito)."""
    path = PROJECT_ROOT / "scripts" / "51_build_rgb_dataset_from_fields.py"
    spec = importlib.util.spec_from_file_location("rgb_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    arch = str(cfg.get("architecture", "unet")).lower()
    encoder = cfg.get("encoder_name", "resnet34")
    in_ch = int(cfg.get("in_channels", 3))
    kwargs = dict(encoder_name=encoder, encoder_weights=None, in_channels=in_ch, classes=1)
    if arch == "unet":
        return smp.Unet(**kwargs)
    if arch in {"unetplusplus", "unet++"}:
        return smp.UnetPlusPlus(**kwargs)
    if arch in {"deeplabv3plus", "deeplab"}:
        return smp.DeepLabV3Plus(**kwargs)
    if arch == "fpn":
        return smp.FPN(**kwargs)
    if arch == "pan":
        return smp.PAN(**kwargs)
    raise ValueError(f"Arquitetura nao suportada: {arch}")


def load_model(checkpoint: Path, device: torch.device):
    import pathlib
    if sys.platform.startswith("win"):
        pathlib.PosixPath = pathlib.WindowsPath
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model = build_model(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def normalize_imagenet(rgb_uint8: np.ndarray) -> np.ndarray:
    x = np.transpose(rgb_uint8.astype(np.float32) / 255.0, (2, 0, 1))
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return mask.astype(np.uint8)
    labels, n = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), np.uint8))
    if n == 0:
        return mask.astype(np.uint8)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area_px
    keep[0] = False
    return keep[labels].astype(np.uint8)


@torch.no_grad()
def predict_mask(model, rgb: np.ndarray, device, infer_size: int, threshold: float, min_area_px: int):
    h, w = rgb.shape[:2]
    x = torch.from_numpy(normalize_imagenet(rgb)).unsqueeze(0).to(device)
    x = F.interpolate(x, size=(infer_size, infer_size), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(model(x))
    prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    pred = remove_small_components((prob >= threshold).astype(np.uint8), min_area_px)
    return prob, pred


def area_ha_for_polygon(geom, lat0: float) -> float:
    m_per_lon = METERS_PER_DEG_LAT * math.cos(math.radians(lat0))
    return float(geom.area) * m_per_lon * METERS_PER_DEG_LAT / 10000.0


def overlay(rgb: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    out = rgb.copy()
    m = mask.astype(bool)
    out[m] = (0.45 * out[m] + 0.55 * np.array(color, dtype=np.float32)).astype(np.uint8)
    return out


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p = pred.astype(bool); g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    denom = p.sum() + g.sum()
    dice = (2 * inter / denom) if denom else 1.0
    iou = (inter / union) if union else 1.0
    return float(dice), float(iou)


def save_panel(rgb, prob, pred, gt, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 4 if gt is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(cols * 4.0, 4.2))
    axes = np.asarray(axes).reshape(-1)
    axes[0].imshow(rgb); axes[0].set_title("RGB (Esri) - entrada"); axes[0].axis("off")
    axes[1].imshow(prob, cmap="viridis", vmin=0, vmax=1); axes[1].set_title("Probabilidade"); axes[1].axis("off")
    axes[2].imshow(overlay(rgb, pred, (0, 220, 0))); axes[2].set_title("Talhoes preditos"); axes[2].axis("off")
    if gt is not None:
        axes[3].imshow(overlay(rgb, gt, (255, 90, 0))); axes[3].set_title("Gabarito (real)"); axes[3].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def bounds_from_area_geojson(path: Path) -> tuple[float, float, float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    geom = shape(data["geometry"] if data.get("type") == "Feature" else data)
    return geom.bounds


def gt_mask_from_answers(path: Path, transform, h: int, w: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", []) if data.get("type") == "FeatureCollection" else [data]
    shapes_ = [(shape(f["geometry"]), 1) for f in feats if f.get("geometry")]
    if not shapes_:
        return None
    return rasterize(shapes_, out_shape=(h, w), fill=0, transform=transform, all_touched=False, dtype="uint8")


def collect_jobs(args) -> list[dict[str, Any]]:
    jobs = []
    if args.bbox:
        jobs.append({"name": args.name or "area", "bounds": tuple(args.bbox), "answers": None})
    elif args.case_dir:
        d = args.case_dir
        jobs.append({"name": d.name, "bounds": bounds_from_area_geojson(d / "area.geojson"),
                     "answers": (d / "answers.geojson") if (d / "answers.geojson").exists() else None})
    elif args.cases_root:
        for d in sorted(p for p in args.cases_root.iterdir() if p.is_dir()):
            area = d / "area.geojson"
            if area.exists():
                jobs.append({"name": d.name, "bounds": bounds_from_area_geojson(area),
                             "answers": (d / "answers.geojson") if (d / "answers.geojson").exists() else None})
        if args.max_cases:
            jobs = jobs[: args.max_cases]
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description="Prediz talhoes em RGB (Esri) com o modelo treinado e gera paineis da tese.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "rgb_fields" / "predictions")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    src.add_argument("--case-dir", type=Path, help="Pasta com area.geojson (+ answers.geojson opcional).")
    src.add_argument("--cases-root", type=Path, help="Pasta de casos (ex.: fields/testes_simulados).")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--max-dim", type=int, default=640)
    ap.add_argument("--infer-size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area-px", type=int, default=60)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    builder = _load_builder()
    model, cfg = load_model(args.checkpoint, device)
    print(f"Modelo: {cfg.get('architecture','unet')}/{cfg.get('encoder_name','resnet34')} "
          f"in_channels={cfg.get('in_channels',3)} | dispositivo={device}")

    jobs = collect_jobs(args)
    if not jobs:
        raise RuntimeError("Nenhuma area de entrada (use --bbox, --case-dir ou --cases-root).")
    print(f"Areas a processar: {len(jobs)}")

    panels_dir = args.output_dir / "panels"
    vectors_dir = args.output_dir / "vectors"
    panels_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    index_rows: list[dict[str, Any]] = []

    for job in jobs:
        name = job["name"]
        try:
            rgb, actual_bounds, zoom = builder.fetch_esri_rgb(job["bounds"], args.max_dim, session, 30.0, 3)
            h, w = rgb.shape[:2]
            transform = from_bounds(*actual_bounds, w, h)
            lat0 = (actual_bounds[1] + actual_bounds[3]) / 2.0

            prob, pred = predict_mask(model, rgb, device, args.infer_size, args.threshold, args.min_area_px)

            features, _ = polygonize_mask(pred, transform=transform, crs=None, min_area_px=0)
            for f in features:
                g = shape(f["geometry"])
                f["properties"]["area_ha"] = round(area_ha_for_polygon(g, lat0), 4)
            write_geojson(features, vectors_dir / f"{name}_pred_fields.geojson", crs="EPSG:4326")

            gt = None
            dice = iou = None
            if job["answers"]:
                gt = gt_mask_from_answers(job["answers"], transform, h, w)
                if gt is not None:
                    dice, iou = dice_iou(pred, gt)

            title = f"{name} | z{zoom} {w}x{h} | preditos={len(features)}"
            if dice is not None:
                title += f" | Dice={dice:.3f} IoU={iou:.3f}"
            save_panel(rgb, prob, pred, gt, panels_dir / f"{name}_panel.png", title)

            index_rows.append({
                "name": name, "zoom": zoom, "width": w, "height": h,
                "pred_fields": len(features), "pred_coverage": round(float(pred.mean()), 4),
                "dice": None if dice is None else round(dice, 4),
                "iou": None if iou is None else round(iou, 4),
                "panel": str(panels_dir / f"{name}_panel.png"),
                "geojson": str(vectors_dir / f"{name}_pred_fields.geojson"),
            })
            msg = f"[ok] {name}: {len(features)} talhoes preditos"
            if dice is not None:
                msg += f" | Dice={dice:.3f} IoU={iou:.3f}"
            print(msg)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERRO] {name}: {exc}")
            index_rows.append({"name": name, "error": str(exc)})

    index_path = args.output_dir / "predictions_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for r in index_rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    valid = [r for r in index_rows if "error" not in r and r.get("dice") is not None]
    print(f"\nFinalizado. Paineis: {panels_dir}")
    print(f"Vetores: {vectors_dir}")
    print(f"Indice: {index_path}")
    if valid:
        mean_dice = sum(r["dice"] for r in valid) / len(valid)
        mean_iou = sum(r["iou"] for r in valid) / len(valid)
        print(f"Media (com gabarito): Dice={mean_dice:.3f} | IoU={mean_iou:.3f} sobre {len(valid)} areas")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

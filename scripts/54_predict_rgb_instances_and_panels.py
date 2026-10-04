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
    path = PROJECT_ROOT / "scripts" / "51_build_rgb_dataset_from_fields.py"
    spec = importlib.util.spec_from_file_location("rgb_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(checkpoint: Path, device: torch.device):
    import pathlib
    if sys.platform.startswith("win"):
        pathlib.PosixPath = pathlib.WindowsPath
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    arch_map = {"unet": "Unet", "unetplusplus": "UnetPlusPlus", "deeplabv3plus": "DeepLabV3Plus", "fpn": "FPN", "pan": "PAN"}
    model = smp.create_model(
        arch=arch_map[str(cfg.get("architecture", "unet")).lower()],
        encoder_name=cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=int(cfg.get("in_channels", 3)),
        classes=int(cfg.get("classes", 2)),
        activation=None,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def normalize_imagenet(rgb_uint8: np.ndarray) -> np.ndarray:
    x = np.transpose(rgb_uint8.astype(np.float32) / 255.0, (2, 0, 1))
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def _flip(t, mode):
    if mode == "id":
        return t
    if mode == "h":
        return torch.flip(t, dims=[3])
    if mode == "v":
        return torch.flip(t, dims=[2])
    return torch.flip(t, dims=[2, 3])  # hv


@torch.no_grad()
def predict_area_boundary(model, rgb, device, infer_size, tta=False):
    h, w = rgb.shape[:2]
    x = torch.from_numpy(normalize_imagenet(rgb)).unsqueeze(0).to(device)
    x = F.interpolate(x, size=(infer_size, infer_size), mode="bilinear", align_corners=False)
    modes = ["id", "h", "v", "hv"] if tta else ["id"]
    acc = None
    for m in modes:
        p = torch.sigmoid(model(_flip(x, m)))
        p = _flip(p, m)  # desfaz o flip no espaco da predicao
        acc = p if acc is None else acc + p
    prob = (acc / len(modes))
    prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0].cpu().numpy()
    return prob[0], prob[1]  # area_prob, boundary_prob


@torch.no_grad()
def _infer_window(model, x, tta):
    ch, cw = x.shape[-2:]
    ph, pw = (32 - ch % 32) % 32, (32 - cw % 32) % 32
    xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
    modes = ["id", "h", "v", "hv"] if tta else ["id"]
    acc = None
    for m in modes:
        p = torch.sigmoid(model(_flip(xp, m)))
        p = _flip(p, m)
        acc = p if acc is None else acc + p
    return (acc / len(modes))[:, :, :ch, :cw]


@torch.no_grad()
def predict_sliding(model, rgb, device, tta, win=512, overlap=64):
    """Inferencia em resolucao NATIVA por janela deslizante (preserva o GSD do treino; sem resize)."""
    h, w = rgb.shape[:2]
    x_full = torch.from_numpy(normalize_imagenet(rgb)).unsqueeze(0).to(device)
    if h <= win and w <= win:
        prob = _infer_window(model, x_full, tta)[0].cpu().numpy()
        return prob[0], prob[1]
    step = max(1, win - overlap)
    ys = sorted(set(list(range(0, max(1, h - win + 1), step)) + [max(0, h - win)]))
    xs = sorted(set(list(range(0, max(1, w - win + 1), step)) + [max(0, w - win)]))
    area = np.zeros((h, w), np.float32)
    bnd = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for y in ys:
        for x0 in xs:
            yh, xw = min(y + win, h), min(x0 + win, w)
            p = _infer_window(model, x_full[:, :, y:yh, x0:xw], tta)[0].cpu().numpy()
            area[y:yh, x0:xw] += p[0]
            bnd[y:yh, x0:xw] += p[1]
            cnt[y:yh, x0:xw] += 1.0
    cnt = np.maximum(cnt, 1.0)
    return area / cnt, bnd / cnt


def watershed_instances(area_prob, boundary_prob, area_thr, bnd_thr, min_seed_dist, min_area_px):
    """Separa talhoes colados via watershed por transformada de distancia.

    1. interior = area & (sem borda); 2. uma semente por pico de distancia (centro de talhao,
    separados por >= min_seed_dist px); 3. cresce as sementes ate preencher a area.
    Isso evita a super-segmentacao de rotular componentes de uma borda ruidosa.
    """
    fg = area_prob >= area_thr
    if not fg.any():
        return np.zeros_like(fg, dtype=np.int32), 0
    interior = fg & (boundary_prob < bnd_thr)
    dist = ndimage.distance_transform_edt(interior)
    win = 2 * int(min_seed_dist) + 1
    peaks = (dist == ndimage.maximum_filter(dist, size=win)) & (dist >= min_seed_dist)
    markers, n = ndimage.label(peaks, structure=np.ones((3, 3), np.uint8))
    if n == 0:
        markers, n = ndimage.label(fg, structure=np.ones((3, 3), np.uint8))
    # cresce sementes ate preencher a area (atribui cada pixel ao marcador mais proximo)
    idx = ndimage.distance_transform_edt(markers == 0, return_distances=False, return_indices=True)
    grown = markers[tuple(idx)]
    instances = np.where(fg, grown, 0).astype(np.int32)
    # remove instancias pequenas e reindexa
    out = np.zeros_like(instances)
    new_id = 0
    for lbl in np.unique(instances):
        if lbl == 0:
            continue
        comp = instances == lbl
        if comp.sum() >= min_area_px:
            new_id += 1
            out[comp] = new_id
    return out, new_id


def watershed_from_energy(area_prob, energy, area_thr, min_seed_dist, energy_seed_thr, min_area_px):
    """Separa instancias usando o mapa de ENERGIA predito: picos de energia = centros = sementes."""
    fg = area_prob >= area_thr
    if not fg.any():
        return np.zeros_like(fg, dtype=np.int32), 0
    win = 2 * int(min_seed_dist) + 1
    peaks = (energy == ndimage.maximum_filter(energy, size=win)) & (energy >= energy_seed_thr) & fg
    markers, n = ndimage.label(peaks, structure=np.ones((3, 3), np.uint8))
    if n == 0:
        markers, n = ndimage.label(fg, structure=np.ones((3, 3), np.uint8))
    idx = ndimage.distance_transform_edt(markers == 0, return_distances=False, return_indices=True)
    grown = markers[tuple(idx)]
    instances = np.where(fg, grown, 0).astype(np.int32)
    out = np.zeros_like(instances)
    nid = 0
    for lbl in np.unique(instances):
        if lbl == 0:
            continue
        comp = instances == lbl
        if comp.sum() >= min_area_px:
            nid += 1
            out[comp] = nid
    return out, nid


def separate(area_prob, ch1, args):
    """Escolhe o metodo de instancia: energia (picos) ou borda (dist. transform)."""
    if getattr(args, "energy", False):
        return watershed_from_energy(area_prob, ch1, args.area_thr, args.min_seed_dist,
                                     args.energy_seed_thr, args.min_area_px)
    return watershed_instances(area_prob, ch1, args.area_thr, args.bnd_thr, args.min_seed_dist, args.min_area_px)


def area_ha(geom, lat0):
    return float(geom.area) * (METERS_PER_DEG_LAT * math.cos(math.radians(lat0))) * METERS_PER_DEG_LAT / 10000.0


def color_instances(rgb, instances):
    out = rgb.copy().astype(np.float32)
    rng = np.random.default_rng(7)
    for lbl in np.unique(instances):
        if lbl == 0:
            continue
        color = rng.integers(40, 255, size=3).astype(np.float32)
        m = instances == lbl
        out[m] = 0.45 * out[m] + 0.55 * color
    return out.clip(0, 255).astype(np.uint8)


def overlay(rgb, mask, color):
    out = rgb.copy(); m = mask.astype(bool)
    out[m] = (0.45 * out[m] + 0.55 * np.array(color, np.float32)).astype(np.uint8)
    return out


def dice_iou(pred_mask, gt):
    p = pred_mask.astype(bool); g = gt.astype(bool)
    inter = np.logical_and(p, g).sum(); union = np.logical_or(p, g).sum(); denom = p.sum() + g.sum()
    return (float(2 * inter / denom) if denom else 1.0), (float(inter / union) if union else 1.0)


def save_panel(rgb, area_prob, ch1, instances, gt, out_path, title):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 5 if gt is not None else 4
    fig, ax = plt.subplots(1, cols, figsize=(cols * 3.7, 4.0))
    ax = np.asarray(ax).reshape(-1)
    ax[0].imshow(rgb); ax[0].set_title("RGB (Esri)"); ax[0].axis("off")
    ax[1].imshow(area_prob, cmap="viridis", vmin=0, vmax=1); ax[1].set_title("Prob. area"); ax[1].axis("off")
    ax[2].imshow(ch1, cmap="magma", vmin=0, vmax=1); ax[2].set_title("Borda / energia"); ax[2].axis("off")
    ax[3].imshow(color_instances(rgb, instances)); ax[3].set_title("Talhoes (instancias)"); ax[3].axis("off")
    if gt is not None:
        ax[4].imshow(overlay(rgb, gt, (255, 90, 0))); ax[4].set_title("Gabarito (real)"); ax[4].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def bounds_from_area_geojson(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return shape(data["geometry"] if data.get("type") == "Feature" else data).bounds


def gt_from_answers(path: Path, transform, h, w):
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", []) if data.get("type") == "FeatureCollection" else [data]
    shp = [(shape(f["geometry"]), 1) for f in feats if f.get("geometry")]
    if not shp:
        return None, 0
    mask = rasterize(shp, out_shape=(h, w), fill=0, transform=transform, all_touched=False, dtype="uint8")
    return mask, len(shp)


def collect_jobs(args):
    jobs = []
    if args.bbox:
        jobs.append({"name": args.name or "area", "bounds": tuple(args.bbox), "answers": None})
    elif args.case_dir:
        d = args.case_dir
        jobs.append({"name": d.name, "bounds": bounds_from_area_geojson(d / "area.geojson"),
                     "answers": (d / "answers.geojson") if (d / "answers.geojson").exists() else None})
    elif args.cases_root:
        for d in sorted(p for p in args.cases_root.iterdir() if p.is_dir()):
            if (d / "area.geojson").exists():
                jobs.append({"name": d.name, "bounds": bounds_from_area_geojson(d / "area.geojson"),
                             "answers": (d / "answers.geojson") if (d / "answers.geojson").exists() else None})
        if args.max_cases:
            jobs = jobs[: args.max_cases]
    return jobs


def gt_instance_count(area_gt: np.ndarray, boundary_gt: np.ndarray, min_area_px: int) -> int:
    """Conta talhoes do gabarito separando pela borda real (banda 2)."""
    interior = area_gt.astype(bool) & ~boundary_gt.astype(bool)
    lbl, n = ndimage.label(interior, structure=np.ones((3, 3), np.uint8))
    if n == 0:
        return 0
    sizes = np.bincount(lbl.ravel())
    return int(sum(1 for s in sizes[1:] if s >= min_area_px))


def run_local_tiles(args, model, device) -> None:
    """Avaliacao justa em tiles locais (in-distribution) com gabarito COMPLETO (mascara 2 bandas)."""
    import rasterio
    img_dir = args.images_root / "images" / args.split
    mask_dir = args.images_root / "masks" / args.split
    images = sorted(img_dir.glob("*.tif"))
    if args.max_tiles:
        images = images[: args.max_tiles]
    panels_dir = args.output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Tiles locais ({args.split}): {len(images)}")
    for img_path in images:
        name = img_path.stem
        with rasterio.open(img_path) as s:
            rgb = np.transpose(s.read([1, 2, 3]), (1, 2, 0)).astype(np.uint8)
        mask_path = mask_dir / f"{name.replace('_ortho_', '_ortholabel_')}.tif"
        with rasterio.open(mask_path) as s:
            gt_area = (s.read(1) > 0).astype(np.uint8)
            gt_bnd = (s.read(2) > 0).astype(np.uint8) if s.count >= 2 else np.zeros_like(gt_area)
        area_prob, ch1 = predict_area_boundary(model, rgb, device, args.infer_size, tta=args.tta)
        instances, n_inst = separate(area_prob, ch1, args)
        dice, iou = dice_iou(instances > 0, gt_area)
        n_gt = gt_instance_count(gt_area, gt_bnd, args.min_area_px)
        title = f"{name} | instancias={n_inst} vs gabarito~{n_gt} | area Dice={dice:.3f} IoU={iou:.3f}"
        save_panel(rgb, area_prob, ch1, instances, gt_area, panels_dir / f"{name}_panel.png", title)
        rows.append({"name": name, "pred_instances": n_inst, "gt_fields": n_gt,
                     "count_error": n_inst - n_gt, "area_dice": round(dice, 4), "area_iou": round(iou, 4)})
        print(f"[ok] {name}: {n_inst} instancias (gabarito ~{n_gt}) | Dice={dice:.3f}")
    index_path = args.output_dir / "instances_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
    md = sum(r["area_dice"] for r in rows) / len(rows)
    mi = sum(r["area_iou"] for r in rows) / len(rows)
    ce = sum(abs(r["count_error"]) for r in rows) / len(rows)
    print(f"\nPaineis: {panels_dir}\nIndice: {index_path}")
    print(f"Media (in-distribution, gabarito completo): area Dice={md:.3f} IoU={mi:.3f} | "
          f"erro medio de contagem={ce:.1f} (predito {sum(r['pred_instances'] for r in rows)} vs gabarito {sum(r['gt_fields'] for r in rows)})")


def main():
    ap = argparse.ArgumentParser(description="Prediz talhoes como INSTANCIAS (area+borda+watershed) e gera paineis.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "rgb_fields" / "instances_v1")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    src.add_argument("--case-dir", type=Path)
    src.add_argument("--cases-root", type=Path)
    src.add_argument("--images-root", type=Path, help="Avalia tiles locais (in-distribution): <root>/images/<split>.")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--max-dim", type=int, default=640)
    ap.add_argument("--infer-size", type=int, default=512)
    ap.add_argument("--fixed-zoom", type=int, default=None,
                    help="Busca o print neste zoom Esri (igual ao treino) e infere por janela deslizante na res. nativa.")
    ap.add_argument("--max-px", type=int, default=1408, help="Lado maximo do print em px no modo --fixed-zoom.")
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--area-thr", type=float, default=0.5)
    ap.add_argument("--bnd-thr", type=float, default=0.5)
    ap.add_argument("--min-seed-dist", type=int, default=20,
                    help="Distancia minima (px) entre centros de talhao; maior = menos instancias.")
    ap.add_argument("--min-area-px", type=int, default=60)
    ap.add_argument("--energy", action="store_true", help="Modelo area+energia: separa por picos de energia.")
    ap.add_argument("--energy-seed-thr", type=float, default=0.35, help="Energia minima para ser semente.")
    ap.add_argument("--tta", action="store_true", help="Test-time augmentation por flips (h/v/hv).")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    builder = _load_builder()
    model, cfg = load_model(args.checkpoint, device)
    print(f"Modelo: {cfg.get('architecture')}/{cfg.get('encoder_name')} classes={cfg.get('classes')} | {device}")

    if args.images_root:
        run_local_tiles(args, model, device)
        return

    jobs = collect_jobs(args)
    if not jobs:
        raise RuntimeError("Sem area de entrada.")
    print(f"Areas: {len(jobs)}")

    panels_dir = args.output_dir / "panels"; vectors_dir = args.output_dir / "vectors"
    panels_dir.mkdir(parents=True, exist_ok=True); vectors_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    rows = []

    for job in jobs:
        name = job["name"]
        try:
            if args.fixed_zoom:
                b = job["bounds"]
                clon, clat = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                wpx0, _ = builder._lonlat_to_pixel(b[0], clat, args.fixed_zoom)
                wpx1, _ = builder._lonlat_to_pixel(b[2], clat, args.fixed_zoom)
                _, hpy0 = builder._lonlat_to_pixel(clon, b[3], args.fixed_zoom)
                _, hpy1 = builder._lonlat_to_pixel(clon, b[1], args.fixed_zoom)
                span = int(max(abs(wpx1 - wpx0), abs(hpy1 - hpy0)))
                tile_px = max(args.win, min(span, args.max_px))
                tile_px -= tile_px % 32
                rgb, bnds, zoom = builder.fetch_esri_fixed(clon, clat, args.fixed_zoom, tile_px, session, 30.0, 3)
                h, w = rgb.shape[:2]
                area_prob, ch1 = predict_sliding(model, rgb, device, args.tta, args.win, args.overlap)
            else:
                rgb, bnds, zoom = builder.fetch_esri_rgb(job["bounds"], args.max_dim, session, 30.0, 3)
                h, w = rgb.shape[:2]
                area_prob, ch1 = predict_area_boundary(model, rgb, device, args.infer_size, tta=args.tta)
            transform = from_bounds(*bnds, w, h)
            lat0 = (bnds[1] + bnds[3]) / 2.0
            instances, n_inst = separate(area_prob, ch1, args)

            features = []
            for lbl in range(1, n_inst + 1):
                feats, _ = polygonize_mask((instances == lbl).astype(np.uint8), transform=transform, crs=None, min_area_px=0)
                for f in feats:
                    g = shape(f["geometry"])
                    f["properties"]["instance_id"] = lbl
                    f["properties"]["area_ha"] = round(area_ha(g, lat0), 4)
                    features.append(f)
            write_geojson(features, vectors_dir / f"{name}_pred_instances.geojson", crs="EPSG:4326")

            gt = None; n_gt = 0; dice = iou = None
            if job["answers"]:
                gt, n_gt = gt_from_answers(job["answers"], transform, h, w)
                if gt is not None:
                    dice, iou = dice_iou(instances > 0, gt)

            title = f"{name} | z{zoom} | instancias={n_inst}"
            if n_gt:
                title += f" vs gabarito={n_gt}"
            if dice is not None:
                title += f" | area Dice={dice:.3f} IoU={iou:.3f}"
            save_panel(rgb, area_prob, ch1, instances, gt, panels_dir / f"{name}_panel.png", title)

            rows.append({"name": name, "zoom": zoom, "pred_instances": n_inst, "gt_fields": n_gt,
                         "count_error": (n_inst - n_gt) if n_gt else None,
                         "area_dice": None if dice is None else round(dice, 4),
                         "area_iou": None if iou is None else round(iou, 4)})
            print(f"[ok] {name}: {n_inst} instancias" + (f" (gabarito {n_gt}, Dice={dice:.3f})" if dice is not None else ""))
        except Exception as exc:  # noqa: BLE001
            print(f"[ERRO] {name}: {exc}")
            rows.append({"name": name, "error": str(exc)})

    index_path = args.output_dir / "instances_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fn = sorted({k for r in rows for k in r})
        w_ = csv.DictWriter(f, fieldnames=fn); w_.writeheader(); w_.writerows(rows)

    valid = [r for r in rows if r.get("area_dice") is not None]
    print(f"\nPaineis: {panels_dir}\nVetores: {vectors_dir}\nIndice: {index_path}")
    if valid:
        md = sum(r["area_dice"] for r in valid) / len(valid)
        mi = sum(r["area_iou"] for r in valid) / len(valid)
        ce = [abs(r["count_error"]) for r in valid if r["count_error"] is not None]
        print(f"Media: area Dice={md:.3f} IoU={mi:.3f} | erro medio de contagem={sum(ce)/len(ce):.1f} talhoes "
              f"(predito {sum(r['pred_instances'] for r in valid)} vs gabarito {sum(r['gt_fields'] for r in valid)})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

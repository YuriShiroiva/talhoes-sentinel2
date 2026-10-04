from __future__ import annotations

import argparse
import csv
import io
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely import wkt
from shapely.geometry import Point, box
from shapely.strtree import STRtree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELDS_CSV = PROJECT_ROOT / "fields" / "reais" / "fields_yuri.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "prepared" / "rgb_fields_yuri"

METERS_PER_DEG_LAT = 111_320.0

# --- Esri World Imagery (mesma fonte XYZ usada pela plataforma) ---
TILE_SIZE = 256
MAX_ZOOM = 19
ESRI_TILE_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
USER_AGENT = "tcc-sentinel2-fieldboundary/1.0 (academic test)"


def _clip_lat(lat: float) -> float:
    return max(min(lat, 85.05112878), -85.05112878)


def _lonlat_to_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = _clip_lat(lat)
    sin_lat = math.sin(math.radians(lat))
    world = TILE_SIZE * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * world
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * world
    return x, y


def _pixel_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    world = TILE_SIZE * (2 ** zoom)
    lon = x / world * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / world
    lat = math.degrees(math.atan(math.sinh(n)))
    return lon, lat


def zoom_for_bounds(bounds: tuple[float, float, float, float], max_dim: int) -> int:
    min_lon, min_lat, max_lon, max_lat = bounds
    for zoom in range(MAX_ZOOM, 0, -1):
        west, north = _lonlat_to_pixel(min_lon, max_lat, zoom)
        east, south = _lonlat_to_pixel(max_lon, min_lat, zoom)
        if abs(east - west) <= max_dim and abs(south - north) <= max_dim:
            return zoom
    return 1


def fetch_esri_rgb(
    bounds: tuple[float, float, float, float],
    max_dim: int,
    session: requests.Session,
    timeout: float,
    retries: int,
) -> tuple[np.ndarray, tuple[float, float, float, float], int]:
    """Baixa e mosaica tiles Esri para a bbox; retorna (rgb HxWx3 uint8, bounds_reais, zoom)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    zoom = zoom_for_bounds(bounds, max_dim=max_dim)
    west_px, north_px = _lonlat_to_pixel(min_lon, max_lat, zoom)
    east_px, south_px = _lonlat_to_pixel(max_lon, min_lat, zoom)

    n_tiles = 2 ** zoom
    x0 = max(0, min(int(math.floor(min(west_px, east_px) / TILE_SIZE)), n_tiles - 1))
    x1 = max(0, min(int(math.floor((max(west_px, east_px) - 1) / TILE_SIZE)), n_tiles - 1))
    y0 = max(0, min(int(math.floor(min(north_px, south_px) / TILE_SIZE)), n_tiles - 1))
    y1 = max(0, min(int(math.floor((max(north_px, south_px) - 1) / TILE_SIZE)), n_tiles - 1))

    mosaic = Image.new("RGB", ((x1 - x0 + 1) * TILE_SIZE, (y1 - y0 + 1) * TILE_SIZE))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            url = ESRI_TILE_URL.format(z=zoom, y=ty, x=tx)
            tile_img = None
            last_err: Exception | None = None
            for attempt in range(retries):
                try:
                    resp = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
                    resp.raise_for_status()
                    tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(0.5 * (attempt + 1))
            if tile_img is None:
                raise RuntimeError(f"Falha ao baixar tile z{zoom} x{tx} y{ty}: {last_err}")
            mosaic.paste(tile_img, ((tx - x0) * TILE_SIZE, (ty - y0) * TILE_SIZE))

    crop_left = int(round(min(west_px, east_px) - x0 * TILE_SIZE))
    crop_top = int(round(min(north_px, south_px) - y0 * TILE_SIZE))
    crop_right = max(int(round(max(west_px, east_px) - x0 * TILE_SIZE)), crop_left + 1)
    crop_bottom = max(int(round(max(north_px, south_px) - y0 * TILE_SIZE)), crop_top + 1)
    image = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))

    aw, an = _pixel_to_lonlat(x0 * TILE_SIZE + crop_left, y0 * TILE_SIZE + crop_top, zoom)
    ae, as_ = _pixel_to_lonlat(x0 * TILE_SIZE + crop_right, y0 * TILE_SIZE + crop_bottom, zoom)
    actual_bounds = (aw, as_, ae, an)  # (min_lon, min_lat, max_lon, max_lat)
    return np.asarray(image, dtype=np.uint8), actual_bounds, zoom


def fetch_esri_fixed(center_lon, center_lat, zoom, tile_px, session, timeout, retries):
    """Baixa um tile de tamanho fixo (tile_px) em zoom FIXO -> GSD (m/px) constante entre tiles."""
    cx, cy = _lonlat_to_pixel(center_lon, center_lat, zoom)
    half = tile_px / 2.0
    w_px, e_px, n_px, s_px = cx - half, cx + half, cy - half, cy + half
    n_tiles = 2 ** zoom
    x0 = max(0, min(int(math.floor(w_px / TILE_SIZE)), n_tiles - 1))
    x1 = max(0, min(int(math.floor((e_px - 1) / TILE_SIZE)), n_tiles - 1))
    y0 = max(0, min(int(math.floor(n_px / TILE_SIZE)), n_tiles - 1))
    y1 = max(0, min(int(math.floor((s_px - 1) / TILE_SIZE)), n_tiles - 1))
    mosaic = Image.new("RGB", ((x1 - x0 + 1) * TILE_SIZE, (y1 - y0 + 1) * TILE_SIZE))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            url = ESRI_TILE_URL.format(z=zoom, y=ty, x=tx)
            tile_img = None
            last_err = None
            for attempt in range(retries):
                try:
                    resp = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
                    resp.raise_for_status()
                    tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(0.5 * (attempt + 1))
            if tile_img is None:
                raise RuntimeError(f"Falha tile z{zoom} x{tx} y{ty}: {last_err}")
            mosaic.paste(tile_img, ((tx - x0) * TILE_SIZE, (ty - y0) * TILE_SIZE))
    crop_left = int(round(w_px - x0 * TILE_SIZE))
    crop_top = int(round(n_px - y0 * TILE_SIZE))
    crop_right, crop_bottom = crop_left + tile_px, crop_top + tile_px
    image = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))
    aw, an = _pixel_to_lonlat(x0 * TILE_SIZE + crop_left, y0 * TILE_SIZE + crop_top, zoom)
    ae, as_ = _pixel_to_lonlat(x0 * TILE_SIZE + crop_right, y0 * TILE_SIZE + crop_bottom, zoom)
    return np.asarray(image, dtype=np.uint8), (aw, as_, ae, an), zoom


# --- fields + clustering (mesma logica do script 50, deduplicado) ---
def fix_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    fixed = geom.buffer(0)
    return None if (fixed.is_empty or not fixed.is_valid) else fixed


def in_lonlat_range(geom) -> bool:
    minx, miny, maxx, maxy = geom.bounds
    return -180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90


def load_fields(fields_csv: Path, chunksize: int, max_fields: int | None) -> list[Any]:
    geoms: list[Any] = []
    seen: set[bytes] = set()
    reader = pd.read_csv(fields_csv, usecols=["geom"], chunksize=chunksize, dtype="string", keep_default_na=False)
    for chunk in reader:
        for geom_text in chunk["geom"].astype(str):
            if max_fields is not None and len(geoms) >= max_fields:
                return geoms
            if not geom_text:
                continue
            try:
                geom = fix_geometry(wkt.loads(geom_text))
            except Exception:
                geom = None
            if geom is None or not in_lonlat_range(geom):
                continue
            key = geom.wkb
            if key in seen:
                continue
            seen.add(key)
            geoms.append(geom)
    return geoms


def cluster_by_proximity(centroids: np.ndarray, neighbor_dist_deg: float, max_per_case: int) -> list[list[int]]:
    n = len(centroids)
    points = [Point(float(x), float(y)) for x, y in centroids]
    tree = STRtree(points)
    assigned = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []
    for seed in range(n):
        if assigned[seed]:
            continue
        cluster = [seed]
        assigned[seed] = True
        frontier = [seed]
        while frontier and len(cluster) < max_per_case:
            cur = frontier.pop()
            cx, cy = float(centroids[cur][0]), float(centroids[cur][1])
            qbox = box(cx - neighbor_dist_deg, cy - neighbor_dist_deg, cx + neighbor_dist_deg, cy + neighbor_dist_deg)
            for j in tree.query(qbox):
                j = int(j)
                if assigned[j]:
                    continue
                if points[cur].distance(points[j]) <= neighbor_dist_deg:
                    assigned[j] = True
                    cluster.append(j)
                    frontier.append(j)
                    if len(cluster) >= max_per_case:
                        break
        clusters.append(cluster)
    return clusters


def padded_bounds(tight: tuple[float, float, float, float], margin_frac: float, min_margin_m: float):
    minx, miny, maxx, maxy = tight
    lat0 = (miny + maxy) / 2.0
    m_per_lon = max(METERS_PER_DEG_LAT * math.cos(math.radians(lat0)), 1.0)
    width_m = (maxx - minx) * m_per_lon
    height_m = (maxy - miny) * METERS_PER_DEG_LAT
    margin_m = max(margin_frac * max(width_m, height_m), min_margin_m)
    pad_lon = margin_m / m_per_lon
    pad_lat = margin_m / METERS_PER_DEG_LAT
    return (minx - pad_lon, miny - pad_lat, maxx + pad_lon, maxy + pad_lat)


def write_geotiff_rgb(path: Path, rgb: np.ndarray, transform, crs: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=rgb.shape[0], width=rgb.shape[1],
        count=3, dtype="uint8", crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(np.transpose(rgb, (2, 0, 1)))


def write_geotiff_mask(path: Path, mask: np.ndarray, transform, crs: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
        count=1, dtype="uint8", crs=crs, transform=transform, compress="deflate", nodata=0,
    ) as dst:
        dst.write(mask.astype(np.uint8), 1)


def save_preview(path: Path, rgb: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = rgb.copy()
    edge = mask.astype(bool) ^ np.pad(mask, 1, mode="edge")[2:, 1:-1].astype(bool)
    overlay[mask.astype(bool)] = (0.5 * overlay[mask.astype(bool)] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    Image.fromarray(np.concatenate([rgb, overlay], axis=1)).save(path)


def build_dataset(
    fields_csv: Path, output_root: Path, chunksize: int, max_fields: int | None,
    neighbor_dist_m: float, max_per_case: int, min_fields: int, margin_frac: float,
    min_margin_m: float, max_dim: int, val_fraction: float, max_tiles: int | None,
    sleep_s: float, timeout: float, retries: int, preview: bool,
    fixed_zoom: int | None, tile_px: int, neg_frac: float, neg_min_dist_m: float,
) -> None:
    print(f"Lendo e deduplicando fields de {fields_csv} ...")
    geoms = load_fields(fields_csv, chunksize, max_fields)
    print(f"Fields unicos validos: {len(geoms)}")
    if not geoms:
        raise RuntimeError("Nenhum field valido.")

    centroids = np.array([[g.centroid.x, g.centroid.y] for g in geoms], dtype=np.float64)
    tree = STRtree(geoms)  # mascaras COMPLETAS (todos os talhoes do tile) + rejeicao de negativos
    neighbor_dist_deg = neighbor_dist_m / METERS_PER_DEG_LAT
    print(f"Agrupando por proximidade (<= {neighbor_dist_m:.0f} m, max {max_per_case}/tile) ...")
    clusters = cluster_by_proximity(centroids, neighbor_dist_deg, max_per_case)
    clusters = [c for c in clusters if len(c) >= min_fields]
    random.Random(42).shuffle(clusters)
    if max_tiles is not None:
        clusters = clusters[:max_tiles]
    mode = f"zoom FIXO {fixed_zoom} ({tile_px}px)" if fixed_zoom else "zoom adaptativo"
    print(f"Tiles positivos: {len(clusters)} | {mode}")

    for split in ["train", "val"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "masks" / split).mkdir(parents=True, exist_ok=True)
    if preview:
        (output_root / "previews").mkdir(parents=True, exist_ok=True)

    val_every = max(int(round(1.0 / val_fraction)), 2) if val_fraction > 0 else 0
    session = requests.Session()
    rows: list[dict[str, Any]] = []

    def complete_mask(actual_bounds, h, w, transform):
        clip = box(*actual_bounds)
        shapes = []
        for j in tree.query(clip):
            inter = geoms[int(j)].intersection(clip)
            if not inter.is_empty:
                shapes.append((inter, 1))
        return rasterize(shapes, out_shape=(h, w), fill=0, transform=transform, all_touched=False,
                         dtype="uint8") if shapes else np.zeros((h, w), dtype=np.uint8)

    def write_tile(idx, rgb, actual_bounds, zoom, n_fields, kind):
        h, w = rgb.shape[:2]
        transform = from_bounds(actual_bounds[0], actual_bounds[1], actual_bounds[2], actual_bounds[3], w, h)
        mask = complete_mask(actual_bounds, h, w, transform)
        split = "val" if (val_every and idx % val_every == 0) else "train"
        stem = f"tile_{idx:05d}_ortho_"
        write_geotiff_rgb(output_root / "images" / split / f"{stem}.tif", rgb, transform, "EPSG:4326")
        write_geotiff_mask(output_root / "masks" / split / f"{stem.replace('_ortho_', '_ortholabel_')}.tif",
                           mask, transform, "EPSG:4326")
        if preview:
            save_preview(output_root / "previews" / f"{stem}.png", rgb, mask)
        rows.append({"tile": stem, "split": split, "kind": kind, "fields": n_fields, "width": w, "height": h,
                     "zoom": zoom, "positive_ratio": round(float(mask.mean()), 5),
                     "min_lon": actual_bounds[0], "min_lat": actual_bounds[1],
                     "max_lon": actual_bounds[2], "max_lat": actual_bounds[3]})
        return split

    ok = 0
    for idx, cluster in enumerate(clusters):
        sub = [geoms[i] for i in cluster]
        try:
            if fixed_zoom:
                cx = float(np.mean([g.centroid.x for g in sub]))
                cy = float(np.mean([g.centroid.y for g in sub]))
                rgb, ab, zoom = fetch_esri_fixed(cx, cy, fixed_zoom, tile_px, session, timeout, retries)
            else:
                minx, miny, maxx, maxy = sub[0].bounds
                for g in sub[1:]:
                    bx = g.bounds
                    minx, miny, maxx, maxy = min(minx, bx[0]), min(miny, bx[1]), max(maxx, bx[2]), max(maxy, bx[3])
                rgb, ab, zoom = fetch_esri_rgb(padded_bounds((minx, miny, maxx, maxy), margin_frac, min_margin_m),
                                               max_dim, session, timeout, retries)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip pos {idx}] {exc}")
            continue
        write_tile(idx, rgb, ab, zoom, len(sub), "positive")
        ok += 1
        if ok % 20 == 0 or ok == 1:
            print(f"  {ok} positivos ok (ultimo: z{zoom}, pos={rows[-1]['positive_ratio']:.3f})")
        if sleep_s > 0:
            time.sleep(sleep_s)

    # tiles NEGATIVOS (sem talhao) para reduzir falsos positivos em mata/estrada/cidade
    n_neg = int(round(neg_frac * len(clusters) / max(1e-9, 1.0 - neg_frac))) if neg_frac > 0 else 0
    if n_neg > 0:
        neg_zoom = fixed_zoom or 15
        neg_dist_deg = neg_min_dist_m / METERS_PER_DEG_LAT
        rng = random.Random(123)
        base, made, attempts = len(clusters), 0, 0
        print(f"Gerando ~{n_neg} tiles negativos perto das fazendas (>= {neg_min_dist_m:.0f} m de qualquer talhao) ...")
        while made < n_neg and attempts < n_neg * 80:
            attempts += 1
            # ancora num talhao aleatorio e desloca para o entorno (mata/estrada/solo nu, sem talhao)
            clon, clat = centroids[rng.randrange(len(centroids))]
            ang = rng.uniform(0.0, 2.0 * math.pi)
            dist_m = rng.uniform(neg_min_dist_m, neg_min_dist_m * 2.5)
            lon = clon + (dist_m * math.cos(ang)) / (METERS_PER_DEG_LAT * math.cos(math.radians(clat)))
            lat = clat + (dist_m * math.sin(ang)) / METERS_PER_DEG_LAT
            p = Point(lon, lat)
            if any(geoms[int(j)].distance(p) <= neg_dist_deg for j in tree.query(p.buffer(neg_dist_deg))):
                continue
            try:
                rgb, ab, zoom = fetch_esri_fixed(lon, lat, neg_zoom, tile_px, session, timeout, retries)
            except Exception:  # noqa: BLE001
                continue
            write_tile(base + made, rgb, ab, zoom, 0, "negative")
            made += 1
            ok += 1
            if sleep_s > 0:
                time.sleep(sleep_s)
        print(f"Negativos gerados: {made}/{n_neg}")

    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    index_path = output_root / "metadata" / "tiles_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["tile"])
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    n_pos = sum(1 for r in rows if r["kind"] == "positive")
    n_neg_made = sum(1 for r in rows if r["kind"] == "negative")
    print(f"\nDataset pronto: {ok} tiles ({n_pos} positivos / {n_neg_made} negativos | {n_train} train / {n_val} val) em {output_root}")
    print(f"Indice: {index_path}")
    print("Proximo: gerar mascaras 2-bandas (borda) e treinar boundary-aware:")
    print(f"  python scripts/53_make_boundary_masks_for_images.py --root {output_root}")
    print(f"  python scripts/30_train_orthos_boundary_aware_multitask.py --root {output_root} "
          "--output-dir outputs/rgb_fields/boundary_v3 --architecture unet --encoder-name resnet34 "
          "--batch-size 2 --num-workers 4 --epochs 30 --use-tversky-area")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Constroi dataset RGB (Esri World Imagery) + mascara a partir dos talhoes reais.")
    p.add_argument("--fields-csv", type=Path, default=DEFAULT_FIELDS_CSV)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--chunksize", type=int, default=20000)
    p.add_argument("--max-fields", type=int, default=None)
    p.add_argument("--neighbor-dist-m", type=float, default=600.0)
    p.add_argument("--max-per-case", type=int, default=12)
    p.add_argument("--min-fields", type=int, default=1)
    p.add_argument("--margin-frac", type=float, default=0.15, help="Borda menor que no print (fields enchem o tile).")
    p.add_argument("--min-margin-m", type=float, default=80.0)
    p.add_argument("--max-dim", type=int, default=640, help="Lado maximo do tile em pixels (modo zoom adaptativo).")
    p.add_argument("--fixed-zoom", type=int, default=None, help="Se setado, GSD FIXO nesse zoom Esri (recomendado: 16).")
    p.add_argument("--tile-px", type=int, default=512, help="Tamanho do tile em pixels no modo zoom fixo.")
    p.add_argument("--neg-frac", type=float, default=0.0, help="Fracao de tiles NEGATIVOS (sem talhao). Ex.: 0.2.")
    p.add_argument("--neg-min-dist-m", type=float, default=800.0, help="Distancia minima do centro negativo a qualquer talhao.")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--max-tiles", type=int, default=None, help="Limita o numero de tiles (smoke/escala).")
    p.add_argument("--sleep-s", type=float, default=0.1, help="Pausa entre tiles (educado com o servidor).")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--preview", action="store_true", help="Salva PNG RGB|overlay por tile para validar alinhamento.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.fields_csv.exists():
        raise FileNotFoundError(args.fields_csv)
    build_dataset(
        fields_csv=args.fields_csv, output_root=args.output_root, chunksize=args.chunksize,
        max_fields=args.max_fields, neighbor_dist_m=args.neighbor_dist_m, max_per_case=args.max_per_case,
        min_fields=args.min_fields, margin_frac=args.margin_frac, min_margin_m=args.min_margin_m,
        max_dim=args.max_dim, val_fraction=args.val_fraction, max_tiles=args.max_tiles,
        sleep_s=args.sleep_s, timeout=args.timeout, retries=args.retries, preview=args.preview,
        fixed_zoom=args.fixed_zoom, tile_px=args.tile_px, neg_frac=args.neg_frac, neg_min_dist_m=args.neg_min_dist_m,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

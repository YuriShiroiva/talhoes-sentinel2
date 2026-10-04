from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rasterio
import rasterio.warp
import xarray as xr
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely import wkt
from shapely.geometry import box
from shapely.ops import transform as shapely_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELDS_CSV = PROJECT_ROOT / "fields" / "fields_yuri.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sentinel2_from_fields"
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
BAND_ASSETS = {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"}


def fix_geometry(geom):
    if geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    fixed = geom.buffer(0)
    if fixed.is_empty or not fixed.is_valid:
        return None
    return fixed


def lonlat_bbox_from_fields(fields_csv: Path, max_fields: int | None = None) -> tuple[float, float, float, float]:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    kept = 0

    for chunk in pd.read_csv(fields_csv, usecols=["geom"], chunksize=5000, dtype="string", keep_default_na=False):
        for geom_text in chunk["geom"].astype(str):
            if max_fields is not None and kept >= max_fields:
                return minx, miny, maxx, maxy
            try:
                geom = fix_geometry(wkt.loads(geom_text))
            except Exception:
                geom = None
            if geom is None:
                continue

            bx = geom.bounds
            if not (-180 <= bx[0] <= 180 and -180 <= bx[2] <= 180 and -90 <= bx[1] <= 90 and -90 <= bx[3] <= 90):
                continue
            minx = min(minx, bx[0])
            miny = min(miny, bx[1])
            maxx = max(maxx, bx[2])
            maxy = max(maxy, bx[3])
            kept += 1

    if kept == 0:
        raise RuntimeError(f"Nenhuma geometria valida em lon/lat encontrada em {fields_csv}")
    return minx, miny, maxx, maxy


def query_sentinel2_items(
    bbox_lonlat: tuple[float, float, float, float],
    date_range: str,
    max_cloud: float,
    limit: int,
) -> list[dict]:
    date_range = normalize_date_range(date_range)
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox_lonlat),
        "datetime": date_range,
        "limit": limit,
    }
    response = requests.post(EARTH_SEARCH_URL, json=payload, timeout=60)
    if not response.ok:
        raise RuntimeError(f"STAC search falhou: HTTP {response.status_code}: {response.text[:1000]}")
    features = response.json().get("features", [])
    filtered = [
        item
        for item in features
        if float(item.get("properties", {}).get("eo:cloud_cover", 1000.0)) <= max_cloud
    ]
    return sorted(filtered, key=lambda item: float(item.get("properties", {}).get("eo:cloud_cover", 1000.0)))


def normalize_date_range(value: str) -> str:
    if "/" not in value:
        return value
    start, end = value.split("/", 1)
    if len(start) == 10 and "T" not in start:
        start = f"{start}T00:00:00Z"
    if len(end) == 10 and "T" not in end:
        end = f"{end}T23:59:59Z"
    return f"{start}/{end}"


def choose_item(items: list[dict]) -> dict:
    for item in items:
        assets = item.get("assets", {})
        if all(asset in assets for asset in BAND_ASSETS.values()):
            return item
    raise RuntimeError("Nenhuma cena retornada tem todos os assets blue/green/red/nir.")


def utm_crs_for_bbox(bbox_lonlat: tuple[float, float, float, float]) -> CRS:
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    lon = (min_lon + max_lon) / 2.0
    lat = (min_lat + max_lat) / 2.0
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def aligned_projected_grid(
    bbox_lonlat: tuple[float, float, float, float],
    dst_crs: CRS,
    resolution: float,
    max_pixels: int,
):
    to_dst = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True).transform
    projected = shapely_transform(to_dst, box(*bbox_lonlat))
    minx, miny, maxx, maxy = projected.bounds

    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    pixels = width * height
    if pixels > max_pixels:
        scale = float(np.sqrt(pixels / max_pixels))
        resolution = resolution * scale
        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))

    transform = from_origin(minx, maxy, resolution, resolution)
    return transform, width, height, resolution


def read_asset_to_grid(href: str, dst_crs: CRS, transform, width: int, height: int) -> np.ndarray:
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF"):
        with rasterio.open(href) as src:
            data = np.zeros((height, width), dtype=np.float32)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=0,
            )
    if float(np.nanmax(data)) > 2000:
        data = data / 10000.0
    return np.clip(np.nan_to_num(data, nan=0.0), 0.0, 1.0).astype(np.float32)


def write_netcdf(
    item: dict,
    bbox_lonlat: tuple[float, float, float, float],
    output_path: Path,
    resolution: float,
    max_pixels: int,
) -> None:
    dst_crs = utm_crs_for_bbox(bbox_lonlat)
    transform, width, height, actual_resolution = aligned_projected_grid(bbox_lonlat, dst_crs, resolution, max_pixels)

    bands = {}
    for canonical, asset_name in BAND_ASSETS.items():
        href = item["assets"][asset_name]["href"]
        bands[canonical] = read_asset_to_grid(href, dst_crs, transform, width, height)

    ndvi = (bands["B8"] - bands["B4"]) / (bands["B8"] + bands["B4"] + 1e-6)
    bands["NDVI"] = np.clip(ndvi, -1.0, 1.0).astype(np.float32)

    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    date = item.get("properties", {}).get("datetime", datetime.now(UTC).isoformat())

    data_vars = {
        band: (("time", "y", "x"), arr[None, :, :].astype(np.float32))
        for band, arr in bands.items()
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": [np.datetime64(date[:10])], "y": y.astype(np.float64), "x": x.astype(np.float64)},
    )
    ds["spatial_ref"] = xr.DataArray(0)
    ds["spatial_ref"].attrs["spatial_ref"] = dst_crs.to_wkt()
    ds.attrs["source_stac_id"] = item.get("id", "")
    ds.attrs["source_cloud_cover"] = float(item.get("properties", {}).get("eo:cloud_cover", -1))
    ds.attrs["bbox_lonlat"] = json.dumps(list(bbox_lonlat))
    ds.attrs["resolution_m"] = float(actual_resolution)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    ds.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa Sentinel-2 L2A para bbox dos fields ou bbox do mapa.")
    parser.add_argument("--fields-csv", type=Path, default=DEFAULT_FIELDS_CSV)
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date-range", type=str, default="2024-01-01/2026-06-19")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--max-pixels", type=int, default=1_500_000)
    parser.add_argument("--max-fields-for-bbox", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bbox:
        bbox_lonlat = tuple(args.bbox)
    else:
        bbox_lonlat = lonlat_bbox_from_fields(args.fields_csv, args.max_fields_for_bbox)

    if bbox_lonlat[0] >= bbox_lonlat[2] or bbox_lonlat[1] >= bbox_lonlat[3]:
        raise ValueError("BBox invalida. Ordem: min_lon min_lat max_lon max_lat.")

    items = query_sentinel2_items(bbox_lonlat, args.date_range, args.max_cloud, args.limit)
    if not items:
        raise RuntimeError("Nenhuma cena Sentinel-2 encontrada para a bbox/data/nuvem informadas.")
    item = choose_item(items)

    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in item.get("id", "sentinel2"))
    output_path = args.output_dir / f"{safe_id}.nc"
    write_netcdf(item, bbox_lonlat, output_path, args.resolution, args.max_pixels)
    print(f"Cena escolhida: {item.get('id')}")
    print(f"Nuvem: {item.get('properties', {}).get('eo:cloud_cover')}")
    print(f"Arquivo: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from shapely import wkt
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELDS_CSV = PROJECT_ROOT / "fields" / "fields_yuri.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "prepared" / "sentinel2_fields_yuri"


@dataclass(frozen=True)
class ImageGrid:
    path: Path
    image_name: str
    width: int
    height: int
    transform: Affine
    crs: CRS
    bounds_geom: object


def crs_from_dataset(ds: xr.Dataset) -> CRS | None:
    if "spatial_ref" not in ds:
        return None

    attrs = ds["spatial_ref"].attrs
    if "spatial_ref" in attrs:
        return CRS.from_wkt(attrs["spatial_ref"])
    if "crs_wkt" in attrs:
        return CRS.from_wkt(attrs["crs_wkt"])
    if "epsg_code" in attrs:
        return CRS.from_epsg(int(str(attrs["epsg_code"]).split(":")[-1]))
    return None


def transform_from_xy(ds: xr.Dataset) -> Affine:
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("Imagem sem coordenadas x/y; nao consigo rasterizar fields com seguranca.")

    x = np.asarray(ds.coords["x"].values, dtype=np.float64)
    y = np.asarray(ds.coords["y"].values, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        raise ValueError("Coordenadas x/y insuficientes para calcular transform.")

    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    west = float(x[0] - dx / 2.0)
    north = float(y[0] - dy / 2.0) if dy < 0 else float(y[0] + dy / 2.0)
    return from_origin(west, north, abs(dx), abs(dy))


def image_grid(path: Path) -> ImageGrid:
    ds = xr.open_dataset(path)
    try:
        crs = crs_from_dataset(ds)
        if crs is None:
            raise ValueError(f"Imagem sem CRS detectavel: {path}")

        if "x" not in ds.sizes or "y" not in ds.sizes:
            raise ValueError(f"Imagem sem dimensoes x/y: {path}")

        width = int(ds.sizes["x"])
        height = int(ds.sizes["y"])
        transform = transform_from_xy(ds)
        bounds = rasterio.transform.array_bounds(height, width, transform)
        bounds_geom = box(*bounds)
        return ImageGrid(
            path=path,
            image_name=path.name,
            width=width,
            height=height,
            transform=transform,
            crs=crs,
            bounds_geom=bounds_geom,
        )
    finally:
        ds.close()


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    images = sorted(input_path.rglob("*.nc"))
    if not images:
        raise FileNotFoundError(f"Nenhuma imagem .nc encontrada em {input_path}")
    return images


def fix_geometry(geom):
    if geom.is_empty:
        return None
    if geom.is_valid:
        return geom

    fixed = geom.buffer(0)
    if fixed.is_empty or not fixed.is_valid:
        return None
    return fixed


def source_geometry_in_range(geom, source_crs: CRS) -> bool:
    if not source_crs.is_geographic:
        return True

    minx, miny, maxx, maxy = geom.bounds
    return -180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90


def load_fields_for_grids(
    fields_csv: Path,
    grids: list[ImageGrid],
    geom_column: str,
    source_crs: CRS,
    chunksize: int,
    max_fields: int | None,
) -> tuple[list[object], dict[str, int]]:
    target_crs = grids[0].crs
    if any(grid.crs != target_crs for grid in grids):
        raise ValueError("As imagens precisam estar no mesmo CRS para este preparo. Separe por CRS/UTM.")

    to_target = Transformer.from_crs(source_crs, target_crs, always_xy=True).transform
    union_bounds = grids[0].bounds_geom
    for grid in grids[1:]:
        union_bounds = union_bounds.union(grid.bounds_geom)

    kept: list[object] = []
    stats = {
        "rows": 0,
        "missing_geom": 0,
        "invalid_geom": 0,
        "source_out_of_range": 0,
        "outside_images": 0,
        "kept": 0,
    }

    columns = pd.read_csv(fields_csv, nrows=0).columns.tolist()
    if geom_column not in columns:
        raise ValueError(f"Coluna de geometria '{geom_column}' nao existe. Colunas: {columns}")

    usecols = [geom_column]
    reader = pd.read_csv(
        fields_csv,
        usecols=usecols,
        chunksize=chunksize,
        dtype={geom_column: "string"},
        keep_default_na=False,
    )

    progress = tqdm(reader, desc="Lendo fields", unit="chunk")
    for chunk in progress:
        for geom_text in chunk[geom_column].astype(str):
            if max_fields is not None and stats["rows"] >= max_fields:
                progress.close()
                return kept, stats
            stats["rows"] += 1

            if not geom_text:
                stats["missing_geom"] += 1
                continue

            try:
                geom = fix_geometry(wkt.loads(geom_text))
            except Exception:
                geom = None

            if geom is None:
                stats["invalid_geom"] += 1
                continue

            if not source_geometry_in_range(geom, source_crs):
                stats["source_out_of_range"] += 1
                continue

            geom_projected = fix_geometry(shapely_transform(to_target, geom))
            if geom_projected is None:
                stats["invalid_geom"] += 1
                continue

            if not geom_projected.intersects(union_bounds):
                stats["outside_images"] += 1
                continue

            kept.append(geom_projected)
            stats["kept"] += 1

    return kept, stats


def split_name(index: int, total_images: int, val_fraction: float) -> str:
    if val_fraction <= 0 or total_images <= 1:
        return "train"
    val_count = max(1, int(round(total_images * val_fraction)))
    val_count = min(val_count, total_images - 1)
    return "val" if index < val_count else "train"


def write_mask(mask_path: Path, mask: np.ndarray, grid: ImageGrid) -> None:
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "deflate",
        "predictor": 2,
        "nodata": 0,
    }
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def copy_or_link_image(src: Path, dst: Path, hardlink: bool) -> None:
    if dst.exists():
        return
    if hardlink:
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def prepare_dataset(
    fields_csv: Path,
    images_path: Path,
    output_root: Path,
    geom_column: str,
    source_crs: CRS,
    chunksize: int,
    val_fraction: float,
    all_touched: bool,
    hardlink_images: bool,
    max_fields: int | None,
) -> None:
    image_paths = iter_images(images_path)
    grids = [image_grid(path) for path in tqdm(image_paths, desc="Lendo grids", unit="img")]

    fields, field_stats = load_fields_for_grids(
        fields_csv=fields_csv,
        grids=grids,
        geom_column=geom_column,
        source_crs=source_crs,
        chunksize=chunksize,
        max_fields=max_fields,
    )

    tree = STRtree(fields) if fields else None
    metadata_rows: dict[str, list[dict[str, object]]] = {"train": [], "val": []}
    summary_rows: list[dict[str, object]] = []

    for split in ["train", "val"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "masks" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)

    for idx, grid in enumerate(tqdm(grids, desc="Rasterizando masks", unit="img")):
        split = split_name(idx, len(grids), val_fraction)
        candidate_indexes: Iterable[int] = []
        if tree is not None:
            candidate_indexes = tree.query(grid.bounds_geom)

        shapes = []
        for candidate_idx in candidate_indexes:
            geom = fields[int(candidate_idx)]
            if geom.intersects(grid.bounds_geom):
                clipped = geom.intersection(grid.bounds_geom)
                if not clipped.is_empty:
                    shapes.append((clipped, 1))

        if shapes:
            mask = rasterize(
                shapes,
                out_shape=(grid.height, grid.width),
                fill=0,
                transform=grid.transform,
                all_touched=all_touched,
                dtype="uint8",
            )
        else:
            mask = np.zeros((grid.height, grid.width), dtype=np.uint8)

        image_dst = output_root / "images" / split / grid.image_name
        mask_dst = output_root / "masks" / split / grid.image_name.replace(".nc", ".tif")
        copy_or_link_image(grid.path, image_dst, hardlink_images)
        write_mask(mask_dst, mask, grid)

        positive_pixels = int(mask.sum())
        row = {
            "image": grid.image_name,
            "mask": mask_dst.name,
            "positive_pixels": positive_pixels,
            "field_polygons": len(shapes),
            "split": split,
        }
        metadata_rows[split].append(row)
        summary_rows.append(row)

    for split, rows in metadata_rows.items():
        pd.DataFrame(rows).to_csv(output_root / "metadata" / f"{split}.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(output_root / "metadata" / "all_images.csv", index=False)
    with (output_root / "metadata" / "field_ingest_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(field_stats.keys()))
        writer.writeheader()
        writer.writerow(field_stats)

    print("Preparo concluido")
    print(f"Output: {output_root}")
    print(f"Imagens: {len(grids)}")
    print(f"Fields lidos: {field_stats['rows']}")
    print(f"Fields aproveitados na area das imagens: {field_stats['kept']}")
    print(f"Masks com pixels positivos: {sum(1 for row in summary_rows if row['positive_pixels'] > 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rasteriza fields WKT sobre imagens Sentinel-2 .nc e gera dataset treinavel."
    )
    parser.add_argument("--fields-csv", type=Path, default=DEFAULT_FIELDS_CSV)
    parser.add_argument("--images", type=Path, required=True, help="Pasta ou arquivo .nc Sentinel-2.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--geom-column", type=str, default="geom")
    parser.add_argument("--source-crs", type=str, default="EPSG:4326")
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--all-touched", action="store_true", help="Marca pixels tocados pelo poligono.")
    parser.add_argument("--copy-images", action="store_true", help="Copia imagens em vez de tentar hardlink.")
    parser.add_argument("--max-fields", type=int, default=None, help="Limite util para smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.fields_csv.exists():
        raise FileNotFoundError(args.fields_csv)
    if not args.images.exists():
        raise FileNotFoundError(args.images)
    if not 0 <= args.val_fraction < 1:
        raise ValueError("--val-fraction precisa estar em [0, 1).")

    prepare_dataset(
        fields_csv=args.fields_csv,
        images_path=args.images,
        output_root=args.output_root,
        geom_column=args.geom_column,
        source_crs=CRS.from_string(args.source_crs),
        chunksize=args.chunksize,
        val_fraction=args.val_fraction,
        all_touched=args.all_touched,
        hardlink_images=not args.copy_images,
        max_fields=args.max_fields,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

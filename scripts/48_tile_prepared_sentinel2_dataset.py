from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.windows import Window
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def split_for_tile(tile_index: int, val_fraction: float) -> str:
    if val_fraction <= 0:
        return "train"
    interval = max(2, int(round(1.0 / val_fraction)))
    return "val" if tile_index % interval == 0 else "train"


def write_tile_nc(src_ds: xr.Dataset, output_path: Path, row: int, col: int, tile_size: int) -> None:
    tile = src_ds.isel(y=slice(row, row + tile_size), x=slice(col, col + tile_size)).copy(deep=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile.to_netcdf(output_path)
    tile.close()


def write_tile_mask(src_mask: rasterio.io.DatasetReader, output_path: Path, row: int, col: int, tile_size: int) -> int:
    window = Window(col, row, tile_size, tile_size)
    mask = src_mask.read(1, window=window).astype(np.uint8)
    transform = src_mask.window_transform(window)
    profile = src_mask.profile.copy()
    profile.update(
        height=tile_size,
        width=tile_size,
        transform=transform,
        dtype="uint8",
        count=1,
        compress="deflate",
        nodata=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)
    return int(mask.sum())


def tile_dataset(
    input_root: Path,
    output_root: Path,
    tile_size: int,
    stride: int,
    val_fraction: float,
    min_positive_pixels: int,
    keep_empty_ratio: float,
) -> None:
    rows = []
    tile_index = 0
    kept_empty = 0
    seen_empty = 0

    for split in ["train", "val"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "masks" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)

    image_paths = sorted((input_root / "images").rglob("*.nc"))
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem .nc encontrada em {input_root / 'images'}")

    for image_path in tqdm(image_paths, desc="Gerando tiles", unit="img"):
        original_split = image_path.parent.name
        mask_path = input_root / "masks" / original_split / image_path.name.replace(".nc", ".tif")
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        ds = xr.open_dataset(image_path)
        mask_src = rasterio.open(mask_path)
        try:
            height = int(ds.sizes["y"])
            width = int(ds.sizes["x"])
            if mask_src.height != height or mask_src.width != width:
                raise ValueError(f"Imagem/mask com shapes diferentes: {image_path.name}")

            max_row = height - tile_size
            max_col = width - tile_size
            if max_row < 0 or max_col < 0:
                continue

            for row in range(0, max_row + 1, stride):
                for col in range(0, max_col + 1, stride):
                    window = Window(col, row, tile_size, tile_size)
                    mask_preview = mask_src.read(1, window=window)
                    positive_pixels = int(mask_preview.sum())

                    if positive_pixels < min_positive_pixels:
                        seen_empty += 1
                        if keep_empty_ratio <= 0:
                            continue
                        allowed_empty = int(max(1, tile_index) * keep_empty_ratio)
                        if kept_empty >= allowed_empty:
                            continue
                        kept_empty += 1

                    out_split = split_for_tile(tile_index, val_fraction)
                    tile_name = f"{image_path.stem}_r{row:05d}_c{col:05d}.nc"
                    mask_name = tile_name.replace(".nc", ".tif")
                    out_image = output_root / "images" / out_split / tile_name
                    out_mask = output_root / "masks" / out_split / mask_name

                    write_tile_nc(ds, out_image, row, col, tile_size)
                    positive_pixels = write_tile_mask(mask_src, out_mask, row, col, tile_size)
                    rows.append(
                        {
                            "image": tile_name,
                            "mask": mask_name,
                            "positive_pixels": positive_pixels,
                            "source_image": image_path.name,
                            "row": row,
                            "col": col,
                            "split": out_split,
                        }
                    )
                    tile_index += 1
        finally:
            ds.close()
            mask_src.close()

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        raise RuntimeError("Nenhum tile gerado. Reduza --min-positive-pixels ou ajuste bbox/stride.")

    all_df.to_csv(output_root / "metadata" / "all_tiles.csv", index=False)
    for split in ["train", "val"]:
        split_df = all_df[all_df["split"] == split].copy()
        split_df.to_csv(output_root / "metadata" / f"{split}.csv", index=False)

    print(f"Tiles gerados: {len(all_df)}")
    print(f"Train: {int((all_df['split'] == 'train').sum())}")
    print(f"Val: {int((all_df['split'] == 'val').sum())}")
    print(f"Tiles positivos: {int((all_df['positive_pixels'] > 0).sum())}")
    print(f"Output: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quebra dataset Sentinel-2 preparado em tiles treinaveis.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-positive-pixels", type=int, default=1)
    parser.add_argument("--keep-empty-ratio", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(args.input_root)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_root} ja existe. Use --overwrite ou escolha outro output.")

    tile_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        tile_size=args.tile_size,
        stride=args.stride,
        val_fraction=args.val_fraction,
        min_positive_pixels=args.min_positive_pixels,
        keep_empty_ratio=args.keep_empty_ratio,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise

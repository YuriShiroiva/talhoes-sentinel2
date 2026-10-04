from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr


def normalize_percentile(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    arr = arr.astype(np.float32)
    valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo, hi = np.percentile(valid, [p_low, p_high])

    if np.isclose(lo, hi):
        return np.zeros_like(arr, dtype=np.float32)

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)

    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def make_rgb_from_dataset(ds: xr.Dataset, time_mode: str = "median", time_index: int = -1) -> np.ndarray:
    required = ["B4", "B3", "B2"]

    for band in required:
        if band not in ds:
            raise ValueError(f"Banda ausente no .nc: {band}")

    bands = []

    for band in required:
        data = ds[band]

        if "time" in data.dims:
            if time_mode == "median":
                arr = data.median(dim="time", skipna=True).values
            elif time_mode == "mean":
                arr = data.mean(dim="time", skipna=True).values
            elif time_mode == "index":
                arr = data.isel(time=time_index).values
            else:
                raise ValueError(f"time_mode inválido: {time_mode}")
        else:
            arr = data.values

        bands.append(normalize_percentile(arr))

    rgb = np.stack(bands, axis=-1)
    rgb = np.clip(rgb, 0, 1)

    return rgb


def get_mask_binary(mask: np.ndarray, positive_values: List[int] | None = None) -> np.ndarray:
    if positive_values is None:
        return (mask > 0).astype(np.uint8)

    positive_values = [int(v) for v in positive_values]
    return np.isin(mask, positive_values).astype(np.uint8)


def inspect_one_pair(
    image_path: Path,
    mask_path: Path,
    positive_values: List[int] | None,
) -> Dict:
    ds = xr.open_dataset(image_path)

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        mask_crs = str(src.crs) if src.crs is not None else None
        mask_transform = str(src.transform)
        mask_bounds = tuple(src.bounds)

    binary_mask = get_mask_binary(mask, positive_values)

    unique_values, unique_counts = np.unique(mask, return_counts=True)
    unique_dict = {str(int(v)): int(c) for v, c in zip(unique_values, unique_counts)}

    row = {
        "image_name": image_path.name,
        "mask_name": mask_path.name,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "nc_dims": json.dumps({k: int(v) for k, v in ds.sizes.items()}, ensure_ascii=False),
        "nc_data_vars": ",".join([v for v in ds.data_vars]),
        "nc_coords": ",".join([c for c in ds.coords]),
        "time_count": int(ds.sizes.get("time", 0)),
        "y_size": int(ds.sizes.get("y", 0)),
        "x_size": int(ds.sizes.get("x", 0)),
        "mask_shape": str(mask.shape),
        "mask_unique": json.dumps(unique_dict, ensure_ascii=False),
        "mask_positive_pixels": int(binary_mask.sum()),
        "mask_positive_ratio": float(binary_mask.mean()),
        "mask_crs": mask_crs,
        "mask_transform": mask_transform,
        "mask_bounds": json.dumps(mask_bounds),
    }

    if "time" in ds.coords:
        times = [str(t) for t in ds["time"].values]
        row["times"] = json.dumps(times, ensure_ascii=False)
    else:
        row["times"] = "[]"

    if "x" in ds.coords:
        x_values = ds["x"].values
        row["x_min"] = float(np.nanmin(x_values))
        row["x_max"] = float(np.nanmax(x_values))

    if "y" in ds.coords:
        y_values = ds["y"].values
        row["y_min"] = float(np.nanmin(y_values))
        row["y_max"] = float(np.nanmax(y_values))

    ds.close()

    return row


def save_preview(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    positive_values: List[int] | None,
    time_mode: str,
    time_index: int,
):
    ds = xr.open_dataset(image_path)
    rgb = make_rgb_from_dataset(ds, time_mode=time_mode, time_index=time_index)

    ndvi_preview = None
    if "NDVI" in ds:
        ndvi = ds["NDVI"]
        if "time" in ndvi.dims:
            if time_mode == "median":
                ndvi_arr = ndvi.median(dim="time", skipna=True).values
            elif time_mode == "mean":
                ndvi_arr = ndvi.mean(dim="time", skipna=True).values
            else:
                ndvi_arr = ndvi.isel(time=time_index).values
        else:
            ndvi_arr = ndvi.values

        ndvi_preview = normalize_percentile(ndvi_arr)

    ds.close()

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    binary_mask = get_mask_binary(mask, positive_values)

    fig, axes = plt.subplots(1, 4 if ndvi_preview is not None else 3, figsize=(16, 4))

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB\nB4/B3/B2")
    axes[0].axis("off")

    if ndvi_preview is not None:
        axes[1].imshow(ndvi_preview)
        axes[1].set_title("NDVI")
        axes[1].axis("off")
        mask_ax = axes[2]
        overlay_ax = axes[3]
    else:
        mask_ax = axes[1]
        overlay_ax = axes[2]

    mask_ax.imshow(binary_mask)
    mask_ax.set_title("Máscara binária")
    mask_ax.axis("off")

    overlay_ax.imshow(rgb)
    overlay = np.ma.masked_where(binary_mask <= 0, binary_mask)
    overlay_ax.imshow(overlay, cmap="autumn", alpha=0.45, vmin=0, vmax=1)
    overlay_ax.set_title("Overlay")
    overlay_ax.axis("off")

    fig.suptitle(image_path.name, fontsize=10)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_pairs(root: Path, split: str, metadata_csv: Path | None = None) -> List[Tuple[Path, Path]]:
    image_dir = root / "images" / split
    mask_dir = root / "masks" / split

    if metadata_csv is not None and metadata_csv.exists():
        df = pd.read_csv(metadata_csv)

        if "image" not in df.columns:
            raise ValueError(f"O metadata CSV precisa ter a coluna 'image'. Colunas encontradas: {list(df.columns)}")

        names = df["image"].astype(str).tolist()
    else:
        names = [p.name for p in sorted(image_dir.glob("*.nc"))]

    pairs = []

    for image_name in names:
        image_path = image_dir / image_name
        mask_path = mask_dir / image_name.replace(".nc", ".tif")

        if image_path.exists() and mask_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Inspeciona o dataset Sentinel-2 do AI4Boundaries.")

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/ai4boundaries/sentinel2"),
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sentinel2/test_023_inspection_val"),
    )
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--num-previews", type=int, default=12)
    parser.add_argument(
        "--positive-values",
        type=str,
        default="",
        help="Valores da máscara considerados positivos. Ex: '1,2'. Vazio = mask > 0.",
    )
    parser.add_argument(
        "--time-mode",
        type=str,
        default="median",
        choices=["median", "mean", "index"],
    )
    parser.add_argument("--time-index", type=int, default=-1)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = args.root / "metadata" / f"{args.split}.csv"

    positive_values = None
    if args.positive_values.strip():
        positive_values = [int(v.strip()) for v in args.positive_values.split(",") if v.strip()]

    pairs = build_pairs(args.root, args.split, metadata_csv=metadata_csv)

    print(f"Root: {args.root}")
    print(f"Split: {args.split}")
    print(f"Metadata: {metadata_csv if metadata_csv.exists() else 'não encontrado'}")
    print(f"Pares encontrados: {len(pairs)}")
    print(f"Positive values: {positive_values if positive_values is not None else 'mask > 0'}")

    if not pairs:
        raise RuntimeError("Nenhum par .nc/.tif encontrado.")

    rows = []

    for idx, (image_path, mask_path) in enumerate(pairs[: args.max_samples]):
        print(f"Inspecionando {idx + 1}/{min(args.max_samples, len(pairs))}: {image_path.name}")

        row = inspect_one_pair(
            image_path=image_path,
            mask_path=mask_path,
            positive_values=positive_values,
        )
        rows.append(row)

        if idx < args.num_previews:
            preview_path = args.output_dir / "previews" / f"preview_{idx + 1:02d}_{image_path.stem}.png"
            save_preview(
                image_path=image_path,
                mask_path=mask_path,
                output_path=preview_path,
                positive_values=positive_values,
                time_mode=args.time_mode,
                time_index=args.time_index,
            )

    summary_df = pd.DataFrame(rows)

    summary_csv = args.output_dir / "sentinel2_inspection_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    dataset_info = {
        "root": str(args.root),
        "split": args.split,
        "num_pairs_total": len(pairs),
        "num_pairs_inspected": len(rows),
        "metadata_csv": str(metadata_csv) if metadata_csv.exists() else None,
        "positive_values": positive_values,
        "time_mode": args.time_mode,
        "time_index": args.time_index,
        "first_image": str(pairs[0][0]),
        "first_mask": str(pairs[0][1]),
    }

    info_json = args.output_dir / "dataset_info.json"
    info_json.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nResumo rápido:")
    print(summary_df[["image_name", "nc_dims", "nc_data_vars", "mask_shape", "mask_unique", "mask_positive_ratio"]].head(10).to_string(index=False))

    print("\nArquivos salvos:")
    print("-", summary_csv)
    print("-", info_json)
    print("-", args.output_dir / "previews")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy import ndimage
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from field_boundary.models.factory import build_model


def build_official_val_pairs(data_root: Path) -> List[Tuple[Path, Path]]:
    image_dir = data_root / "orthos" / "images" / "val"
    mask_dir = data_root / "orthos" / "masks" / "val"

    pairs = []

    for image_path in sorted(image_dir.glob("*_ortho_1m_512.tif")):
        mask_name = image_path.name.replace(
            "_ortho_1m_512.tif",
            "_ortholabel_1m_512.tif",
        )
        mask_path = mask_dir / mask_name

        if mask_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


def infer_architecture_from_state_dict(state_dict: Dict) -> str | None:
    keys = list(state_dict.keys())

    if any(k.startswith("decoder.aspp") or k.startswith("decoder.block1") for k in keys):
        return "deeplabv3plus"

    if any(k.startswith("decoder.blocks") for k in keys):
        return "unet"

    return None


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    state_dict = checkpoint["model_state_dict"]

    architecture = config.get("architecture") or infer_architecture_from_state_dict(state_dict) or "unet"
    encoder_name = config.get("encoder_name", "resnet34")
    in_channels = int(config.get("in_channels", 3))

    print(f"\nCarregando modelo: {checkpoint_path}")
    print(f"Arquitetura: {architecture}")
    print(f"Encoder: {encoder_name}")
    print(f"In channels: {in_channels}")

    model = build_model(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def normalize_image_percentile(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)

    for c in range(img.shape[0]):
        band = img[c]
        valid = band[np.isfinite(band)]

        if valid.size == 0:
            continue

        p2, p98 = np.percentile(valid, [2, 98])

        if np.isclose(p2, p98):
            out[c] = 0.0
            continue

        band = np.clip(band, p2, p98)
        out[c] = (band - p2) / (p98 - p2 + eps)

    return out


def force_channel_count(img: np.ndarray, in_channels: int) -> np.ndarray:
    channels, _, _ = img.shape

    if channels == in_channels:
        return img

    if channels > in_channels:
        return img[:in_channels]

    missing = in_channels - channels
    last_band = img[-1:, :, :]
    padding = np.repeat(last_band, repeats=missing, axis=0)

    return np.concatenate([img, padding], axis=0)


def prepare_image_tensor(image_path: Path, image_size: int, in_channels: int):
    with rasterio.open(image_path) as src:
        img = src.read()
        profile = src.profile
        transform = src.transform
        crs = src.crs
        original_height = src.height
        original_width = src.width

    img = force_channel_count(img, in_channels)
    img = normalize_image_percentile(img)

    tensor = torch.from_numpy(img).float().unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    return tensor, profile, transform, crs, original_height, original_width


def read_original_mask(mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    return (mask > 0).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_object_area: int) -> np.ndarray:
    if min_object_area <= 0:
        return mask.astype(np.uint8)

    mask_bool = mask.astype(bool)
    labeled, num_features = ndimage.label(mask_bool)

    if num_features == 0:
        return mask.astype(np.uint8)

    component_sizes = np.bincount(labeled.ravel())
    keep = component_sizes >= min_object_area
    keep[0] = False

    cleaned = keep[labeled]

    return cleaned.astype(np.uint8)


def postprocess_mask(mask: np.ndarray, min_object_area: int, fill_holes: bool) -> np.ndarray:
    out = mask.astype(np.uint8)

    if fill_holes:
        out = ndimage.binary_fill_holes(out.astype(bool)).astype(np.uint8)

    out = remove_small_components(out, min_object_area=min_object_area)

    return out.astype(np.uint8)


def pixel_area_m2_from_transform(transform) -> float:
    """
    Calcula área do pixel pelo determinante da transformação afim.
    Para ortofotos 1 m, isso tende a ser próximo de 1 m².
    """
    return abs(transform.a * transform.e - transform.b * transform.d)


def polygonize_mask(mask: np.ndarray, transform, image_id: str, source: str, crs_text: str):
    features = []

    mask_uint8 = mask.astype(np.uint8)

    for geom, value in shapes(mask_uint8, mask=mask_uint8.astype(bool), transform=transform):
        if int(value) != 1:
            continue

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "image_id": image_id,
                    "source": source,
                    "crs": crs_text,
                },
                "geometry": geom,
            }
        )

    return features


def save_geojson(features: List[Dict], output_path: Path, crs_text: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    collection = {
        "type": "FeatureCollection",
        "name": output_path.stem,
        "crs": {
            "type": "name",
            "properties": {
                "name": crs_text,
            },
        },
        "features": features,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(collection, f)


def summarize_area_metrics(df: pd.DataFrame) -> Dict:
    non_empty = df[df["real_area_m2"] > 0].copy()
    empty = df[df["real_area_m2"] == 0].copy()

    summary = {
        "num_samples": int(len(df)),
        "num_non_empty": int(len(non_empty)),
        "num_empty": int(len(empty)),
        "pred_area_mean_m2": float(df["pred_area_m2"].mean()),
        "real_area_mean_m2": float(df["real_area_m2"].mean()),
        "abs_error_mean_m2": float(df["abs_error_m2"].mean()),
        "abs_error_median_m2": float(df["abs_error_m2"].median()),
        "rmse_area_m2": float(np.sqrt(np.mean(df["area_error_m2"] ** 2))),
        "pred_polygon_count_mean": float(df["pred_polygon_count"].mean()),
        "pred_polygon_count_median": float(df["pred_polygon_count"].median()),
    }

    if len(non_empty) > 0:
        summary.update(
            {
                "non_empty_abs_error_mean_m2": float(non_empty["abs_error_m2"].mean()),
                "non_empty_abs_error_median_m2": float(non_empty["abs_error_m2"].median()),
                "non_empty_mape_pct": float(non_empty["area_error_pct"].mean()),
                "non_empty_median_ape_pct": float(non_empty["area_error_pct"].median()),
            }
        )

    if len(empty) > 0:
        summary.update(
            {
                "empty_pred_area_mean_m2": float(empty["pred_area_m2"].mean()),
                "empty_false_positive_rate": float((empty["pred_area_m2"] > 0).mean()),
            }
        )

    return summary


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Teste 012 — Poligonização e erro de área do ensemble")

    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ai4boundaries"))
    parser.add_argument("--unet-checkpoint", type=Path, required=True)
    parser.add_argument("--deeplab-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)

    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--unet-weight", type=float, default=0.70)
    parser.add_argument("--min-object-area", type=int, default=0)
    parser.add_argument("--fill-holes", action="store_true")

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-geojson", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    pairs = build_official_val_pairs(args.data_root)

    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]

    print(f"Amostras avaliadas: {len(pairs)}")

    unet_model = load_model(args.unet_checkpoint, device)
    deeplab_model = load_model(args.deeplab_checkpoint, device)

    deeplab_weight = 1.0 - args.unet_weight

    rows = []
    pred_features_all = []
    gt_features_all = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        if idx % 50 == 0:
            print(f"Processando {idx}/{len(pairs)}...")

        image_id = image_path.stem.replace("_ortho_1m_512", "")

        image_tensor, profile, transform, crs, original_height, original_width = prepare_image_tensor(
            image_path=image_path,
            image_size=args.image_size,
            in_channels=args.in_channels,
        )

        image_tensor = image_tensor.to(device)

        probs_unet = torch.sigmoid(unet_model(image_tensor))
        probs_deeplab = torch.sigmoid(deeplab_model(image_tensor))

        probs_final = args.unet_weight * probs_unet + deeplab_weight * probs_deeplab

        probs_original = F.interpolate(
            probs_final,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )

        prob_np = probs_original[0, 0].detach().cpu().numpy()
        pred_mask = (prob_np >= args.threshold).astype(np.uint8)

        pred_mask = postprocess_mask(
            pred_mask,
            min_object_area=args.min_object_area,
            fill_holes=args.fill_holes,
        )

        gt_mask = read_original_mask(mask_path)

        pixel_area_m2 = pixel_area_m2_from_transform(transform)

        pred_pixels = int(pred_mask.sum())
        real_pixels = int(gt_mask.sum())

        pred_area_m2 = pred_pixels * pixel_area_m2
        real_area_m2 = real_pixels * pixel_area_m2

        area_error_m2 = pred_area_m2 - real_area_m2
        abs_error_m2 = abs(area_error_m2)

        if real_area_m2 > 0:
            area_error_pct = abs_error_m2 / real_area_m2 * 100.0
        else:
            area_error_pct = np.nan

        labeled_pred, pred_polygon_count = ndimage.label(pred_mask.astype(bool))
        labeled_gt, gt_polygon_count = ndimage.label(gt_mask.astype(bool))

        crs_text = crs.to_string() if crs is not None else "unknown"

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "crs": crs_text,
                "pixel_area_m2": pixel_area_m2,
                "threshold": args.threshold,
                "unet_weight": args.unet_weight,
                "deeplab_weight": deeplab_weight,
                "min_object_area": args.min_object_area,
                "fill_holes": args.fill_holes,
                "pred_pixels": pred_pixels,
                "real_pixels": real_pixels,
                "pred_area_m2": pred_area_m2,
                "real_area_m2": real_area_m2,
                "area_error_m2": area_error_m2,
                "abs_error_m2": abs_error_m2,
                "area_error_pct": area_error_pct,
                "pred_polygon_count": int(pred_polygon_count),
                "gt_polygon_count": int(gt_polygon_count),
            }
        )

        if args.export_geojson:
            pred_features_all.extend(
                polygonize_mask(
                    pred_mask,
                    transform,
                    image_id=image_id,
                    source="prediction",
                    crs_text=crs_text,
                )
            )

            gt_features_all.extend(
                polygonize_mask(
                    gt_mask,
                    transform,
                    image_id=image_id,
                    source="ground_truth",
                    crs_text=crs_text,
                )
            )

    df = pd.DataFrame(rows)

    metrics_path = args.output_dir / "area_metrics.csv"
    df.to_csv(metrics_path, index=False)

    summary = summarize_area_metrics(df)

    summary_path = args.output_dir / "area_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.export_geojson:
        crs_text = rows[0]["crs"] if rows else "unknown"

        save_geojson(
            pred_features_all,
            args.output_dir / "predicted_polygons.geojson",
            crs_text=crs_text,
        )

        save_geojson(
            gt_features_all,
            args.output_dir / "groundtruth_polygons.geojson",
            crs_text=crs_text,
        )

    print("\nResumo de área:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nArquivos salvos em: {args.output_dir}")
    print(f"- {metrics_path}")
    print(f"- {summary_path}")

    if args.export_geojson:
        print(f"- {args.output_dir / 'predicted_polygons.geojson'}")
        print(f"- {args.output_dir / 'groundtruth_polygons.geojson'}")


if __name__ == "__main__":
    main()
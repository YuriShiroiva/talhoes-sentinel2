from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from rasterio.features import shapes, sieve
from rasterio.warp import transform_geom

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from field_boundary.data.dataset import AI4BoundariesSegmentationDataset
from field_boundary.models.factory import build_model


try:
    from shapely.geometry import shape as shapely_shape
    from shapely.geometry import mapping as shapely_mapping
    SHAPELY_AVAILABLE = True
except Exception:
    SHAPELY_AVAILABLE = False


def infer_architecture_from_state_dict(state_dict: Dict) -> str | None:
    keys = list(state_dict.keys())

    if any(k.startswith("decoder.aspp") or k.startswith("decoder.block1") for k in keys):
        return "deeplabv3plus"

    if any(k.startswith("decoder.blocks") for k in keys):
        return "unet"

    return None


def load_segmenter(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    state_dict = checkpoint["model_state_dict"]

    architecture = config.get("architecture") or infer_architecture_from_state_dict(state_dict) or "unet"
    encoder_name = config.get("encoder_name", "resnet34")
    in_channels = int(config.get("in_channels", 3))

    print(f"\nCarregando segmentador: {checkpoint_path}")
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


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in ["true", "1", "yes", "sim"]


def image_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().cpu().numpy()

    if img.shape[0] >= 3:
        rgb = np.transpose(img[:3], (1, 2, 0))
    else:
        rgb = np.repeat(img[0:1], 3, axis=0)
        rgb = np.transpose(rgb, (1, 2, 0))

    rgb = np.nan_to_num(rgb)
    rgb = np.clip(rgb, 0, 1)

    return rgb


def select_cases(
    df: pd.DataFrame,
    top_k: int,
    min_target_ratio: float,
    include_empty: bool,
) -> pd.DataFrame:
    df = df.copy()

    required = [
        "image_path",
        "mask_path",
        "dice_after_gate",
        "iou_after_gate",
        "precision_after_gate",
        "recall_after_gate",
        "f1_after_gate",
        "pass_gate",
        "prob_non_empty",
        "target_pixels",
        "total_pixels",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "O CSV precisa ser o per_sample_metrics.csv gerado pelo script 10_evaluate_final_unet_pipeline.py. "
            f"Colunas ausentes: {missing}"
        )

    if "target_ratio" not in df.columns:
        df["target_ratio"] = df["target_pixels"] / df["total_pixels"]

    df["pass_gate_bool"] = df["pass_gate"].apply(to_bool)

    if not include_empty:
        df = df[df["target_pixels"] > 0].copy()
        df = df[df["pass_gate_bool"]].copy()

    df = df[df["target_ratio"] >= min_target_ratio].copy()

    selected = df.sort_values(
        ["dice_after_gate", "iou_after_gate", "precision_after_gate"],
        ascending=False,
    ).head(top_k)

    return selected.reset_index(drop=True)


def upsample_mask_to_raster(mask_2d: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    tensor = torch.from_numpy(mask_2d.astype(np.float32))[None, None, :, :]
    resized = F.interpolate(tensor, size=(out_height, out_width), mode="nearest")
    return resized[0, 0].numpy().astype(np.uint8)


def simplify_geometry_if_possible(geometry: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    if tolerance <= 0 or not SHAPELY_AVAILABLE:
        return geometry

    try:
        geom = shapely_shape(geometry)
        geom = geom.simplify(tolerance, preserve_topology=True)
        return shapely_mapping(geom)
    except Exception:
        return geometry


def polygonize_mask(
    mask: np.ndarray,
    transform,
    image_id: str,
    rank: int,
    row: pd.Series,
    src_crs,
    min_area_pixels: int,
    simplify_tolerance: float,
) -> List[Dict[str, Any]]:
    mask = mask.astype(np.uint8)

    if min_area_pixels > 0:
        mask = sieve(mask, size=min_area_pixels, connectivity=8).astype(np.uint8)

    features = []

    for geom, value in shapes(mask, mask=mask == 1, transform=transform, connectivity=8):
        if int(value) != 1:
            continue

        geom = simplify_geometry_if_possible(geom, simplify_tolerance)

        properties = {
            "image_id": image_id,
            "rank": int(rank),
            "source_image": str(row["image_path"]),
            "prob_non_empty": float(row["prob_non_empty"]),
            "dice_after_gate": float(row["dice_after_gate"]),
            "iou_after_gate": float(row["iou_after_gate"]),
            "precision_after_gate": float(row["precision_after_gate"]),
            "recall_after_gate": float(row["recall_after_gate"]),
            "f1_after_gate": float(row["f1_after_gate"]),
            "target_ratio": float(row.get("target_ratio", 0.0)),
            "crs_source": str(src_crs) if src_crs else None,
        }

        feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": geom,
        }

        features.append(feature)

    return features


def transform_features_to_wgs84(features: List[Dict[str, Any]], src_crs) -> List[Dict[str, Any]]:
    if src_crs is None:
        return []

    transformed = []

    for feature in features:
        geom_wgs84 = transform_geom(
            src_crs,
            "EPSG:4326",
            feature["geometry"],
            precision=7,
        )

        new_feature = {
            "type": "Feature",
            "properties": dict(feature["properties"]),
            "geometry": geom_wgs84,
        }

        new_feature["properties"]["crs_source"] = str(src_crs)
        new_feature["properties"]["crs_output"] = "EPSG:4326"

        transformed.append(new_feature)

    return transformed


def save_feature_collection(features: List[Dict[str, Any]], path: Path, name: str):
    collection = {
        "type": "FeatureCollection",
        "name": name,
        "features": features,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")


def save_preview(
    rgb: np.ndarray,
    target_mask: np.ndarray,
    pred_mask: np.ndarray,
    path: Path,
    title: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(rgb)
    axes[0].set_title("Ortofoto")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    target_overlay = np.ma.masked_where(target_mask <= 0, target_mask)
    axes[1].imshow(target_overlay, cmap="autumn", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title("Máscara real")
    axes[1].axis("off")

    axes[2].imshow(rgb)
    pred_overlay = np.ma.masked_where(pred_mask <= 0, pred_mask)
    axes[2].imshow(pred_overlay, cmap="autumn", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Talhão candidato IA")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Gera GeoJSONs de talhões candidatos a partir dos melhores exemplos do pipeline final."
    )

    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/final_model/test020_unet_pipeline_clean/per_sample_metrics.csv"),
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        type=Path,
        default=Path("outputs/tests/test_019_unet_imagenet_3000samples_20epochs/best_unet_baseline.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_model/test020_top25_geojson"),
    )

    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--segmentation-threshold", type=float, default=0.65)
    parser.add_argument("--min-target-ratio", type=float, default=0.01)
    parser.add_argument("--include-empty", action="store_true")

    parser.add_argument(
        "--min-area-pixels",
        type=int,
        default=64,
        help="Remove componentes menores que esse número de pixels após redimensionar para o raster original.",
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=0.0,
        help="Simplificação geométrica na unidade do CRS do raster. Use 0 para não simplificar.",
    )

    args = parser.parse_args()

    if not args.metrics_csv.exists():
        raise FileNotFoundError(f"Não encontrei o CSV: {args.metrics_csv}")

    if not args.segmenter_checkpoint.exists():
        raise FileNotFoundError(f"Não encontrei o checkpoint: {args.segmenter_checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    df = pd.read_csv(args.metrics_csv)
    selected = select_cases(
        df=df,
        top_k=args.top_k,
        min_target_ratio=args.min_target_ratio,
        include_empty=args.include_empty,
    )

    if selected.empty:
        raise RuntimeError("Nenhum caso selecionado. Reduza --min-target-ratio ou use --include-empty.")

    selected_path = args.output_dir / "selected_cases_for_geojson.csv"
    selected.to_csv(selected_path, index=False)

    segmenter = load_segmenter(args.segmenter_checkpoint, device)

    all_features_source = []
    all_features_wgs84 = []
    summary_rows = []

    individual_source_dir = args.output_dir / "individual_source_crs"
    individual_wgs84_dir = args.output_dir / "individual_wgs84"
    preview_dir = args.output_dir / "previews"

    for rank, row in selected.iterrows():
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        image_id = image_path.stem.replace("_ortho_1m_512", "")

        print(f"\nProcessando Top {rank + 1:02d}: {image_path.name}")

        dataset = AI4BoundariesSegmentationDataset(
            pairs=[(image_path, mask_path)],
            image_size=args.image_size,
            in_channels=args.in_channels,
            augment=False,
        )

        sample = dataset[0]

        image = sample["image"].unsqueeze(0).to(device)
        target_small = sample["mask"].detach().cpu().numpy()[0].astype(np.uint8)

        probs = torch.sigmoid(segmenter(image))
        prob_np = probs[0, 0].detach().cpu().numpy()
        pred_small = (prob_np >= args.segmentation_threshold).astype(np.uint8)

        pass_gate = str(row["pass_gate"]).strip().lower() in ["true", "1", "yes", "sim"]

        if not pass_gate:
            pred_small[:] = 0

        with rasterio.open(image_path) as src:
            out_height = src.height
            out_width = src.width
            transform = src.transform
            src_crs = src.crs

        pred_original = upsample_mask_to_raster(pred_small, out_height, out_width)
        target_original = upsample_mask_to_raster(target_small, out_height, out_width)

        features_source = polygonize_mask(
            mask=pred_original,
            transform=transform,
            image_id=image_id,
            rank=rank + 1,
            row=row,
            src_crs=src_crs,
            min_area_pixels=args.min_area_pixels,
            simplify_tolerance=args.simplify_tolerance,
        )

        features_wgs84 = transform_features_to_wgs84(features_source, src_crs)

        all_features_source.extend(features_source)
        all_features_wgs84.extend(features_wgs84)

        source_path = individual_source_dir / f"top_{rank + 1:02d}_{image_id}_source_crs.geojson"
        save_feature_collection(features_source, source_path, name=f"top_{rank + 1:02d}_{image_id}_source_crs")

        if features_wgs84:
            wgs84_path = individual_wgs84_dir / f"top_{rank + 1:02d}_{image_id}_wgs84.geojson"
            save_feature_collection(features_wgs84, wgs84_path, name=f"top_{rank + 1:02d}_{image_id}_wgs84")

        rgb = image_to_rgb(sample["image"])
        preview_path = preview_dir / f"top_{rank + 1:02d}_{image_id}_preview.png"
        save_preview(
            rgb=rgb,
            target_mask=target_small,
            pred_mask=pred_small,
            path=preview_path,
            title=f"Top {rank + 1:02d} | Dice={row['dice_after_gate']:.3f} | polígonos={len(features_source)}",
        )

        summary_rows.append(
            {
                "rank": rank + 1,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "image_id": image_id,
                "num_polygons": len(features_source),
                "has_wgs84": bool(features_wgs84),
                "source_crs": str(src_crs) if src_crs else "",
                "dice_after_gate": float(row["dice_after_gate"]),
                "iou_after_gate": float(row["iou_after_gate"]),
                "precision_after_gate": float(row["precision_after_gate"]),
                "recall_after_gate": float(row["recall_after_gate"]),
                "f1_after_gate": float(row["f1_after_gate"]),
                "prob_non_empty": float(row["prob_non_empty"]),
                "target_ratio": float(row.get("target_ratio", 0.0)),
                "individual_source_geojson": str(source_path),
            }
        )

        print(f"Polígonos gerados: {len(features_source)}")
        print(f"GeoJSON source CRS: {source_path}")

    combined_source_path = args.output_dir / "top25_talhoes_candidatos_source_crs.geojson"
    save_feature_collection(all_features_source, combined_source_path, name="top25_talhoes_candidatos_source_crs")

    if all_features_wgs84:
        combined_wgs84_path = args.output_dir / "top25_talhoes_candidatos_wgs84.geojson"
        save_feature_collection(all_features_wgs84, combined_wgs84_path, name="top25_talhoes_candidatos_wgs84")
    else:
        combined_wgs84_path = None

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = args.output_dir / "summary_geojson_generation.csv"
    summary_df.to_csv(summary_csv, index=False)

    readme = args.output_dir / "README_geojson_demo.md"
    lines = [
        "# Demonstração de talhões candidatos gerados pela IA",
        "",
        "Este diretório contém GeoJSONs gerados a partir das melhores predições do pipeline final.",
        "",
        "## Arquivos principais",
        "",
        f"- GeoJSON combinado no CRS original: `{combined_source_path.name}`",
        f"- GeoJSON combinado em WGS84: `{combined_wgs84_path.name if combined_wgs84_path else 'não gerado; raster sem CRS identificado'}`",
        f"- Resumo CSV: `{summary_csv.name}`",
        "- GeoJSONs individuais: `individual_source_crs/`",
        "- Previews PNG: `previews/`",
        "",
        "## Observação importante",
        "",
        "Se os rasters originais tiverem CRS/georreferenciamento válido, o script também gera uma versão WGS84, adequada para abrir em ferramentas web/GIS.",
        "Se não houver CRS, o GeoJSON no CRS original pode estar em coordenadas locais/pixel e deve ser usado apenas como demonstração técnica.",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")

    print("\nFinalizado!")
    print(f"Casos selecionados: {selected_path}")
    print(f"GeoJSON combinado source CRS: {combined_source_path}")
    if combined_wgs84_path:
        print(f"GeoJSON combinado WGS84: {combined_wgs84_path}")
    print(f"Resumo: {summary_csv}")
    print(f"Previews: {preview_dir}")
    print(f"README: {readme}")


if __name__ == "__main__":
    main()

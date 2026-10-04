from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re
import sys
from pathlib import Path
from typing import Any

import torch  # importar torch ANTES de rasterio/GDAL (evita conflito de DLL no Windows)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from scipy import ndimage
from tqdm import tqdm

try:
    import xarray as xr
except ImportError as exc:
    raise RuntimeError("Faltou xarray. Instale com: pip install xarray netCDF4 h5netcdf") from exc

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise RuntimeError("Faltou segmentation_models_pytorch. Instale as dependencias do projeto.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from field_boundary.geospatial.polygonize import polygonize_mask, write_geojson, _pixel_area
from field_boundary.data.dataset import normalize_image_percentile


DEFAULT_CHECKPOINTS = [
    PROJECT_ROOT / "outputs/sentinel2/test_025_unet_s2_stack30_a100/best_sentinel2_unet.pt",
    PROJECT_ROOT / "outputs/sentinel2/test_026_deeplab_s2_stack30_a100/best_sentinel2_unet.pt",
    PROJECT_ROOT / "outputs/sentinel2/test_028_unet_s2_stack30_resnet50_a100/best_sentinel2_unet.pt",
    PROJECT_ROOT / "outputs/sentinel2/test_035_unet_s2_stack30_tversky_mild_sampling/best_sentinel2_unet.pt",
    PROJECT_ROOT / "outputs/sentinel2/test_038_deeplab_s2_stack30_resnet50_a100/best_sentinel2_unet.pt",
    PROJECT_ROOT / "outputs/sentinel2/test_052_unetplusplus_s2_stack30_resnet50_a100/best_sentinel2_unet.pt",
]

DEFAULT_WEIGHTS = "0.22,0.12,0.21,0.13,0.13,0.19"
BAND_ALIASES = {
    "B2": ["B2", "B02", "b2", "b02", "blue", "BLUE"],
    "B3": ["B3", "B03", "b3", "b03", "green", "GREEN"],
    "B4": ["B4", "B04", "b4", "b04", "red", "RED"],
    "B8": ["B8", "B08", "b8", "b08", "nir", "NIR"],
    "NDVI": ["NDVI", "ndvi"],
}


def safe_name(value: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(value)).stem)


def find_var(ds: xr.Dataset, canonical: str) -> str | None:
    for candidate in BAND_ALIASES.get(canonical, [canonical]):
        if candidate in ds.data_vars:
            return candidate
    return None


def to_time_y_x(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Array precisa ser 2D ou 3D; recebido shape={arr.shape}")

    small_axes = [idx for idx, size in enumerate(arr.shape) if size <= 30]
    if 0 in small_axes:
        return arr
    if 2 in small_axes:
        return np.moveaxis(arr, 2, 0)
    if 1 in small_axes:
        return np.moveaxis(arr, 1, 0)
    return arr


def normalize_band(arr: np.ndarray, band: str) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if band == "NDVI":
        return np.clip(arr, -1.0, 1.0).astype(np.float32)

    maxv = float(np.max(arr)) if arr.size else 1.0
    if maxv > 2000:
        arr = arr / 10000.0
    elif maxv > 5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def crs_from_dataset(ds: xr.Dataset) -> CRS | None:
    if "spatial_ref" in ds:
        attrs = ds["spatial_ref"].attrs
        if "spatial_ref" in attrs:
            return CRS.from_wkt(attrs["spatial_ref"])
        if "crs_wkt" in attrs:
            return CRS.from_wkt(attrs["crs_wkt"])
        if "epsg_code" in attrs:
            return CRS.from_epsg(int(str(attrs["epsg_code"]).split(":")[-1]))
    return None


def transform_from_xy(ds: xr.Dataset, width: int, height: int) -> Affine:
    if "x" not in ds.coords or "y" not in ds.coords:
        return Affine.identity()

    x = np.asarray(ds.coords["x"].values, dtype=np.float64)
    y = np.asarray(ds.coords["y"].values, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return Affine.identity()

    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    west = float(x[0] - dx / 2.0)
    north = float(y[0] - dy / 2.0) if dy < 0 else float(y[0] + dy / 2.0)
    return from_origin(west, north, abs(dx), abs(dy))


def read_netcdf_stack(
    image_path: Path,
    bands: list[str],
    time_steps: int,
    stack_order: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], Affine, CRS | None]:
    ds = xr.open_dataset(image_path)
    try:
        # Le bandas CRUAS (sem normalizar aqui); a normalizacao do treino
        # (percentil 2-98 por canal) e aplicada no stack montado, mais abaixo.
        band_data: dict[str, np.ndarray] = {}
        for band in bands:
            var = find_var(ds, band)
            if var is not None:
                band_data[band] = to_time_y_x(ds[var].values).astype(np.float32)
                continue

            if band == "NDVI":
                red_var = find_var(ds, "B4")
                nir_var = find_var(ds, "B8")
                if red_var is None or nir_var is None:
                    raise ValueError(f"Nao achei NDVI nem B4/B8 em {image_path}. Variaveis: {list(ds.data_vars)}")
                red = to_time_y_x(ds[red_var].values).astype(np.float32)
                nir = to_time_y_x(ds[nir_var].values).astype(np.float32)
                band_data[band] = ((nir - red) / (nir + red + 1e-6)).astype(np.float32)
                continue

            raise ValueError(f"Nao achei banda {band} em {image_path}. Variaveis: {list(ds.data_vars)}")

        for band in bands:
            arr = np.nan_to_num(band_data[band], nan=0.0, posinf=0.0, neginf=0.0)
            if arr.shape[0] >= time_steps:
                arr = arr[:time_steps]
            else:
                arr = np.concatenate([arr, np.repeat(arr[-1:], time_steps - arr.shape[0], axis=0)], axis=0)
            band_data[band] = arr.astype(np.float32)

        channels = []
        if stack_order == "time_band":
            for time_idx in range(time_steps):
                for band in bands:
                    channels.append(band_data[band][time_idx])
        elif stack_order == "band_time":
            for band in bands:
                for time_idx in range(time_steps):
                    channels.append(band_data[band][time_idx])
        else:
            raise ValueError("--stack-order deve ser time_band ou band_time")

        x = np.stack(channels, axis=0).astype(np.float32)
        x = normalize_image_percentile(x)  # MESMA normalizacao do treino (percentil 2-98 por canal)
        transform = transform_from_xy(ds, width=x.shape[-1], height=x.shape[-2])
        crs = crs_from_dataset(ds)
        return x, band_data, transform, crs
    finally:
        ds.close()


def read_geotiff_stack(
    image_path: Path,
    bands: list[str],
    time_steps: int,
    stack_order: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], Affine, CRS | None]:
    with rasterio.open(image_path) as src:
        arr = src.read().astype(np.float32)
        transform = src.transform
        crs = src.crs

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    expected = len(bands) * time_steps
    if arr.shape[0] == expected:
        raw = arr
    elif arr.shape[0] >= len(bands):
        raw = np.concatenate([arr[: len(bands)]] * time_steps, axis=0).astype(np.float32)
    else:
        raise ValueError(f"GeoTIFF precisa ter {len(bands)} ou {expected} bandas. Recebido: {arr.shape[0]}")

    # band_data CRU (para o RGB do painel; stretch_rgb cuida do contraste)
    band_data: dict[str, np.ndarray] = {}
    if raw.shape[0] == expected:
        for band_idx, band in enumerate(bands):
            frames = []
            for time_idx in range(time_steps):
                src_idx = time_idx * len(bands) + band_idx if stack_order == "time_band" else band_idx * time_steps + time_idx
                frames.append(raw[src_idx])
            band_data[band] = np.stack(frames, axis=0)

    x = normalize_image_percentile(raw)  # MESMA normalizacao do treino (percentil 2-98 por canal)
    return x, band_data, transform, crs


def read_image_stack(
    image_path: Path,
    bands: list[str],
    time_steps: int,
    stack_order: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], Affine, CRS | None]:
    suffix = image_path.suffix.lower()
    if suffix == ".nc":
        return read_netcdf_stack(image_path, bands, time_steps, stack_order)
    if suffix in {".tif", ".tiff"}:
        return read_geotiff_stack(image_path, bands, time_steps, stack_order)
    raise ValueError(f"Formato nao suportado: {image_path}")


def build_model(architecture: str, encoder_name: str, in_channels: int, classes: int = 1) -> torch.nn.Module:
    arch = str(architecture).lower()
    if arch == "unet":
        return smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)
    if arch in {"unetplusplus", "unet++"}:
        return smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)
    if arch in {"deeplab", "deeplabv3plus", "deeplabv3+"}:
        return smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)
    if arch == "fpn":
        return smp.FPN(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)
    if arch == "pan":
        return smp.PAN(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)
    raise ValueError(f"Arquitetura nao suportada: {architecture}")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if sys.platform.startswith("win"):
        pathlib.PosixPath = pathlib.WindowsPath

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    architecture = cfg.get("architecture") or cfg.get("arch") or cfg.get("model") or "unet"
    encoder = cfg.get("encoder_name") or cfg.get("encoder") or cfg.get("backbone") or "resnet34"
    in_channels = int(cfg.get("in_channels", 30))
    classes = int(cfg.get("classes", 1))

    model = build_model(architecture, encoder, in_channels, classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def parse_weights(text: str, n_models: int) -> list[float]:
    weights = np.array([float(x.strip()) for x in text.split(",") if x.strip()], dtype=np.float32)
    if len(weights) != n_models:
        raise ValueError(f"--weights precisa ter {n_models} valores. Recebido: {len(weights)}")
    weights = weights / weights.sum()
    return weights.tolist()


def apply_tta(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "id":
        return x
    if mode == "h":
        return torch.flip(x, dims=[3])
    if mode == "v":
        return torch.flip(x, dims=[2])
    if mode == "hv":
        return torch.flip(x, dims=[2, 3])
    raise ValueError(mode)


def tta_modes(mode: str) -> list[str]:
    if mode == "none":
        return ["id"]
    if mode == "flips":
        return ["id", "h", "v", "hv"]
    raise ValueError("--tta-mode deve ser none ou flips")


def remove_small_components(mask: np.ndarray, min_area: int, connectivity: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)
    structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    labels, n = ndimage.label(mask.astype(bool), structure=structure)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8)


@torch.no_grad()
def predict_ensemble(
    x_np: np.ndarray,
    models: list[torch.nn.Module],
    weights: list[float],
    device: torch.device,
    threshold: float,
    min_area: int,
    connectivity: int,
    tta_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(x_np[None]).float().to(device)
    prob_sum = None

    for mode in tta_modes(tta_mode):
        xt = apply_tta(x, mode)
        p_mode = None
        for model, weight in zip(models, weights):
            logits = model(xt)
            if logits.shape[1] > 1:
                logits = logits[:, 0:1]
            p = torch.sigmoid(logits)
            p_mode = weight * p if p_mode is None else p_mode + weight * p
        p_mode = apply_tta(p_mode, mode)
        prob_sum = p_mode if prob_sum is None else prob_sum + p_mode

    prob = (prob_sum / len(tta_modes(tta_mode)))[0, 0].float().cpu().numpy()
    raw = (prob >= threshold).astype(np.uint8)
    return prob, remove_small_components(raw, min_area=min_area, connectivity=connectivity)


@torch.no_grad()
def predict_softmax3(x_np: np.ndarray, model: torch.nn.Module, device: torch.device, tta_mode: str) -> np.ndarray:
    """Modelo 3-classes (interior/borda/fundo) -> softmax media (com TTA por flips)."""
    x = torch.from_numpy(x_np[None]).float().to(device)
    prob_sum = None
    for mode in tta_modes(tta_mode):
        xt = apply_tta(x, mode)
        p = torch.softmax(model(xt), dim=1)
        p = apply_tta(p, mode)  # flips sao auto-inversos
        prob_sum = p if prob_sum is None else prob_sum + p
    return (prob_sum / len(tta_modes(tta_mode)))[0].float().cpu().numpy()  # (3,H,W)


def instances_from_probs(prob3: np.ndarray, seed_min: int, field_thr: float, min_area: int) -> np.ndarray:
    """Separa talhoes: interior conexo (a borda quebra a conexao) + Voronoi dentro do campo."""
    p_int, p_bnd, p_bg = prob3
    field = p_bg < field_thr
    interior = np.argmax(prob3, 0) == 0
    seeds, n = ndimage.label(interior, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(field.shape, np.int32)
    sizes = np.bincount(seeds.ravel())
    for s in np.where(sizes < seed_min)[0]:
        if s != 0:
            seeds[seeds == s] = 0
    if seeds.max() == 0:
        return np.zeros(field.shape, np.int32)
    idx = ndimage.distance_transform_edt(seeds == 0, return_indices=True)[1]
    grown = seeds[tuple(idx)]
    grown[~field] = 0
    out = np.zeros_like(grown); new = 1
    for i in [v for v in np.unique(grown) if v > 0]:
        m = grown == i
        if int(m.sum()) >= min_area:
            out[m] = new; new += 1
    return out.astype(np.int32)


def polygonize_instances(label_mask, transform, crs, min_area_px, simplify_tolerance):
    """Vetoriza uma mascara ROTULADA: um poligono por instancia (talhao).

    Area em PIXELS contada direto da mascara (robusto a CRS geografico, onde
    poly.area sai em graus^2); m2/ha so quando o CRS e projetado.
    """
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as shp_shape, mapping as shp_mapping

    lab = np.asarray(label_mask).astype(np.int32)
    pixel_footprint = abs(float(transform.a) * float(transform.e)) or 1.0  # area do pixel em unidades do CRS^2
    projected_m2 = _pixel_area(transform, crs)  # m2/pixel se projetado; None se geografico
    features, rows = [], []
    out_id = 1
    for geom, value in rio_shapes(lab, mask=lab > 0, transform=transform):
        if int(value) <= 0:
            continue
        poly = shp_shape(geom)
        if simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        area_px = float(poly.area / pixel_footprint)  # contagem de pixels (qualquer CRS)
        if area_px < min_area_px:
            continue
        area_m2 = float(poly.area) if projected_m2 else None
        props = {"candidate_id": out_id, "area_px": area_px, "area_m2": area_m2,
                 "area_ha": None if area_m2 is None else area_m2 / 10000.0}
        features.append({"type": "Feature", "properties": props, "geometry": shp_mapping(poly)})
        rows.append(props); out_id += 1
    return features, rows


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        band = rgb[..., channel].astype(np.float32)
        lo, hi = np.percentile(band, [2, 98])
        out[..., channel] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1) if hi > lo else np.clip(band, 0, 1)
    return out


def make_rgb(band_data: dict[str, np.ndarray]) -> np.ndarray | None:
    if not all(band in band_data for band in ["B2", "B3", "B4"]):
        return None
    rgb = np.stack(
        [
            np.median(band_data["B4"], axis=0),
            np.median(band_data["B3"], axis=0),
            np.median(band_data["B2"], axis=0),
        ],
        axis=-1,
    )
    return stretch_rgb(rgb)


def write_raster(path: Path, arr: np.ndarray, transform: Affine, crs: CRS | None, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def save_panel(rgb: np.ndarray | None, prob: np.ndarray, pred: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 3 if rgb is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(cols * 4.2, 4.0))
    axes = np.asarray(axes).reshape(-1)
    idx = 0
    if rgb is not None:
        axes[idx].imshow(rgb)
        axes[idx].set_title("Sentinel-2 RGB")
        axes[idx].axis("off")
        idx += 1
    axes[idx].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    axes[idx].set_title("Probabilidade")
    axes[idx].axis("off")
    idx += 1
    axes[idx].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[idx].set_title("Talhoes candidatos")
    axes[idx].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        col = "image_path" if "image_path" in df.columns else df.columns[0]
        return [Path(value) for value in df[col].dropna().astype(str).tolist()]
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob("*") if p.suffix.lower() in {".nc", ".tif", ".tiff"}])


def write_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("candidate_id,area_px,area_m2,area_ha\n", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline end-to-end: Sentinel-2 gratuito/local -> mascara -> talhoes candidatos."
    )
    parser.add_argument("--input", type=Path, required=True, help="Arquivo .nc/.tif, pasta ou CSV com coluna image_path.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/sentinel2/end_to_end_fields")
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--bands", type=str, default="B2,B3,B4,B8,NDVI")
    parser.add_argument("--time-steps", type=int, default=6)
    parser.add_argument("--stack-order", choices=["time_band", "band_time"], default="band_time")
    parser.add_argument("--threshold", type=float, default=0.49)
    parser.add_argument("--min-area", type=int, default=90)
    parser.add_argument("--connectivity", type=int, default=8)
    parser.add_argument("--tta-mode", choices=["none", "flips"], default="flips")
    parser.add_argument("--simplify-tolerance", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--instance-checkpoint", type=Path, default=None,
                        help="Modelo 3-classes (interior/borda/fundo) -> talhoes SEPARADOS (1 poligono por talhao).")
    parser.add_argument("--seed-min", type=int, default=250, help="Tamanho min. da semente de interior (modo instancia).")
    parser.add_argument("--field-thr", type=float, default=0.5, help="Limiar de campo = 1 - P(fundo) (modo instancia).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rasters_dir = args.output_dir / "rasters"
    vectors_dir = args.output_dir / "vectors"
    panels_dir = args.output_dir / "panels"
    rasters_dir.mkdir(exist_ok=True)
    vectors_dir.mkdir(exist_ok=True)
    panels_dir.mkdir(exist_ok=True)

    image_paths = collect_inputs(args.input)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem .nc/.tif encontrada em {args.input}")

    instance_mode = args.instance_checkpoint is not None
    bands = [x.strip() for x in args.bands.split(",") if x.strip()]
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    models: list[torch.nn.Module] = []
    weights: list[float] = []
    inst_model = None
    if instance_mode:
        if not args.instance_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint de instancia nao encontrado: {args.instance_checkpoint}")
        print(f"Modo INSTANCIA (talhoes separados). Carregando {args.instance_checkpoint.name}...")
        inst_model, inst_cfg = load_model(args.instance_checkpoint, device)
        print(f"- 3-classes: {inst_cfg.get('architecture', 'unet')}/{inst_cfg.get('encoder_name', 'resnet34')} "
              f"classes={inst_cfg.get('classes')} | seed_min={args.seed_min} field_thr={args.field_thr} min_area={args.min_area}")
    else:
        missing = [p for p in args.checkpoints if not p.exists()]
        if missing:
            raise FileNotFoundError("Checkpoints nao encontrados:\n" + "\n".join(str(p) for p in missing))
        weights = parse_weights(args.weights, len(args.checkpoints))
        print("Modo ENSEMBLE (semantico). Carregando modelos...")
        for ckpt in args.checkpoints:
            model, cfg = load_model(ckpt, device)
            models.append(model)
            print(f"- {ckpt.name}: {cfg.get('architecture', cfg.get('model', 'unet'))} / {cfg.get('encoder_name', cfg.get('encoder', 'resnet34'))}")

    index_rows: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, Any]] = []

    for image_path in tqdm(image_paths, desc="Inferindo talhoes"):
        name = safe_name(image_path)
        try:
            x_np, band_data, transform, crs = read_image_stack(image_path, bands, args.time_steps, args.stack_order)

            prob_path = rasters_dir / f"{name}_probability.tif"
            mask_path = rasters_dir / f"{name}_candidate_mask.tif"
            geojson_path = vectors_dir / f"{name}_candidate_fields.geojson"
            csv_path = vectors_dir / f"{name}_candidate_fields.csv"
            panel_path = panels_dir / f"{name}_panel.png"

            if instance_mode:
                prob3 = predict_softmax3(x_np, inst_model, device, args.tta_mode)
                inst = instances_from_probs(prob3, args.seed_min, args.field_thr, args.min_area)
                prob = (prob3[0] + prob3[1]).astype(np.float32)  # P(campo) = interior + borda
                pred = (inst > 0).astype(np.uint8)
                write_raster(prob_path, prob, transform, crs, "float32")
                write_raster(mask_path, inst.astype(np.int32), transform, crs, "int32")  # rotulo por talhao
                features, candidate_rows = polygonize_instances(
                    inst, transform=transform, crs=crs,
                    min_area_px=args.min_area, simplify_tolerance=args.simplify_tolerance,
                )
            else:
                prob, pred = predict_ensemble(
                    x_np=x_np, models=models, weights=weights, device=device,
                    threshold=args.threshold, min_area=args.min_area,
                    connectivity=args.connectivity, tta_mode=args.tta_mode,
                )
                write_raster(prob_path, prob, transform, crs, "float32")
                write_raster(mask_path, pred, transform, crs, "uint8")
                features, candidate_rows = polygonize_mask(
                    pred, transform=transform, crs=crs,
                    min_area_px=args.min_area, simplify_tolerance=args.simplify_tolerance,
                )
            write_geojson(features, geojson_path, crs=crs)

            for row in candidate_rows:
                row = dict(row)
                row["image_path"] = str(image_path)
                row["candidate_geojson"] = str(geojson_path)
                all_candidate_rows.append(row)
            write_rows_csv(candidate_rows, csv_path)

            save_panel(make_rgb(band_data), prob, pred, panel_path, title=name)

            index_rows.append(
                {
                    "image_path": str(image_path),
                    "probability_tif": str(prob_path),
                    "candidate_mask_tif": str(mask_path),
                    "candidate_geojson": str(geojson_path),
                    "candidate_csv": str(csv_path),
                    "panel_png": str(panel_path),
                    "num_candidates": len(candidate_rows),
                    "threshold": args.threshold,
                    "min_area_px": args.min_area,
                }
            )
        except Exception as exc:
            print(f"[ERRO] {image_path}: {exc}")
            index_rows.append({"image_path": str(image_path), "error": str(exc)})

    pd.DataFrame(index_rows).to_csv(args.output_dir / "end_to_end_index.csv", index=False)
    if all_candidate_rows:
        pd.DataFrame(all_candidate_rows).to_csv(args.output_dir / "all_candidate_fields.csv", index=False)
    else:
        write_rows_csv([], args.output_dir / "all_candidate_fields.csv")

    print("\nFinalizado.")
    print(f"Indice: {args.output_dir / 'end_to_end_index.csv'}")
    print(f"Talhoes candidatos: {args.output_dir / 'all_candidate_fields.csv'}")
    print(f"Rasters: {rasters_dir}")
    print(f"Vetores: {vectors_dir}")
    print(f"Paineis: {panels_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from tqdm import tqdm

import matplotlib.pyplot as plt

try:
    import xarray as xr
except ImportError as exc:
    raise RuntimeError("Este script precisa de xarray. Instale com: pip install xarray netCDF4") from exc

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise RuntimeError("Este script precisa de segmentation_models_pytorch instalado no ambiente da VM.") from exc


BAND_ALIASES = {
    "B2": ["B2", "B02", "b2", "b02", "blue", "BLUE"],
    "B3": ["B3", "B03", "b3", "b03", "green", "GREEN"],
    "B4": ["B4", "B04", "b4", "b04", "red", "RED"],
    "B8": ["B8", "B08", "b8", "b08", "nir", "NIR"],
    "NDVI": ["NDVI", "ndvi"],
}


def build_model(architecture: str, encoder_name: str, in_channels: int, classes: int = 1) -> nn.Module:
    arch = str(architecture).lower()

    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )

    if arch in {"unetplusplus", "unet++"}:
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )

    if arch in {"deeplab", "deeplabv3plus", "deeplabv3+"}:
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )

    if arch == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )

    raise ValueError(f"Arquitetura não suportada: {architecture}")


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    architecture = (
        cfg.get("architecture")
        or cfg.get("arch")
        or cfg.get("model")
        or "unet"
    )

    encoder = (
        cfg.get("encoder_name")
        or cfg.get("encoder")
        or cfg.get("backbone")
        or "resnet34"
    )

    in_channels = int(cfg.get("in_channels", 30))
    classes = int(cfg.get("classes", 1))

    model = build_model(
        architecture=architecture,
        encoder_name=encoder,
        in_channels=in_channels,
        classes=classes,
    ).to(device)

    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state)
    model.eval()

    return model, cfg


def find_var(ds: xr.Dataset, canonical: str):
    for name in BAND_ALIASES.get(canonical, [canonical]):
        if name in ds.data_vars:
            return name
    return None


def as_time_y_x(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    # Remove dimensões unitárias.
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr[None, :, :]

    if arr.ndim != 3:
        raise ValueError(f"Esperava array 2D ou 3D, recebi shape={arr.shape}")

    # Heurística: dimensão temporal costuma ser pequena.
    shape = arr.shape
    small_axes = [i for i, s in enumerate(shape) if s <= 20]

    if 0 in small_axes:
        return arr

    if 2 in small_axes:
        return np.moveaxis(arr, 2, 0)

    if 1 in small_axes:
        return np.moveaxis(arr, 1, 0)

    # fallback: assume já time,y,x
    return arr


def prepare_reflectance(arr: np.ndarray, band: str) -> np.ndarray:
    arr = arr.astype(np.float32)

    if band != "NDVI":
        # Sentinel-2 pode estar em reflectância 0..1, 0..10000 ou 0..255.
        maxv = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
        if maxv > 2000:
            arr = arr / 10000.0
        elif maxv > 5:
            arr = arr / 255.0

        arr = np.clip(arr, 0.0, 1.0)
    else:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1.0, 1.0)

    return arr.astype(np.float32)


def read_sentinel2_stack(
    image_path: Path,
    bands: list[str],
    time_steps: int,
    stack_order: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ds = xr.open_dataset(image_path)

    band_arrays = {}

    for band in bands:
        var = find_var(ds, band)

        if var is not None:
            arr = as_time_y_x(ds[var].values)
            arr = prepare_reflectance(arr, band)
            band_arrays[band] = arr
            continue

        if band == "NDVI":
            b4_var = find_var(ds, "B4")
            b8_var = find_var(ds, "B8")
            if b4_var is None or b8_var is None:
                raise ValueError(f"Não encontrei NDVI nem B4/B8 para calcular em {image_path}")

            red = prepare_reflectance(as_time_y_x(ds[b4_var].values), "B4")
            nir = prepare_reflectance(as_time_y_x(ds[b8_var].values), "B8")
            band_arrays["NDVI"] = (nir - red) / (nir + red + 1e-6)
            continue

        raise ValueError(f"Não encontrei banda {band} em {image_path}. Variáveis: {list(ds.data_vars)}")

    # Ajusta time_steps.
    for band in bands:
        arr = band_arrays[band]
        if arr.shape[0] >= time_steps:
            arr = arr[:time_steps]
        else:
            pad = np.repeat(arr[-1:], time_steps - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        band_arrays[band] = arr

    channels = []

    if stack_order == "time_band":
        for t in range(time_steps):
            for band in bands:
                channels.append(band_arrays[band][t])
    elif stack_order == "band_time":
        for band in bands:
            for t in range(time_steps):
                channels.append(band_arrays[band][t])
    else:
        raise ValueError("--stack-order deve ser time_band ou band_time")

    x = np.stack(channels, axis=0).astype(np.float32)
    return x, band_arrays


def image_to_mask_path(image_path: Path) -> Path:
    s = str(image_path)
    s = s.replace("/images/", "/masks/")

    # Tenta substituições comuns.
    s = s.replace("_S2_10m_256.nc", "_label_10m_256.nc")
    s = s.replace("_s2_10m_256.nc", "_label_10m_256.nc")

    return Path(s)


def read_mask(mask_path: Path) -> np.ndarray:
    ds = xr.open_dataset(mask_path)

    # Escolhe primeira variável 2D/3D plausível.
    candidates = list(ds.data_vars)
    if not candidates:
        raise ValueError(f"Máscara sem variáveis: {mask_path}")

    preferred = [
        "mask", "label", "labels", "field", "field_boundary",
        "boundary", "target", "y", "data"
    ]

    var = None
    for p in preferred:
        if p in ds.data_vars:
            var = p
            break

    if var is None:
        var = candidates[0]

    arr = np.asarray(ds[var].values)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        # Se tiver canal/time, pega o primeiro plano.
        small_axes = [i for i, s in enumerate(arr.shape) if s <= 20]
        if 0 in small_axes:
            arr = arr[0]
        elif 2 in small_axes:
            arr = arr[:, :, 0]
        else:
            arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"Máscara com shape inesperado {arr.shape} em {mask_path}")

    return (arr > 0).astype(np.uint8)


def tta_list(mode: str) -> list[str]:
    if mode == "none":
        return ["id"]
    if mode == "flips":
        return ["id", "h", "v", "hv"]
    raise ValueError("--tta-mode suportado: none ou flips")


def apply_tta(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "id":
        return x
    if name == "h":
        return torch.flip(x, dims=[3])
    if name == "v":
        return torch.flip(x, dims=[2])
    if name == "hv":
        return torch.flip(x, dims=[2, 3])
    raise ValueError(name)


def invert_tta(y: torch.Tensor, name: str) -> torch.Tensor:
    return apply_tta(y, name)


def remove_small_components(mask: np.ndarray, min_area: int, connectivity: int = 8) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)

    structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=np.uint8,
    )

    labels, n = ndimage.label(mask.astype(bool), structure=structure)
    if n == 0:
        return mask.astype(np.uint8)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8)


@torch.no_grad()
def predict_mask(
    x_np: np.ndarray,
    models: list[nn.Module],
    weights: list[float],
    device: torch.device,
    threshold: float,
    min_area: int,
    tta_mode: str,
    connectivity: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(x_np[None]).float().to(device)

    probs_total = None
    ttas = tta_list(tta_mode)

    for tta in ttas:
        xt = apply_tta(x, tta)
        probs_t = None

        for model, w in zip(models, weights):
            logits = model(xt)

            if logits.shape[1] > 1:
                logits = logits[:, 0:1]

            p = torch.sigmoid(logits)
            probs_t = w * p if probs_t is None else probs_t + w * p

        probs_t = invert_tta(probs_t, tta)
        probs_total = probs_t if probs_total is None else probs_total + probs_t

    prob = (probs_total / len(ttas))[0, 0].float().cpu().numpy()
    raw = (prob >= threshold).astype(np.uint8)
    pred = remove_small_components(raw, min_area=min_area, connectivity=connectivity)

    return prob, pred


def percentile_stretch(img: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)

    for c in range(img.shape[-1]):
        band = img[..., c]
        lo, hi = np.nanpercentile(band, [p_low, p_high])
        if hi <= lo:
            out[..., c] = np.clip(band, 0, 1)
        else:
            out[..., c] = np.clip((band - lo) / (hi - lo), 0, 1)

    return out


def make_rgb(band_arrays: dict[str, np.ndarray], time_index: str = "median") -> np.ndarray:
    red = band_arrays["B4"]
    green = band_arrays["B3"]
    blue = band_arrays["B2"]

    if time_index == "median":
        r = np.median(red, axis=0)
        g = np.median(green, axis=0)
        b = np.median(blue, axis=0)
    else:
        idx = int(time_index)
        r = red[idx]
        g = green[idx]
        b = blue[idx]

    rgb = np.stack([r, g, b], axis=-1)
    return percentile_stretch(rgb)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.42) -> np.ndarray:
    out = rgb.copy()
    m = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * color_arr
    return np.clip(out, 0, 1)


def error_map_rgb(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    h, w = target.shape
    out = np.ones((h, w, 3), dtype=np.float32)

    tp = (target == 1) & (pred == 1)
    fp = (target == 0) & (pred == 1)
    fn = (target == 1) & (pred == 0)

    # TP verde, FP vermelho, FN azul.
    out[tp] = np.array([0.15, 0.70, 0.25])
    out[fp] = np.array([0.90, 0.15, 0.10])
    out[fn] = np.array([0.10, 0.25, 0.90])

    return out


def safe_name(path: str) -> str:
    name = Path(str(path)).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def plot_case(
    row: pd.Series,
    rgb: np.ndarray,
    target: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    title_prefix: str,
):
    dice = float(row.get("dice", np.nan))
    iou = float(row.get("iou", np.nan))
    target_ratio = float(row.get("target_ratio", np.nan))
    pred_ratio = float(row.get("pred_ratio", np.nan))
    group = str(row.get("selection_group", row.get("overview_group", "")))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6))

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB\n(mediana temporal)", fontsize=10)

    axes[1].imshow(overlay_mask(rgb, target, (0.1, 0.8, 0.2)))
    axes[1].set_title("Ground truth\nverde", fontsize=10)

    axes[2].imshow(overlay_mask(rgb, pred, (1.0, 0.15, 0.1)))
    axes[2].set_title("Predição\nvermelho", fontsize=10)

    axes[3].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("Mapa de probabilidade", fontsize=10)

    axes[4].imshow(error_map_rgb(target, pred))
    axes[4].set_title("Erro\nTP verde | FP vermelho | FN azul", fontsize=10)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{title_prefix} | {safe_name(row['image_path'])}\n"
        f"Grupo: {group} | Dice={dice:.3f} | IoU={iou:.3f} | "
        f"target_ratio={target_ratio:.3f} | pred_ratio={pred_ratio:.3f}",
        fontsize=12,
        y=1.03,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_contact_sheet(image_paths: list[Path], output_path: Path, cols: int = 2):
    if not image_paths:
        return

    import matplotlib.image as mpimg

    imgs = [mpimg.imread(p) for p in image_paths]
    rows = math.ceil(len(imgs) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8.5, rows * 2.6))
    axes = np.asarray(axes).reshape(-1)

    for ax, img, path in zip(axes, imgs, image_paths):
        ax.imshow(img)
        ax.set_title(path.stem, fontsize=8)
        ax.axis("off")

    for ax in axes[len(imgs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Gera painéis qualitativos Sentinel-2 para apresentação.")
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", type=str, required=True)

    parser.add_argument("--bands", type=str, default="B2,B3,B4,B8,NDVI")
    parser.add_argument("--time-steps", type=int, default=6)
    parser.add_argument("--stack-order", type=str, default="time_band", choices=["time_band", "band_time"])
    parser.add_argument("--threshold", type=float, default=0.49)
    parser.add_argument("--min-area", type=int, default=90)
    parser.add_argument("--connectivity", type=int, default=8)
    parser.add_argument("--tta-mode", type=str, default="flips", choices=["none", "flips"])

    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--groups", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rgb-time-index", type=str, default="median")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = args.output_dir / "panels"
    panels_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.selected_csv)

    if args.groups.strip():
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        df = df[df["selection_group"].isin(groups)].copy()

    df = df.head(args.max_cases).copy()

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    weights = np.array(weights, dtype=np.float32)
    weights = (weights / weights.sum()).tolist()

    if len(weights) != len(args.checkpoints):
        raise ValueError(f"Número de pesos ({len(weights)}) diferente do número de checkpoints ({len(args.checkpoints)}).")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    print("Carregando modelos...")
    models = []
    for ckpt in args.checkpoints:
        model, cfg = load_model(ckpt, device)
        models.append(model)
        print(f"- {ckpt} | cfg={cfg}")

    saved_panels = []
    generated_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Gerando painéis"):
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"]) if "mask_path" in row and pd.notna(row["mask_path"]) else image_to_mask_path(image_path)

        try:
            x_np, band_arrays = read_sentinel2_stack(
                image_path=image_path,
                bands=bands,
                time_steps=args.time_steps,
                stack_order=args.stack_order,
            )
            target = read_mask(mask_path)
            rgb = make_rgb(band_arrays, time_index=args.rgb_time_index)

            prob, pred = predict_mask(
                x_np=x_np,
                models=models,
                weights=weights,
                device=device,
                threshold=args.threshold,
                min_area=args.min_area,
                tta_mode=args.tta_mode,
                connectivity=args.connectivity,
            )

            panel_name = f"{len(saved_panels)+1:02d}_{str(row.get('selection_group', 'case'))}_{safe_name(row['image_path'])}.png"
            panel_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", panel_name)
            panel_path = panels_dir / panel_name

            plot_case(
                row=row,
                rgb=rgb,
                target=target,
                prob=prob,
                pred=pred,
                output_path=panel_path,
                title_prefix="Sentinel-2 operacional v1",
            )

            saved_panels.append(panel_path)

            generated = row.to_dict()
            generated["panel_path"] = str(panel_path)
            generated_rows.append(generated)

        except Exception as exc:
            print(f"[ERRO] Falha ao gerar painel para {image_path}: {exc}")

    pd.DataFrame(generated_rows).to_csv(args.output_dir / "generated_panels_index.csv", index=False)

    create_contact_sheet(
        saved_panels,
        args.output_dir / "boss_overview_contact_sheet.png",
        cols=2,
    )

    print("\nPainéis salvos em:", panels_dir)
    print("Contact sheet:", args.output_dir / "boss_overview_contact_sheet.png")
    print("Índice:", args.output_dir / "generated_panels_index.csv")


if __name__ == "__main__":
    main()

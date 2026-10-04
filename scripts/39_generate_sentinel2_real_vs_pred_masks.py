from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm

import matplotlib.pyplot as plt

try:
    import xarray as xr
except ImportError as exc:
    raise RuntimeError("Faltou xarray. Instale com: pip install xarray netCDF4 h5netcdf") from exc

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise RuntimeError("Faltou segmentation_models_pytorch no ambiente.") from exc


BAND_ALIASES = {
    "B2": ["B2", "B02", "b2", "b02", "blue", "BLUE"],
    "B3": ["B3", "B03", "b3", "b03", "green", "GREEN"],
    "B4": ["B4", "B04", "b4", "b04", "red", "RED"],
    "B8": ["B8", "B08", "b8", "b08", "nir", "NIR"],
    "NDVI": ["NDVI", "ndvi"],
}


def safe_name(value: str) -> str:
    value = Path(str(value)).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def find_var(ds: xr.Dataset, canonical: str) -> str | None:
    for candidate in BAND_ALIASES.get(canonical, [canonical]):
        if candidate in ds.data_vars:
            return candidate
    return None


def to_time_y_x(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr[None, :, :]

    if arr.ndim != 3:
        raise ValueError(f"Array precisa ser 2D ou 3D; recebido shape={arr.shape}")

    shape = arr.shape
    small_axes = [i for i, s in enumerate(shape) if s <= 20]

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


def read_sentinel2_stack(
    image_path: Path,
    bands: list[str],
    time_steps: int,
    stack_order: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ds = xr.open_dataset(image_path)

    band_data: dict[str, np.ndarray] = {}

    for band in bands:
        var = find_var(ds, band)

        if var is not None:
            arr = normalize_band(to_time_y_x(ds[var].values), band)
            band_data[band] = arr
            continue

        if band == "NDVI":
            b4_var = find_var(ds, "B4")
            b8_var = find_var(ds, "B8")

            if b4_var is None or b8_var is None:
                raise ValueError(f"Não achei NDVI nem B4/B8 em {image_path}. Variáveis: {list(ds.data_vars)}")

            red = normalize_band(to_time_y_x(ds[b4_var].values), "B4")
            nir = normalize_band(to_time_y_x(ds[b8_var].values), "B8")
            band_data["NDVI"] = ((nir - red) / (nir + red + 1e-6)).astype(np.float32)
            continue

        raise ValueError(f"Não achei banda {band} em {image_path}. Variáveis: {list(ds.data_vars)}")

    for band in bands:
        arr = band_data[band]

        if arr.shape[0] >= time_steps:
            arr = arr[:time_steps]
        else:
            pad = np.repeat(arr[-1:], time_steps - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)

        band_data[band] = arr.astype(np.float32)

    channels = []

    if stack_order == "time_band":
        for t in range(time_steps):
            for band in bands:
                channels.append(band_data[band][t])
    elif stack_order == "band_time":
        for band in bands:
            for t in range(time_steps):
                channels.append(band_data[band][t])
    else:
        raise ValueError("--stack-order deve ser time_band ou band_time")

    x = np.stack(channels, axis=0).astype(np.float32)
    return x, band_data


def guess_mask_path(image_path: Path) -> Path:
    s = str(image_path)
    s = s.replace("/images/", "/masks/")

    replacements = [
        ("_S2_10m_256.nc", "_label_10m_256.nc"),
        ("_s2_10m_256.nc", "_label_10m_256.nc"),
        ("_S2_", "_label_"),
        ("_s2_", "_label_"),
    ]

    for old, new in replacements:
        candidate = Path(s.replace(old, new))
        if candidate.exists():
            return candidate

    return Path(s)


def read_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Máscara não encontrada: {mask_path}")

    ds = xr.open_dataset(mask_path)

    preferred = [
        "mask",
        "label",
        "labels",
        "field",
        "fields",
        "field_boundary",
        "boundary",
        "target",
        "y",
        "data",
    ]

    var = None

    for name in preferred:
        if name in ds.data_vars:
            var = name
            break

    if var is None:
        for name in ds.data_vars:
            if np.asarray(ds[name].values).ndim >= 2:
                var = name
                break

    if var is None:
        raise ValueError(f"Não achei variável de máscara em {mask_path}. Variáveis: {list(ds.data_vars)}")

    arr = np.asarray(ds[var].values)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
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


def build_model(architecture: str, encoder_name: str, in_channels: int, classes: int):
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


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    architecture = cfg.get("architecture") or cfg.get("arch") or cfg.get("model") or "unet"
    encoder = cfg.get("encoder_name") or cfg.get("encoder") or cfg.get("backbone") or "resnet34"
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

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.uint8)
    else:
        structure = np.array(
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
    modes = tta_modes(tta_mode)

    for mode in modes:
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

    prob = (prob_sum / len(modes))[0, 0].float().cpu().numpy()
    raw_pred = (prob >= threshold).astype(np.uint8)
    pred = remove_small_components(raw_pred, min_area=min_area, connectivity=connectivity)

    return prob, pred


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb)

    for c in range(3):
        band = rgb[..., c]
        lo, hi = np.percentile(band, [2, 98])

        if hi <= lo:
            out[..., c] = np.clip(band, 0, 1)
        else:
            out[..., c] = np.clip((band - lo) / (hi - lo), 0, 1)

    return out


def make_rgb(band_data: dict[str, np.ndarray], time_view: str) -> np.ndarray:
    if time_view == "median":
        r = np.median(band_data["B4"], axis=0)
        g = np.median(band_data["B3"], axis=0)
        b = np.median(band_data["B2"], axis=0)
    else:
        idx = int(time_view)
        r = band_data["B4"][idx]
        g = band_data["B3"][idx]
        b = band_data["B2"][idx]

    return stretch_rgb(np.stack([r, g, b], axis=-1))


def comparison_rgb(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    h, w = target.shape
    out = np.ones((h, w, 3), dtype=np.float32)

    tp = (target == 1) & (pred == 1)
    fp = (target == 0) & (pred == 1)
    fn = (target == 1) & (pred == 0)

    out[tp] = np.array([0.05, 0.65, 0.15])
    out[fp] = np.array([0.90, 0.10, 0.05])
    out[fn] = np.array([0.10, 0.20, 0.90])

    return out


def calc_metrics(target: np.ndarray, pred: np.ndarray) -> dict:
    target = target.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(target, pred).sum()
    fp = np.logical_and(~target, pred).sum()
    fn = np.logical_and(target, ~pred).sum()

    eps = 1e-7
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice_recomputed": float(dice),
        "iou_recomputed": float(iou),
        "precision_recomputed": float(precision),
        "recall_recomputed": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "target_ratio_recomputed": float(target.mean()),
        "pred_ratio_recomputed": float(pred.mean()),
    }


def plot_panel(
    row: pd.Series,
    rgb: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    output_path: Path,
):
    metrics = calc_metrics(target, pred)

    dice = metrics["dice_recomputed"]
    iou = metrics["iou_recomputed"]
    tr = metrics["target_ratio_recomputed"]
    pr = metrics["pred_ratio_recomputed"]

    group = str(row.get("selection_group", row.get("overview_group", "")))
    image_name = safe_name(row["image_path"])

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.3))

    axes[0].imshow(rgb)
    axes[0].set_title("Imagem Sentinel-2\nRGB mediana", fontsize=10)

    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Máscara real\nGround truth", fontsize=10)

    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Máscara criada\nModelo", fontsize=10)

    axes[3].imshow(comparison_rgb(target, pred))
    axes[3].set_title("Comparação\nverde=acerto | vermelho=extra | azul=faltou", fontsize=9)

    axes[4].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    axes[4].set_title("Probabilidade\nantes do threshold", fontsize=10)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{image_name} | {group}\n"
        f"Dice={dice:.3f} | IoU={iou:.3f} | target_ratio={tr:.3f} | pred_ratio={pr:.3f}",
        fontsize=12,
        y=1.03,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return metrics


def make_contact_sheet(panel_paths: list[Path], output_path: Path, cols: int = 2):
    if not panel_paths:
        return

    import matplotlib.image as mpimg

    imgs = [mpimg.imread(p) for p in panel_paths]
    rows = math.ceil(len(imgs) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 9.5, rows * 2.8))
    axes = np.asarray(axes).reshape(-1)

    for ax, img, path in zip(axes, imgs, panel_paths):
        ax.imshow(img)
        ax.set_title(path.stem, fontsize=7)
        ax.axis("off")

    for ax in axes[len(imgs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Gera painéis comparando máscara real vs máscara criada.")
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
    parser.add_argument("--rgb-time", type=str, default="median")

    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--groups", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels_dir = args.output_dir / "mask_comparison_panels"
    masks_dir = args.output_dir / "binary_masks"
    panels_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.selected_csv)

    if args.groups.strip():
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        df = df[df["selection_group"].isin(groups)].copy()

    df = df.head(args.max_cases).copy()

    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    weights = np.array(weights, dtype=np.float32)
    weights = (weights / weights.sum()).tolist()

    if len(weights) != len(args.checkpoints):
        raise ValueError(f"Foram passados {len(weights)} pesos para {len(args.checkpoints)} checkpoints.")

    bands = [x.strip() for x in args.bands.split(",") if x.strip()]
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print("Carregando modelos...")
    models = []

    for ckpt_path in args.checkpoints:
        model, cfg = load_model(ckpt_path, device)
        models.append(model)
        print(f"- {ckpt_path}")
        print(f"  config: {cfg}")

    generated = []
    panel_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Gerando comparações"):
        image_path = Path(row["image_path"])

        if "mask_path" in row and pd.notna(row["mask_path"]):
            mask_path = Path(str(row["mask_path"]))
        else:
            mask_path = guess_mask_path(image_path)

        try:
            x_np, band_data = read_sentinel2_stack(
                image_path=image_path,
                bands=bands,
                time_steps=args.time_steps,
                stack_order=args.stack_order,
            )

            target = read_mask(mask_path)

            prob, pred = predict_ensemble(
                x_np=x_np,
                models=models,
                weights=weights,
                device=device,
                threshold=args.threshold,
                min_area=args.min_area,
                connectivity=args.connectivity,
                tta_mode=args.tta_mode,
            )

            rgb = make_rgb(band_data, time_view=args.rgb_time)

            name = f"{idx + 1:03d}_{str(row.get('selection_group', 'case'))}_{safe_name(row['image_path'])}"
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

            panel_path = panels_dir / f"{name}.png"
            metrics = plot_panel(
                row=row,
                rgb=rgb,
                target=target,
                pred=pred,
                prob=prob,
                output_path=panel_path,
            )

            plt.imsave(masks_dir / f"{name}_real_mask.png", target, cmap="gray", vmin=0, vmax=1)
            plt.imsave(masks_dir / f"{name}_pred_mask.png", pred, cmap="gray", vmin=0, vmax=1)
            plt.imsave(masks_dir / f"{name}_comparison.png", comparison_rgb(target, pred))

            panel_paths.append(panel_path)

            rec = row.to_dict()
            rec.update(metrics)
            rec["panel_path"] = str(panel_path)
            rec["real_mask_png"] = str(masks_dir / f"{name}_real_mask.png")
            rec["pred_mask_png"] = str(masks_dir / f"{name}_pred_mask.png")
            rec["comparison_png"] = str(masks_dir / f"{name}_comparison.png")
            generated.append(rec)

        except Exception as exc:
            print(f"[ERRO] {image_path}: {exc}")

    pd.DataFrame(generated).to_csv(args.output_dir / "generated_mask_comparison_index.csv", index=False)

    make_contact_sheet(
        panel_paths=panel_paths,
        output_path=args.output_dir / "boss_mask_comparison_contact_sheet.png",
        cols=2,
    )

    print("\nFinalizado.")
    print("Painéis:", panels_dir)
    print("Máscaras binárias:", masks_dir)
    print("Prancha geral:", args.output_dir / "boss_mask_comparison_contact_sheet.png")
    print("Índice:", args.output_dir / "generated_mask_comparison_index.csv")


if __name__ == "__main__":
    main()

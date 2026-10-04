from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def read_binary_png(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0

    # plt.imsave com cmap gray pode gerar máscara branca=1, preta=0.
    # Binariza de forma robusta.
    return (arr >= 0.5).astype(np.uint8)


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return (mask ^ eroded).astype(np.uint8)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)

    struct = ndimage.iterate_structure(np.ones((3, 3), dtype=bool), radius)
    return ndimage.binary_dilation(mask.astype(bool), structure=struct)


def make_boundary_rgb(real: np.ndarray, pred: np.ndarray, tolerance: int = 1) -> np.ndarray:
    """
    Comparação de bordas:
      verde = borda predita próxima da borda real
      vermelho = borda predita sem correspondência
      azul = borda real perdida
    """
    br = boundary(real).astype(bool)
    bp = boundary(pred).astype(bool)

    br_tol = dilate(br, tolerance)
    bp_tol = dilate(bp, tolerance)

    matched_pred = bp & br_tol
    extra_pred = bp & (~br_tol)
    missed_real = br & (~bp_tol)

    h, w = real.shape
    out = np.ones((h, w, 3), dtype=np.float32)

    # fundo levemente cinza onde existe máscara real ou predita
    union = real.astype(bool) | pred.astype(bool)
    out[union] = np.array([0.93, 0.93, 0.93])

    out[matched_pred] = np.array([0.05, 0.65, 0.15])  # verde
    out[extra_pred] = np.array([0.90, 0.10, 0.05])    # vermelho
    out[missed_real] = np.array([0.10, 0.20, 0.90])   # azul

    return out


def make_filled_rgb(real: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Comparação cheia:
      verde = interseção correta
      vermelho = falso positivo
      azul = falso negativo
      branco = fundo correto
    """
    real_b = real.astype(bool)
    pred_b = pred.astype(bool)

    h, w = real.shape
    out = np.ones((h, w, 3), dtype=np.float32)

    tp = real_b & pred_b
    fp = (~real_b) & pred_b
    fn = real_b & (~pred_b)

    out[tp] = np.array([0.05, 0.65, 0.15])
    out[fp] = np.array([0.90, 0.10, 0.05])
    out[fn] = np.array([0.10, 0.20, 0.90])

    return out


def dice_iou(real: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    r = real.astype(bool)
    p = pred.astype(bool)

    tp = np.logical_and(r, p).sum()
    fp = np.logical_and(~r, p).sum()
    fn = np.logical_and(r, ~p).sum()

    eps = 1e-7
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return float(dice), float(iou)


def find_pairs(masks_dir: Path) -> list[tuple[Path, Path]]:
    real_files = sorted(masks_dir.glob("*_real.png"))
    if not real_files:
        real_files = sorted(masks_dir.glob("*_real_mask.png"))

    pairs = []

    for real in real_files:
        pred_candidates = [
            Path(str(real).replace("_real_mask.png", "_pred_mask.png")),
            Path(str(real).replace("_real.png", "_pred.png")),
        ]
        pred = next((p for p in pred_candidates if p.exists()), None)

        if pred is not None:
            pairs.append((real, pred))

    return pairs


def plot_clean_panel(real_path: Path, pred_path: Path, output_path: Path, tolerance: int):
    real = read_binary_png(real_path)
    pred = read_binary_png(pred_path)

    dice, iou = dice_iou(real, pred)

    filled = make_filled_rgb(real, pred)
    borders = make_boundary_rgb(real, pred, tolerance=tolerance)

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.0))

    axes[0].imshow(real, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Máscara real", fontsize=11)

    axes[1].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Máscara criada pelo modelo", fontsize=11)

    axes[2].imshow(filled)
    axes[2].set_title("Comparação cheia\nverde=acerto | vermelho=extra | azul=faltou", fontsize=9)

    axes[3].imshow(borders)
    axes[3].set_title(f"Comparação das bordas\ncom tolerância visual de {tolerance}px", fontsize=9)

    for ax in axes:
        ax.axis("off")

    title = real_path.name.replace("_real_mask.png", "").replace("_real.png", "")
    fig.suptitle(
        f"{title}\nDice={dice:.3f} | IoU={iou:.3f}",
        fontsize=12,
        y=1.03,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(panel_paths: list[Path], output_path: Path, cols: int = 2):
    if not panel_paths:
        return

    import matplotlib.image as mpimg

    rows = math.ceil(len(panel_paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8.8, rows * 2.7))
    axes = np.asarray(axes).reshape(-1)

    for ax, path in zip(axes, panel_paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(path.stem, fontsize=7)
        ax.axis("off")

    for ax in axes[len(panel_paths):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Gera PNGs limpos de comparação de máscaras e bordas a partir dos PNGs real/pred já existentes."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Pasta s2_png_real_vs_pred_final ou equivalente.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--tolerance", type=int, default=1)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = args.input_dir / "masks"
    if not masks_dir.exists():
        masks_dir = args.input_dir / "binary_masks"

    if not masks_dir.exists():
        raise FileNotFoundError(f"Não achei pasta de máscaras em {args.input_dir}")

    pairs = find_pairs(masks_dir)

    if not pairs:
        raise RuntimeError(f"Não achei pares *_real/_pred em {masks_dir}")

    pairs = pairs[: args.max_cases]

    panels_dir = args.output_dir / "clean_mask_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    panel_paths = []

    for i, (real, pred) in enumerate(tqdm(pairs, desc="Gerando painéis limpos"), start=1):
        stem = real.name.replace("_real_mask.png", "").replace("_real.png", "")
        out = panels_dir / f"{i:03d}_{safe_stem(stem)}_clean.png"
        plot_clean_panel(real, pred, out, tolerance=args.tolerance)
        panel_paths.append(out)

    make_contact_sheet(panel_paths, args.output_dir / "boss_clean_mask_contact_sheet.png", cols=2)

    print("\nFinalizado.")
    print(f"Painéis limpos gerados: {len(panel_paths)}")
    print("Pasta:", panels_dir)
    print("Prancha:", args.output_dir / "boss_clean_mask_contact_sheet.png")


if __name__ == "__main__":
    main()

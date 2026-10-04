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


def read_mask_png(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return (arr >= 0.5).astype(np.uint8)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def find_pairs(masks_dir: Path) -> list[tuple[Path, Path]]:
    real_files = sorted(list(masks_dir.glob("*_real.png")) + list(masks_dir.glob("*_real_mask.png")))
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


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    struct = ndimage.iterate_structure(np.ones((3, 3), dtype=bool), radius)
    return ndimage.binary_dilation(mask.astype(bool), structure=struct)


def skeleton_or_boundary(mask: np.ndarray, mode: str) -> np.ndarray:
    mask = mask.astype(bool)

    if mode == "raw":
        return mask.astype(np.uint8)

    if mode == "edge":
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
        return (mask ^ eroded).astype(np.uint8)

    raise ValueError("--view-mode deve ser raw ou edge")


def compare_boundary(real_boundary: np.ndarray, pred_boundary: np.ndarray, tolerance: int) -> np.ndarray:
    real = real_boundary.astype(bool)
    pred = pred_boundary.astype(bool)

    real_tol = dilate(real, tolerance)
    pred_tol = dilate(pred, tolerance)

    matched = pred & real_tol
    extra = pred & (~real_tol)
    missed = real & (~pred_tol)

    h, w = real.shape
    out = np.ones((h, w, 3), dtype=np.float32)

    # fundo cinza claro onde há qualquer borda.
    union = real | pred
    out[union] = np.array([0.92, 0.92, 0.92])

    out[matched] = np.array([0.05, 0.65, 0.15])  # verde
    out[extra] = np.array([0.90, 0.10, 0.05])    # vermelho
    out[missed] = np.array([0.10, 0.20, 0.90])   # azul

    return out


def dice_iou(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.astype(bool)
    b = b.astype(bool)

    tp = (a & b).sum()
    fp = ((~a) & b).sum()
    fn = (a & (~b)).sum()

    eps = 1e-7
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return float(dice), float(iou)


def plot_panel(
    real_original: np.ndarray,
    pred_original: np.ndarray,
    real_boundary: np.ndarray,
    pred_boundary: np.ndarray,
    out_path: Path,
    tolerance: int,
    title: str,
):
    comp = compare_boundary(real_boundary, pred_boundary, tolerance=tolerance)
    dice, iou = dice_iou(real_boundary, pred_boundary)

    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.1))

    axes[0].imshow(real_original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Máscara real original\nclasse positiva em branco", fontsize=9)

    axes[1].imshow(pred_original, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Máscara criada original\nclasse positiva em branco", fontsize=9)

    axes[2].imshow(real_boundary, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Bordas reais\nvisualização invertida", fontsize=10)

    axes[3].imshow(pred_boundary, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Bordas criadas\nvisualização invertida", fontsize=10)

    axes[4].imshow(comp)
    axes[4].set_title(
        f"Comparação de bordas tol={tolerance}px\nverde=acerto | vermelho=extra | azul=faltou",
        fontsize=8,
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{title}\nDice borda={dice:.3f} | IoU borda={iou:.3f}", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(paths: list[Path], out_path: Path, cols: int = 2):
    if not paths:
        return

    import matplotlib.image as mpimg

    rows = math.ceil(len(paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 9.2, rows * 2.8))
    axes = np.asarray(axes).reshape(-1)

    for ax, path in zip(axes, paths):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.stem, fontsize=7)
        ax.axis("off")

    for ax in axes[len(paths):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Gera visualização correta de bordas a partir das máscaras real/pred existentes."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Pasta que contém masks/ ou binary_masks/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument(
        "--view-mode",
        choices=["raw", "edge"],
        default="raw",
        help=(
            "raw: mostra a classe de borda como 1 após inversão; "
            "edge: pega só o contorno da região invertida. Para seu caso, comece com raw."
        ),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = args.input_dir / "masks"
    if not masks_dir.exists():
        masks_dir = args.input_dir / "binary_masks"

    if not masks_dir.exists():
        raise FileNotFoundError(f"Não achei pasta masks/ ou binary_masks/ dentro de {args.input_dir}")

    pairs = find_pairs(masks_dir)
    if not pairs:
        raise RuntimeError(f"Não achei pares *_real.png e *_pred.png em {masks_dir}")

    pairs = pairs[: args.max_cases]

    panels_dir = args.output_dir / "boundary_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for i, (real_path, pred_path) in enumerate(tqdm(pairs, desc="Gerando painéis de borda"), start=1):
        real_original = read_mask_png(real_path)
        pred_original = read_mask_png(pred_path)

        # O ponto principal:
        # nos seus PNGs atuais, a classe positiva está branca e as bordas aparecem pretas.
        # Para mostrar borda, precisamos inverter.
        real_inverted = 1 - real_original
        pred_inverted = 1 - pred_original

        real_boundary = skeleton_or_boundary(real_inverted, args.view_mode)
        pred_boundary = skeleton_or_boundary(pred_inverted, args.view_mode)

        stem = real_path.name.replace("_real_mask.png", "").replace("_real.png", "")
        out = panels_dir / f"{i:03d}_{safe_name(stem)}_boundary_view.png"

        plot_panel(
            real_original=real_original,
            pred_original=pred_original,
            real_boundary=real_boundary,
            pred_boundary=pred_boundary,
            out_path=out,
            tolerance=args.tolerance,
            title=stem,
        )

        saved.append(out)

    make_contact_sheet(saved, args.output_dir / "boss_boundary_contact_sheet.png", cols=2)

    print("\nFinalizado.")
    print(f"Painéis gerados: {len(saved)}")
    print("Painéis:", panels_dir)
    print("Prancha geral:", args.output_dir / "boss_boundary_contact_sheet.png")


if __name__ == "__main__":
    main()

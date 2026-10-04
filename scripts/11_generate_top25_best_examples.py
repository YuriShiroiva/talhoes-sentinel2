from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from field_boundary.data.dataset import AI4BoundariesSegmentationDataset
from field_boundary.models.factory import build_model


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


def plot_overlay(ax, rgb: np.ndarray, mask: np.ndarray, title: str, alpha: float = 0.45):
    ax.imshow(rgb)

    overlay = np.ma.masked_where(mask <= 0, mask)
    ax.imshow(
        overlay,
        cmap="autumn",
        alpha=alpha,
        vmin=0,
        vmax=1,
    )

    ax.set_title(title)
    ax.axis("off")


def select_best_cases(
    df: pd.DataFrame,
    top_k: int,
    include_empty: bool,
    min_target_ratio: float,
    max_target_ratio: float | None,
) -> pd.DataFrame:
    df = df.copy()

    required = ["dice_after_gate", "iou_after_gate", "precision_after_gate", "recall_after_gate", "f1_after_gate", "pass_gate", "target_pixels"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "O CSV precisa ser o per_sample_metrics.csv gerado pelo script 10. "
            f"Colunas ausentes: {missing}"
        )

    if "target_ratio" not in df.columns:
        df["target_ratio"] = df["target_pixels"] / df["total_pixels"]

    df["pass_gate_bool"] = df["pass_gate"].apply(to_bool)

    if not include_empty:
        df = df[df["target_pixels"] > 0].copy()

    df = df[df["target_ratio"] >= min_target_ratio].copy()

    if max_target_ratio is not None:
        df = df[df["target_ratio"] <= max_target_ratio].copy()

    if not include_empty:
        df = df[df["pass_gate_bool"]].copy()

    selected = df.sort_values(
        ["dice_after_gate", "iou_after_gate", "precision_after_gate"],
        ascending=False,
    ).head(top_k)

    return selected.reset_index(drop=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Gera as melhores figuras qualitativas do pipeline final."
    )

    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/final_model/test020_unet_pipeline_clean/per_sample_metrics.csv"),
        help="CSV gerado pelo script 10_evaluate_final_unet_pipeline.py.",
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        type=Path,
        default=Path("outputs/tests/test_019_unet_imagenet_3000samples_20epochs/best_unet_baseline.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_model/test020_top25_best_examples"),
    )

    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--segmentation-threshold", type=float, default=0.65)

    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Inclui imagens vazias. Por padrão, gera só casos com talhão real.",
    )
    parser.add_argument(
        "--min-target-ratio",
        type=float,
        default=0.01,
        help="Filtra casos muito pequenos. Use 0 para aceitar todos.",
    )
    parser.add_argument(
        "--max-target-ratio",
        type=float,
        default=None,
        help="Opcional: filtra casos grandes demais. Ex: 0.95.",
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

    selected = select_best_cases(
        df=df,
        top_k=args.top_k,
        include_empty=args.include_empty,
        min_target_ratio=args.min_target_ratio,
        max_target_ratio=args.max_target_ratio,
    )

    if selected.empty:
        raise RuntimeError("Nenhum caso selecionado. Tente reduzir --min-target-ratio ou usar --include-empty.")

    selected_path = args.output_dir / "selected_top_cases.csv"
    selected.to_csv(selected_path, index=False)

    print(f"Casos selecionados: {len(selected)}")
    print(
        selected[
            [
                "image_path",
                "dice_after_gate",
                "iou_after_gate",
                "precision_after_gate",
                "recall_after_gate",
                "f1_after_gate",
                "prob_non_empty",
                "target_ratio",
            ]
        ].to_string(index=False)
    )

    segmenter = load_segmenter(args.segmenter_checkpoint, device)

    for rank, row in selected.iterrows():
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        dataset = AI4BoundariesSegmentationDataset(
            pairs=[(image_path, mask_path)],
            image_size=args.image_size,
            in_channels=args.in_channels,
            augment=False,
        )

        sample = dataset[0]

        image = sample["image"].unsqueeze(0).to(device)
        target_mask = sample["mask"].detach().cpu().numpy()[0].astype(np.float32)

        probs = torch.sigmoid(segmenter(image))
        prob_np = probs[0, 0].detach().cpu().numpy()

        pred_before_gate = (prob_np >= args.segmentation_threshold).astype(np.float32)

        pass_gate = to_bool(row["pass_gate"])

        if pass_gate:
            pred_after_gate = pred_before_gate
        else:
            pred_after_gate = np.zeros_like(pred_before_gate, dtype=np.float32)

        rgb = image_to_rgb(sample["image"])

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(rgb)
        axes[0].set_title("Ortofoto")
        axes[0].axis("off")

        plot_overlay(
            axes[1],
            rgb,
            target_mask,
            "Máscara real",
        )

        plot_overlay(
            axes[2],
            rgb,
            pred_before_gate,
            "U-Net sem gate",
        )

        plot_overlay(
            axes[3],
            rgb,
            pred_after_gate,
            "Pipeline final",
        )

        title = (
            f"Top {rank + 1:02d} | "
            f"Dice={row['dice_after_gate']:.3f} | "
            f"IoU={row['iou_after_gate']:.3f} | "
            f"Precision={row['precision_after_gate']:.3f} | "
            f"Recall={row['recall_after_gate']:.3f} | "
            f"prob_non_empty={row['prob_non_empty']:.3f}"
        )

        fig.suptitle(title, fontsize=10)
        fig.tight_layout()

        safe_stem = image_path.stem.replace("_ortho_1m_512", "")
        output_path = args.output_dir / f"top_{rank + 1:02d}_dice_{row['dice_after_gate']:.3f}_{safe_stem}.png"

        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        print(f"Salvo: {output_path}")

    summary_md = args.output_dir / "README_top25.md"

    lines = [
        "# Top melhores exemplos do pipeline final",
        "",
        "Critério de seleção: maiores valores de `dice_after_gate`, com filtros configuráveis.",
        "",
        f"- CSV de métricas: `{args.metrics_csv}`",
        f"- Checkpoint do segmentador: `{args.segmenter_checkpoint}`",
        f"- Threshold de segmentação: `{args.segmentation_threshold}`",
        f"- Top K: `{args.top_k}`",
        f"- Include empty: `{args.include_empty}`",
        f"- Min target ratio: `{args.min_target_ratio}`",
        "",
        "## Casos selecionados",
        "",
        "| Rank | Imagem | Dice | IoU | Precision | Recall | F1 | Prob. non-empty | Target ratio |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, row in selected.iterrows():
        lines.append(
            "| "
            f"{rank + 1} | "
            f"{Path(row['image_path']).name} | "
            f"{row['dice_after_gate']:.4f} | "
            f"{row['iou_after_gate']:.4f} | "
            f"{row['precision_after_gate']:.4f} | "
            f"{row['recall_after_gate']:.4f} | "
            f"{row['f1_after_gate']:.4f} | "
            f"{row['prob_non_empty']:.4f} | "
            f"{row['target_ratio']:.4f} |"
        )

    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print("\nFinalizado!")
    print(f"Figuras salvas em: {args.output_dir}")
    print(f"CSV dos casos selecionados: {selected_path}")
    print(f"Resumo Markdown: {summary_md}")


if __name__ == "__main__":
    main()

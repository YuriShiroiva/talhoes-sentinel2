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


def to_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).lower() in ["true", "1", "yes", "sim"]


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


def select_cases(df: pd.DataFrame, max_cases_per_group: int = 2) -> pd.DataFrame:
    df = df.copy()

    if "target_pixels" not in df.columns:
        df["target_pixels"] = df["tp"] + df["fn"]

    if "pred_pixels" not in df.columns:
        df["pred_pixels"] = df["tp"] + df["fp"]

    if "total_pixels" not in df.columns:
        df["total_pixels"] = df["tp"] + df["fp"] + df["fn"] + df["tn"]

    df["target_ratio"] = df["target_pixels"] / df["total_pixels"]
    df["pass_gate_bool"] = df["pass_gate"].apply(to_bool)

    cases = []

    non_empty = df[df["target_pixels"] > 0].copy()
    empty = df[df["target_pixels"] == 0].copy()

    # Bons casos com talhão
    good = non_empty.sort_values("dice_gated", ascending=False).head(max_cases_per_group)
    good = good.assign(case_type="bom_com_talhao")
    cases.append(good)

    # Casos ruins com talhão
    bad_pool = non_empty[non_empty["target_ratio"] > 0.05].copy()
    bad = bad_pool.sort_values("dice_gated", ascending=True).head(max_cases_per_group)
    bad = bad.assign(case_type="ruim_com_talhao_visivel")
    cases.append(bad)

    # Imagens vazias bloqueadas pelo gate.
    # No Teste 020, o CSV já está pós-gate, então fp pode ser 0.
    # Mesmo assim, essas imagens são importantes para mostrar o efeito do classificador.
    empty_fixed = empty[
        ~empty["pass_gate_bool"]
    ].sort_values("prob_non_empty", ascending=False).head(max_cases_per_group)
    
    empty_fixed = empty_fixed.assign(case_type="vazio_corrigido_gate")
    cases.append(empty_fixed)

    # Imagens vazias que ainda passaram no gate
    empty_wrong = empty[
        empty["fp_gated"] > 0
    ].sort_values("fp_gated", ascending=False).head(max_cases_per_group)
    empty_wrong = empty_wrong.assign(case_type="vazio_ainda_errado")
    cases.append(empty_wrong)

    # Talhões pequenos
    small = non_empty[
        non_empty["target_ratio"] < 0.05
    ].sort_values("dice_gated", ascending=False).head(max_cases_per_group)
    small = small.assign(case_type="talhao_pequeno")
    cases.append(small)

    selected = pd.concat(cases, ignore_index=True)

    selected["image_path_key"] = selected["image_path"].astype(str).str.replace("\\", "/", regex=False)
    selected = selected.drop_duplicates(subset=["image_path_key"])

    return selected

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
    
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Gera figuras comparativas do pipeline final")

    parser.add_argument(
        "--gate-csv",
        type=Path,
        default=Path("outputs/official_val/test_015_gate_empty_classifier/per_sample_metrics_gate_thr075.csv"),
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=Path("outputs/tests/test_008_unet_imagenet_1000samples_10epochs/best_unet_baseline.pt"),
    )
    parser.add_argument(
        "--deeplab-checkpoint",
        type=Path,
        default=Path("outputs/tests/test_007_deeplabv3plus_1000samples_10epochs/best_unet_baseline.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/test_016_pipeline_examples"),
    )

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--unet-weight", type=float, default=0.70)
    parser.add_argument("--segmentation-threshold", type=float, default=0.60)
    parser.add_argument("--classifier-threshold", type=float, default=0.75)
    parser.add_argument("--max-cases-per-group", type=int, default=2)

    args = parser.parse_args()

    if not args.gate_csv.exists():
        raise FileNotFoundError(f"Não encontrei o CSV do gate: {args.gate_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.gate_csv)
    # Compatibilidade com o Teste 020, que usa nomes sem "_gated"
    if "dice_gated" not in df.columns and "dice" in df.columns:
        df["dice_gated"] = df["dice"]
    
    if "iou_gated" not in df.columns and "iou" in df.columns:
        df["iou_gated"] = df["iou"]
    
    if "fp_gated" not in df.columns and "fp" in df.columns:
        df["fp_gated"] = df["fp"]
    
    if "tp_gated" not in df.columns and "tp" in df.columns:
        df["tp_gated"] = df["tp"]
    
    if "fn_gated" not in df.columns and "fn" in df.columns:
        df["fn_gated"] = df["fn"]
    
    if "tn_gated" not in df.columns and "tn" in df.columns:
        df["tn_gated"] = df["tn"]
        selected = select_cases(df, max_cases_per_group=args.max_cases_per_group)

    print(f"Casos selecionados: {len(selected)}")
    print(selected[["case_type", "image_path", "dice_gated", "prob_non_empty", "pass_gate"]])

    unet = load_segmenter(args.unet_checkpoint, device)
    deeplab = load_segmenter(args.deeplab_checkpoint, device)

    deeplab_weight = 1.0 - args.unet_weight

    for idx, row in selected.iterrows():
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
        mask = sample["mask"].detach().cpu().numpy()[0]

        probs_unet = torch.sigmoid(unet(image))
        probs_deeplab = torch.sigmoid(deeplab(image))

        probs_final = args.unet_weight * probs_unet + deeplab_weight * probs_deeplab
        prob_np = probs_final[0, 0].detach().cpu().numpy()

        ensemble_mask = (prob_np >= args.segmentation_threshold).astype(np.float32)

        prob_non_empty = float(row["prob_non_empty"])
        pass_gate = prob_non_empty >= args.classifier_threshold

        if pass_gate:
            gated_mask = ensemble_mask
        else:
            gated_mask = np.zeros_like(ensemble_mask)

        rgb = image_to_rgb(sample["image"])

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(rgb)
        axes[0].set_title("Ortofoto")
        axes[0].axis("off")

        plot_overlay(
            axes[1],
            rgb,
            mask,
            "Máscara real",
        )

        plot_overlay(
            axes[2],
            rgb,
            ensemble_mask,
            "Ensemble sem gate",
        )

        plot_overlay(
            axes[3],
            rgb,
            gated_mask,
            "Pipeline final",
        )

        title = (
            f"{row['case_type']} | "
            f"prob_non_empty={prob_non_empty:.3f} | "
            f"pass_gate={pass_gate} | "
            f"dice_gate={row['dice_gated']:.3f}"
        )

        fig.suptitle(title, fontsize=11)
        fig.tight_layout()

        safe_name = f"{idx:03d}_{row['case_type']}_{image_path.stem}.png"
        output_path = args.output_dir / safe_name

        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        print(f"Salvo: {output_path}")

    selected.to_csv(args.output_dir / "selected_cases.csv", index=False)

    print(f"\nFiguras salvas em: {args.output_dir}")
    print(f"Lista de casos: {args.output_dir / 'selected_cases.csv'}")


if __name__ == "__main__":
    main()
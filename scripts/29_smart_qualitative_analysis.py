from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Script 29
# Seleção QUALITATIVA INTELIGENTE para o TCC
#
# Este script NÃO roda inferência.
# Ele usa o CSV já gerado pelo script qualitativo anterior:
#   o23_per_sample_metrics_recomputed.csv
#
# Objetivo:
#   1. gerar métricas estratificadas honestas;
#   2. selecionar casos visualmente/analiticamente úteis;
#   3. evitar "melhores" triviais: vazios ou tiles quase cheios;
#   4. separar casos difíceis, pequenos, borda alta e quase cheios.
#
# Para gerar painéis PNG novos, este script tenta reutilizar as imagens já
# existentes nas pastas do qualitativo anterior quando possível. O foco aqui
# é a seleção/tabelas. Para novos painéis, é melhor adaptar o script 28 depois.
# ============================================================


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "target_ratio" not in df.columns:
        df["target_ratio"] = df["target_pixels"] / df["total_pixels"]

    if "pred_ratio" not in df.columns:
        df["pred_ratio"] = df["pred_pixels"] / df["total_pixels"]

    df["abs_area_error"] = (df["pred_ratio"] - df["target_ratio"]).abs()

    # Casos realmente úteis para mostrar em TCC:
    # remove vazios e remove tiles quase cheios.
    df["is_empty"] = df["target_ratio"] == 0
    df["is_almost_full"] = df["target_ratio"] >= 0.90
    df["is_very_large"] = df["target_ratio"] >= 0.75
    df["is_useful_5_75"] = (df["target_ratio"] >= 0.05) & (df["target_ratio"] < 0.75)
    df["is_nontrivial_1_90"] = (df["target_ratio"] >= 0.01) & (df["target_ratio"] < 0.90)
    df["is_small"] = (df["target_ratio"] >= 0.01) & (df["target_ratio"] < 0.05)
    df["is_very_small"] = (df["target_ratio"] > 0) & (df["target_ratio"] < 0.01)

    # Para análise de falsos positivos em vazios.
    df["empty_false_positive"] = df["is_empty"] & (df["pred_pixels"] > 0)

    # Um score de interesse visual: prefere casos não triviais, com alvo parcial,
    # nem fácil demais nem impossível demais.
    # Casos com dice perto de 0.5-0.85 costumam ser mais explicativos.
    df["visual_interest_score"] = (
        (df["target_ratio"].clip(0, 0.75) / 0.75)
        + df["abs_area_error"].clip(0, 0.30) / 0.30
        + (1.0 - (df["dice"] - 0.65).abs()).clip(0, 1)
    )

    return df


def metric_summary(df: pd.DataFrame, name: str) -> Dict:
    metrics = ["dice", "iou", "precision", "recall", "f1"]
    out = {
        "subset": name,
        "count": int(len(df)),
    }

    if len(df) == 0:
        for metric in metrics:
            out[f"{metric}_mean"] = np.nan
            out[f"{metric}_median"] = np.nan
            out[f"{metric}_std"] = np.nan
        return out

    for metric in metrics:
        out[f"{metric}_mean"] = float(df[metric].mean())
        out[f"{metric}_median"] = float(df[metric].median())
        out[f"{metric}_std"] = float(df[metric].std(ddof=0))

    out["target_ratio_mean"] = float(df["target_ratio"].mean())
    out["pred_ratio_mean"] = float(df["pred_ratio"].mean())
    out["abs_area_error_mean"] = float(df["abs_area_error"].mean())

    return out


def build_stratified_tables(df: pd.DataFrame, out_dir: Path) -> None:
    subsets = {
        "all": df,
        "non_empty": df[~df["is_empty"]],
        "useful_5_75": df[df["is_useful_5_75"]],
        "nontrivial_1_90": df[df["is_nontrivial_1_90"]],
        "small_1_5": df[df["is_small"]],
        "very_small_0_1": df[df["is_very_small"]],
        "very_large_75_90": df[(df["target_ratio"] >= 0.75) & (df["target_ratio"] < 0.90)],
        "almost_full_90_100": df[df["is_almost_full"]],
        "empty": df[df["is_empty"]],
        "empty_false_positive": df[df["empty_false_positive"]],
    }

    rows = [metric_summary(subset, name) for name, subset in subsets.items()]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "stratified_metrics_summary.csv", index=False)

    # Por classe original se existir.
    if "mask_size_class" in df.columns:
        by_size = (
            df.groupby("mask_size_class")
            .agg(
                count=("dice", "size"),
                dice_mean=("dice", "mean"),
                dice_median=("dice", "median"),
                iou_mean=("iou", "mean"),
                precision_mean=("precision", "mean"),
                recall_mean=("recall", "mean"),
                f1_mean=("f1", "mean"),
                target_ratio_mean=("target_ratio", "mean"),
                pred_ratio_mean=("pred_ratio", "mean"),
                abs_area_error_mean=("abs_area_error", "mean"),
            )
            .reset_index()
            .sort_values("target_ratio_mean")
        )
        by_size.to_csv(out_dir / "metrics_by_original_mask_size_class.csv", index=False)

    # Por bins percentuais mais úteis.
    bins = [-0.000001, 0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.000001]
    labels = [
        "empty",
        "0_1",
        "1_5",
        "5_10",
        "10_25",
        "25_50",
        "50_75",
        "75_90",
        "90_100",
    ]
    df_bins = df.copy()
    df_bins["target_ratio_bin"] = pd.cut(
        df_bins["target_ratio"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    by_bin = (
        df_bins.groupby("target_ratio_bin", observed=False)
        .agg(
            count=("dice", "size"),
            dice_mean=("dice", "mean"),
            dice_median=("dice", "median"),
            iou_mean=("iou", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            target_ratio_mean=("target_ratio", "mean"),
            pred_ratio_mean=("pred_ratio", "mean"),
            abs_area_error_mean=("abs_area_error", "mean"),
        )
        .reset_index()
    )
    by_bin.to_csv(out_dir / "metrics_by_target_ratio_bin.csv", index=False)


def select_cases(df: pd.DataFrame, n: int, seed: int) -> Dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    groups: Dict[str, pd.DataFrame] = {}

    useful = df[df["is_useful_5_75"]].copy()
    nontrivial = df[df["is_nontrivial_1_90"]].copy()
    small = df[df["is_small"]].copy()
    very_small = df[df["is_very_small"]].copy()
    empty = df[df["is_empty"]].copy()
    almost_full = df[df["is_almost_full"]].copy()

    # Bons casos úteis: não vazios, não quase-cheios.
    groups["01_good_useful_5_75"] = useful.sort_values(["dice", "iou"], ascending=False).head(n)

    # Ruins úteis: não vazios e não quase-cheios.
    groups["02_bad_useful_5_75"] = useful.sort_values(["dice", "iou"], ascending=True).head(n)

    # Casos medianos/explicativos: bom para discussão visual.
    if len(useful) > 0:
        tmp = useful.copy()
        tmp["distance_to_075"] = (tmp["dice"] - 0.75).abs()
        groups["03_intermediate_useful_near_075"] = tmp.sort_values(["distance_to_075", "abs_area_error"]).head(n)

    # Pequenos: bons e ruins separados.
    if len(small) > 0:
        groups["04_good_small_1_5"] = small.sort_values(["dice", "iou"], ascending=False).head(n)
        groups["05_bad_small_1_5"] = small.sort_values(["dice", "iou"], ascending=True).head(n)

    if len(very_small) > 0:
        groups["06_very_small_cases"] = very_small.sort_values(["dice", "target_ratio"], ascending=[True, True]).head(n)

    # Vazios: falsos positivos e acertos separados.
    if len(empty) > 0:
        groups["07_empty_false_positive"] = empty[empty["pred_pixels"] > 0].sort_values(
            ["pred_pixels", "dice"],
            ascending=[False, True],
        ).head(n)
        groups["08_empty_correct"] = empty[empty["pred_pixels"] == 0].head(n)

    # Quase cheio: separado para não inflar "best".
    if len(almost_full) > 0:
        groups["09_almost_full_high_dice"] = almost_full.sort_values(["dice", "iou"], ascending=False).head(n)
        groups["10_almost_full_low_dice"] = almost_full.sort_values(["dice", "iou"], ascending=True).head(n)

    # Maior erro de área em casos não triviais.
    if len(nontrivial) > 0:
        groups["11_high_area_error_nontrivial"] = nontrivial.sort_values(
            "abs_area_error",
            ascending=False,
        ).head(n)

    # Amostra aleatória útil, para evitar cherry-picking.
    if len(useful) > 0:
        idx = list(useful.index)
        k = min(n, len(idx))
        groups["12_random_useful_5_75"] = useful.loc[rng.sample(idx, k=k)]

    return groups


def save_selected_cases(groups: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    selected_root = ensure_dir(out_dir / "selected_cases")

    all_rows = []
    for group_name, group_df in groups.items():
        group_dir = ensure_dir(selected_root / group_name)
        out_csv = group_dir / f"{group_name}.csv"
        group_df.to_csv(out_csv, index=False)

        tmp = group_df.copy()
        tmp.insert(0, "selection_group", group_name)
        all_rows.append(tmp)

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            selected_root / "all_selected_cases.csv",
            index=False,
        )


def plot_metric_by_ratio(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)

    ax.scatter(df["target_ratio"], df["dice"], s=8, alpha=0.35)
    ax.set_xlabel("Proporção da máscara real no tile")
    ax.set_ylabel("Dice por amostra")
    ax.set_title("Dice por proporção de área ocupada pela máscara")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "scatter_dice_vs_target_ratio.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.scatter(df["target_ratio"], df["abs_area_error"], s=8, alpha=0.35)
    ax.set_xlabel("Proporção da máscara real no tile")
    ax.set_ylabel("Erro absoluto de área")
    ax.set_title("Erro de área por proporção da máscara")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "scatter_abs_area_error_vs_target_ratio.png", bbox_inches="tight")
    plt.close(fig)


def write_recommendations(df: pd.DataFrame, out_dir: Path) -> None:
    all_summary = metric_summary(df, "all")
    useful_summary = metric_summary(df[df["is_useful_5_75"]], "useful_5_75")
    non_empty_summary = metric_summary(df[~df["is_empty"]], "non_empty")
    empty_fp_count = int(df["empty_false_positive"].sum())
    empty_count = int(df["is_empty"].sum())

    text = f"""# Análise qualitativa e estratificada — O23

## Por que a seleção anterior era fraca

A seleção baseada apenas no maior Dice tende a escolher casos triviais:
tiles vazios classificados corretamente ou tiles quase totalmente ocupados pela máscara.
Esses casos podem gerar Dice igual a 1, mas não demonstram a capacidade do modelo em segmentar bordas e talhões parciais.

## Resultado global

- Número de amostras: {len(df)}
- Dice global médio: {all_summary.get("dice_mean", float("nan")):.6f}
- IoU global médio: {all_summary.get("iou_mean", float("nan")):.6f}
- F1 global médio: {all_summary.get("f1_mean", float("nan")):.6f}

## Resultado sem vazios

- Número de amostras não vazias: {non_empty_summary.get("count", 0)}
- Dice médio: {non_empty_summary.get("dice_mean", float("nan")):.6f}
- IoU médio: {non_empty_summary.get("iou_mean", float("nan")):.6f}
- F1 médio: {non_empty_summary.get("f1_mean", float("nan")):.6f}

## Resultado em casos úteis: 5% a 75% de ocupação

Esse subconjunto remove casos vazios e casos quase totalmente preenchidos,
representando melhor o desafio prático de segmentação.

- Número de amostras úteis: {useful_summary.get("count", 0)}
- Dice médio: {useful_summary.get("dice_mean", float("nan")):.6f}
- IoU médio: {useful_summary.get("iou_mean", float("nan")):.6f}
- F1 médio: {useful_summary.get("f1_mean", float("nan")):.6f}

## Vazios

- Amostras vazias: {empty_count}
- Vazios com falso positivo: {empty_fp_count}

## Como usar as figuras no TCC

Use preferencialmente as pastas:

1. `01_good_useful_5_75`: bons exemplos não triviais.
2. `02_bad_useful_5_75`: falhas relevantes.
3. `03_intermediate_useful_near_075`: exemplos explicativos intermediários.
4. `04_good_small_1_5` e `05_bad_small_1_5`: análise de talhões pequenos.
5. `07_empty_false_positive`: limitações em imagens vazias.
6. `09_almost_full_high_dice`: mostrar separadamente que casos quase cheios existem, mas não devem ser usados como "melhores" principais.

## Recomendação metodológica

No texto do TCC, apresente o resultado global, mas discuta também o subconjunto útil.
Isso evita uma interpretação inflada do desempenho e mostra maturidade experimental.
"""

    (out_dir / "qualitative_selection_recommendations.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Gera seleção qualitativa inteligente a partir das métricas por amostra.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="Caminho para o o23_per_sample_metrics_recomputed.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_report_assets/o23_smart_analysis"),
    )
    parser.add_argument("--n-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)
    df = add_derived_columns(df)

    # Salva versão enriquecida.
    df.to_csv(args.output_dir / "o23_per_sample_metrics_enriched.csv", index=False)

    build_stratified_tables(df, args.output_dir)

    groups = select_cases(df, n=args.n_per_group, seed=args.seed)
    save_selected_cases(groups, args.output_dir)

    plot_metric_by_ratio(df, args.output_dir)
    write_recommendations(df, args.output_dir)

    print("\nAnálise inteligente criada em:", args.output_dir)
    print("- o23_per_sample_metrics_enriched.csv")
    print("- stratified_metrics_summary.csv")
    print("- metrics_by_target_ratio_bin.csv")
    print("- selected_cases/all_selected_cases.csv")
    print("- qualitative_selection_recommendations.md")
    print("- scatter_dice_vs_target_ratio.png")
    print("- scatter_abs_area_error_vs_target_ratio.png")

    print("\nResumo estratificado:")
    summary = pd.read_csv(args.output_dir / "stratified_metrics_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

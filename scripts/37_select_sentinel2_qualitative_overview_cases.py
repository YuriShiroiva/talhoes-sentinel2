from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["overview_group"] = "other"

    out.loc[(out["target_ratio"] == 0) & (out["pred_pixels"] == 0), "overview_group"] = "empty_correct"
    out.loc[(out["target_ratio"] == 0) & (out["pred_pixels"] > 0), "overview_group"] = "empty_false_positive"

    out.loc[(out["target_ratio"] > 0) & (out["target_ratio"] < 0.01), "overview_group"] = "very_small_0_1"
    out.loc[(out["target_ratio"] >= 0.01) & (out["target_ratio"] < 0.05), "overview_group"] = "small_1_5"
    out.loc[(out["target_ratio"] >= 0.05) & (out["target_ratio"] < 0.25), "overview_group"] = "medium_5_25"
    out.loc[(out["target_ratio"] >= 0.25) & (out["target_ratio"] < 0.75), "overview_group"] = "large_25_75"
    out.loc[(out["target_ratio"] >= 0.75) & (out["target_ratio"] < 0.90), "overview_group"] = "very_large_75_90"
    out.loc[out["target_ratio"] >= 0.90, "overview_group"] = "almost_full_90_100"

    out["quality_band"] = pd.cut(
        out["dice"],
        bins=[-0.001, 0.25, 0.50, 0.65, 0.80, 0.90, 0.98, 0.995, 1.000001],
        labels=[
            "very_bad_0_25",
            "bad_25_50",
            "low_mid_50_65",
            "mid_65_80",
            "good_80_90",
            "very_good_90_98",
            "near_perfect_98_995",
            "perfect_995_100",
        ],
        include_lowest=True,
    ).astype(str)

    out["abs_area_error"] = (out["pred_ratio"] - out["target_ratio"]).abs()

    return out


def pick_quantile_cases(
    df: pd.DataFrame,
    group_name: str,
    subset: pd.DataFrame,
    n: int,
    seed: int,
    exclude_perfect: bool = True,
) -> pd.DataFrame:
    if exclude_perfect:
        subset = subset[subset["dice"] < 0.995].copy()

    if len(subset) == 0:
        return pd.DataFrame()

    # Pega casos espalhados por desempenho, não só os melhores.
    quantiles = np.linspace(0.15, 0.85, n)
    selected = []

    ordered = subset.sort_values("dice").reset_index(drop=True)

    used = set()
    for q in quantiles:
        idx = int(round(q * (len(ordered) - 1)))
        row = ordered.iloc[idx]
        key = row["image_path"]
        if key not in used:
            selected.append(row)
            used.add(key)

    out = pd.DataFrame(selected)
    out.insert(0, "selection_group", group_name)
    out.insert(1, "selection_strategy", "quantile_spread_excluding_perfect" if exclude_perfect else "quantile_spread")
    return out


def pick_top_not_perfect(
    df: pd.DataFrame,
    group_name: str,
    subset: pd.DataFrame,
    n: int,
    min_dice: float,
    max_dice: float,
) -> pd.DataFrame:
    sub = subset[(subset["dice"] >= min_dice) & (subset["dice"] < max_dice)].copy()

    if len(sub) == 0:
        return pd.DataFrame()

    sub = sub.sort_values(["dice", "abs_area_error"], ascending=[False, True]).head(n).copy()
    sub.insert(0, "selection_group", group_name)
    sub.insert(1, "selection_strategy", f"top_between_{min_dice}_{max_dice}")
    return sub


def pick_worst(
    df: pd.DataFrame,
    group_name: str,
    subset: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    if len(subset) == 0:
        return pd.DataFrame()

    sub = subset.sort_values(["dice", "abs_area_error"], ascending=[True, False]).head(n).copy()
    sub.insert(0, "selection_group", group_name)
    sub.insert(1, "selection_strategy", "worst_cases")
    return sub


def main():
    parser = argparse.ArgumentParser(
        description="Seleciona casos qualitativos representativos sem pegar só Dice 100%."
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-perfect-threshold", type=float, default=0.995)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)
    df = add_groups(df)

    df.to_csv(args.output_dir / "metrics_enriched_overview.csv", index=False)

    # Resumo geral por grupo e por faixa de qualidade.
    by_group = (
        df.groupby("overview_group")
        .agg(
            count=("dice", "size"),
            dice_mean=("dice", "mean"),
            dice_median=("dice", "median"),
            dice_min=("dice", "min"),
            dice_max=("dice", "max"),
            iou_mean=("iou", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            target_ratio_mean=("target_ratio", "mean"),
            pred_ratio_mean=("pred_ratio", "mean"),
            abs_area_error_mean=("abs_area_error", "mean"),
        )
        .reset_index()
        .sort_values("dice_mean", ascending=False)
    )

    by_quality = (
        df.groupby(["overview_group", "quality_band"])
        .agg(
            count=("dice", "size"),
            dice_mean=("dice", "mean"),
            target_ratio_mean=("target_ratio", "mean"),
            pred_ratio_mean=("pred_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["overview_group", "quality_band"])
    )

    by_group.to_csv(args.output_dir / "overview_metrics_by_group.csv", index=False)
    by_quality.to_csv(args.output_dir / "overview_quality_distribution.csv", index=False)

    selections = []

    useful = df[(df["target_ratio"] >= 0.05) & (df["target_ratio"] < 0.75)]
    non_empty = df[df["target_ratio"] > 0]
    small = df[(df["target_ratio"] >= 0.01) & (df["target_ratio"] < 0.05)]
    very_small = df[(df["target_ratio"] > 0) & (df["target_ratio"] < 0.01)]
    empty_fp = df[(df["target_ratio"] == 0) & (df["pred_pixels"] > 0)]
    empty_correct = df[(df["target_ratio"] == 0) & (df["pred_pixels"] == 0)]
    very_large = df[(df["target_ratio"] >= 0.75) & (df["target_ratio"] < 0.90)]
    almost_full = df[df["target_ratio"] >= 0.90]

    # Melhores, mas sem Dice 1.0.
    selections.append(
        pick_top_not_perfect(
            df,
            "01_best_non_perfect_overall",
            df,
            args.n_per_group,
            min_dice=0.80,
            max_dice=args.exclude_perfect_threshold,
        )
    )

    selections.append(
        pick_top_not_perfect(
            df,
            "02_best_useful_5_75_non_perfect",
            useful,
            args.n_per_group,
            min_dice=0.80,
            max_dice=args.exclude_perfect_threshold,
        )
    )

    selections.append(
        pick_top_not_perfect(
            df,
            "03_best_small_1_5_non_perfect",
            small,
            args.n_per_group,
            min_dice=0.40,
            max_dice=args.exclude_perfect_threshold,
        )
    )

    # Visão geral: bons, médios e ruins por quantis.
    selections.append(
        pick_quantile_cases(
            df,
            "04_overview_useful_5_75_spread",
            useful,
            args.n_per_group,
            args.seed,
            exclude_perfect=True,
        )
    )

    selections.append(
        pick_quantile_cases(
            df,
            "05_overview_non_empty_spread",
            non_empty,
            args.n_per_group,
            args.seed,
            exclude_perfect=True,
        )
    )

    selections.append(
        pick_quantile_cases(
            df,
            "06_overview_small_1_5_spread",
            small,
            args.n_per_group,
            args.seed,
            exclude_perfect=False,
        )
    )

    selections.append(
        pick_quantile_cases(
            df,
            "07_overview_very_small_spread",
            very_small,
            args.n_per_group,
            args.seed,
            exclude_perfect=False,
        )
    )

    selections.append(
        pick_quantile_cases(
            df,
            "08_overview_large_75_90_spread",
            very_large,
            args.n_per_group,
            args.seed,
            exclude_perfect=True,
        )
    )

    selections.append(
        pick_quantile_cases(
            df,
            "09_overview_almost_full_spread",
            almost_full,
            args.n_per_group,
            args.seed,
            exclude_perfect=True,
        )
    )

    # Falhas importantes.
    selections.append(
        pick_worst(
            df,
            "10_worst_useful_5_75",
            useful,
            args.n_per_group,
        )
    )

    selections.append(
        pick_worst(
            df,
            "11_empty_false_positive",
            empty_fp,
            args.n_per_group,
        )
    )

    # Alguns vazios corretos, mas não precisa focar em 100%.
    if len(empty_correct):
        sub = empty_correct.sort_values("pred_ratio", ascending=True).head(args.n_per_group).copy()
        sub.insert(0, "selection_group", "12_empty_correct_examples")
        sub.insert(1, "selection_strategy", "empty_correct_low_pred_ratio")
        selections.append(sub)

    selected = pd.concat([s for s in selections if s is not None and len(s) > 0], ignore_index=True)

    # Evita duplicados dentro do arquivo consolidado mantendo a primeira razão de seleção.
    selected_unique = selected.drop_duplicates("image_path", keep="first").copy()

    selected.to_csv(args.output_dir / "overview_selected_cases_all_reasons.csv", index=False)
    selected_unique.to_csv(args.output_dir / "overview_selected_cases_unique.csv", index=False)

    # Salva CSV por grupo para ficar fácil de usar em relatório/visualização.
    cases_dir = args.output_dir / "selected_case_groups"
    cases_dir.mkdir(exist_ok=True)

    for group, sub in selected.groupby("selection_group"):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(group))
        sub.to_csv(cases_dir / f"{safe}.csv", index=False)

    print("\nResumo por grupo:")
    print(by_group.to_string(index=False))

    print("\nDistribuição por faixa de qualidade:")
    print(by_quality.to_string(index=False))

    cols = [
        "selection_group",
        "selection_strategy",
        "image_path",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "target_ratio",
        "pred_ratio",
        "abs_area_error",
        "overview_group",
        "quality_band",
    ]

    print("\nCasos selecionados, únicos:")
    print(selected_unique[cols].to_string(index=False))

    print("\nArquivos salvos em:", args.output_dir)
    print("- metrics_enriched_overview.csv")
    print("- overview_metrics_by_group.csv")
    print("- overview_quality_distribution.csv")
    print("- overview_selected_cases_all_reasons.csv")
    print("- overview_selected_cases_unique.csv")
    print("- selected_case_groups/*.csv")


if __name__ == "__main__":
    main()

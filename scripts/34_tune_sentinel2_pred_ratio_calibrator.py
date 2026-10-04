from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "image_path",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "target_ratio",
        "pred_ratio",
        "target_pixels",
        "pred_pixels",
        "total_pixels",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sem colunas necessárias: {missing}")

    return df.copy()


def zero_metrics_for_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    is_empty_target = out["target_pixels"] == 0
    eps = 1e-7
    target_pixels = out["target_pixels"].astype(float).values

    dice = np.where(is_empty_target, 1.0, eps / (target_pixels + eps))
    iou = np.where(is_empty_target, 1.0, eps / (target_pixels + eps))
    precision = np.where(is_empty_target, 1.0, 1.0)
    recall = np.where(is_empty_target, 1.0, eps / (target_pixels + eps))
    f1 = np.where(is_empty_target, 1.0, (2 * precision * recall + eps) / (precision + recall + eps))

    out["dice"] = dice
    out["iou"] = iou
    out["precision"] = precision
    out["recall"] = recall
    out["f1"] = f1
    out["pred_pixels"] = 0.0
    out["pred_ratio"] = 0.0

    return out


def apply_rule(
    df: pd.DataFrame,
    low_zero_thr: float | None,
    window_low: float | None,
    window_high: float | None,
) -> pd.DataFrame:
    out = df.copy()
    zero_mask = np.zeros(len(out), dtype=bool)

    if low_zero_thr is not None:
        zero_mask |= out["pred_ratio"].values < low_zero_thr

    if window_low is not None and window_high is not None:
        zero_mask |= (
            (out["pred_ratio"].values >= window_low)
            & (out["pred_ratio"].values < window_high)
        )

    out["calib_model"] = "keep"

    if zero_mask.any():
        zeroed = zero_metrics_for_rows(out.loc[zero_mask].copy())
        out.loc[zero_mask, "calib_model"] = "zero"

        for col in ["dice", "iou", "precision", "recall", "f1", "pred_pixels", "pred_ratio"]:
            out.loc[zero_mask, col] = zeroed[col].values

    return out


def summarize(df: pd.DataFrame) -> dict:
    subsets = {
        "all": df,
        "non_empty": df[df["target_ratio"] > 0],
        "useful_5_75": df[(df["target_ratio"] >= 0.05) & (df["target_ratio"] < 0.75)],
        "small_1_5": df[(df["target_ratio"] >= 0.01) & (df["target_ratio"] < 0.05)],
        "very_small_0_1": df[(df["target_ratio"] > 0) & (df["target_ratio"] < 0.01)],
        "empty": df[df["target_ratio"] == 0],
        "empty_false_positive_original": df[
            (df["target_ratio"] == 0)
            & (df.get("original_pred_ratio", df["pred_ratio"]) > 0)
        ],
        "very_large_75_90": df[(df["target_ratio"] >= 0.75) & (df["target_ratio"] < 0.90)],
        "almost_full_90_100": df[df["target_ratio"] >= 0.90],
    }

    row = {
        "zero_count": int((df["calib_model"] == "zero").sum()) if "calib_model" in df.columns else 0,
        "keep_count": int((df["calib_model"] == "keep").sum()) if "calib_model" in df.columns else len(df),
    }

    for name, sub in subsets.items():
        row[f"{name}_count"] = int(len(sub))

        for metric in ["dice", "iou", "precision", "recall", "f1"]:
            row[f"{name}_{metric}"] = float(sub[metric].mean()) if len(sub) else float("nan")

        row[f"{name}_pred_ratio_mean"] = float(sub["pred_ratio"].mean()) if len(sub) else float("nan")

    return row


def score(row: dict, mode: str) -> float:
    if mode == "balanced":
        return (
            row["all_dice"]
            + 0.45 * row["useful_5_75_dice"]
            + 0.15 * row["non_empty_dice"]
            + 0.10 * row["empty_dice"]
            + 0.05 * row["small_1_5_dice"]
        )

    if mode == "global":
        return row["all_dice"]

    if mode == "operational":
        return (
            row["all_dice"]
            + 0.35 * row["useful_5_75_dice"]
            + 0.20 * row["empty_dice"]
            + 0.10 * row["non_empty_dice"]
        )

    raise ValueError(f"score_mode inválido: {mode}")


def parse_float_list(text: str) -> list[float | None]:
    vals: list[float | None] = []

    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        if x.lower() in {"none", "null"}:
            vals.append(None)
        else:
            vals.append(float(x))

    return vals


def tune(val_df: pd.DataFrame, args) -> tuple[pd.DataFrame, dict]:
    low_grid = parse_float_list(args.low_zero_grid)
    window_low_grid = parse_float_list(args.window_low_grid)
    window_high_grid = parse_float_list(args.window_high_grid)

    rows = []

    base = val_df.copy()
    base["calib_model"] = "keep"
    base_summary = summarize(base)
    base_summary["low_zero_thr"] = None
    base_summary["window_low"] = None
    base_summary["window_high"] = None
    base_summary["score"] = score(base_summary, args.score_mode)
    rows.append(base_summary)

    for low_thr in low_grid:
        for w_low in window_low_grid:
            for w_high in window_high_grid:
                if w_low is not None and w_high is not None and w_low >= w_high:
                    continue

                if low_thr is None and (w_low is None or w_high is None):
                    continue

                calibrated = apply_rule(
                    val_df,
                    low_zero_thr=low_thr,
                    window_low=w_low,
                    window_high=w_high,
                )

                row = summarize(calibrated)
                row["low_zero_thr"] = low_thr
                row["window_low"] = w_low
                row["window_high"] = w_high
                row["score"] = score(row, args.score_mode)
                rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["score", "all_dice", "useful_5_75_dice", "empty_dice"],
        ascending=False,
    )

    best = ranking.iloc[0].to_dict()
    return ranking, best


def main():
    parser = argparse.ArgumentParser(description="Tuna calibrador S2 por pred_ratio em VAL e aplica em TEST.")
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--low-zero-grid",
        dest="low_zero_grid",
        type=str,
        default="none,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02",
    )
    parser.add_argument(
        "--window-low-grid",
        dest="window_low_grid",
        type=str,
        default="none,0.015,0.02,0.03,0.04,0.05,0.06",
    )
    parser.add_argument(
        "--window-high-grid",
        dest="window_high_grid",
        type=str,
        default="none,0.04,0.05,0.06,0.075,0.10,0.125",
    )
    parser.add_argument("--score-mode", type=str, default="balanced", choices=["balanced", "global", "operational"])

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_df = load_metrics(args.val_csv)
    test_df = load_metrics(args.test_csv)

    val_df["original_pred_ratio"] = val_df["pred_ratio"]
    test_df["original_pred_ratio"] = test_df["pred_ratio"]

    ranking, best = tune(val_df, args)
    ranking.to_csv(args.output_dir / "s2_calibrator_val_ranking.csv", index=False)

    low_thr = best["low_zero_thr"]
    w_low = best["window_low"]
    w_high = best["window_high"]

    low_thr = None if pd.isna(low_thr) else float(low_thr)
    w_low = None if pd.isna(w_low) else float(w_low)
    w_high = None if pd.isna(w_high) else float(w_high)

    val_best = apply_rule(val_df, low_zero_thr=low_thr, window_low=w_low, window_high=w_high)
    test_best = apply_rule(test_df, low_zero_thr=low_thr, window_low=w_low, window_high=w_high)

    val_summary = summarize(val_best)
    test_summary = summarize(test_best)

    val_best.to_csv(args.output_dir / "s2_calibrator_val_best_per_sample.csv", index=False)
    test_best.to_csv(args.output_dir / "s2_calibrator_test_final_per_sample.csv", index=False)

    final = {
        "chosen_on": "val",
        "score_mode": args.score_mode,
        "rule": {
            "low_zero_thr": low_thr,
            "window_low": w_low,
            "window_high": w_high,
        },
        "val_summary": val_summary,
        "test_summary": test_summary,
    }

    (args.output_dir / "s2_calibrator_final_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nMelhor regra escolhida na VAL:")
    print(json.dumps(final["rule"], ensure_ascii=False, indent=2))
    print("score_mode:", args.score_mode)

    cols = [
        "low_zero_thr",
        "window_low",
        "window_high",
        "score",
        "all_dice",
        "non_empty_dice",
        "useful_5_75_dice",
        "small_1_5_dice",
        "very_small_0_1_dice",
        "empty_dice",
        "zero_count",
    ]

    print("\nTop 30 VAL:")
    print(ranking[cols].head(30).to_string(index=False))

    print("\nResumo TEST:")
    rows = []
    for subset in [
        "all",
        "non_empty",
        "useful_5_75",
        "small_1_5",
        "very_small_0_1",
        "empty",
        "very_large_75_90",
        "almost_full_90_100",
    ]:
        rows.append({
            "subset": subset,
            "count": test_summary[f"{subset}_count"],
            "dice": test_summary[f"{subset}_dice"],
            "iou": test_summary[f"{subset}_iou"],
            "precision": test_summary[f"{subset}_precision"],
            "recall": test_summary[f"{subset}_recall"],
            "f1": test_summary[f"{subset}_f1"],
            "pred_ratio_mean": test_summary[f"{subset}_pred_ratio_mean"],
        })

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "s2_calibrator_test_final_by_subset.csv", index=False)
    print(table.to_string(index=False))

    print("\nArquivos salvos em:", args.output_dir)


if __name__ == "__main__":
    main()

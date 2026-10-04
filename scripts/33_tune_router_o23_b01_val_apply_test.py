from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_pair(o23_csv: Path, b01_csv: Path) -> pd.DataFrame:
    o23 = pd.read_csv(o23_csv)
    b01 = pd.read_csv(b01_csv)

    o23 = o23.rename(columns={
        "dice": "o23_dice",
        "iou": "o23_iou",
        "precision": "o23_precision",
        "recall": "o23_recall",
        "f1": "o23_f1",
    })

    b01 = b01.rename(columns={
        "area_dice": "b01_dice",
        "area_iou": "b01_iou",
        "area_precision": "b01_precision",
        "area_recall": "b01_recall",
        "area_f1": "b01_f1",
    })

    cols_o23 = [
        "image_path",
        "o23_dice",
        "o23_iou",
        "o23_precision",
        "o23_recall",
        "o23_f1",
        "target_ratio",
        "pred_ratio",
        "target_pixels",
        "pred_pixels",
        "mask_size_class",
    ]

    cols_b01 = [
        "image_path",
        "b01_dice",
        "b01_iou",
        "b01_precision",
        "b01_recall",
        "b01_f1",
    ]

    for extra in ["boundary_f1", "boundary_dice", "boundary_iou"]:
        if extra in b01.columns:
            cols_b01.append(extra)

    df = o23[cols_o23].merge(b01[cols_b01], on="image_path", how="inner")

    if len(df) == 0:
        raise RuntimeError("Merge vazio entre O23 e B01. Verifique image_path nos CSVs.")

    return df


def apply_router(df: pd.DataFrame, small_thr: float, full_thr: float) -> pd.DataFrame:
    out = df.copy()

    use_b01 = (out["pred_ratio"] >= small_thr) & (out["pred_ratio"] < full_thr)

    out["router_model"] = np.where(use_b01, "b01", "o23")
    out["router_dice"] = np.where(use_b01, out["b01_dice"], out["o23_dice"])
    out["router_iou"] = np.where(use_b01, out["b01_iou"], out["o23_iou"])
    out["router_precision"] = np.where(use_b01, out["b01_precision"], out["o23_precision"])
    out["router_recall"] = np.where(use_b01, out["b01_recall"], out["o23_recall"])
    out["router_f1"] = np.where(use_b01, out["b01_f1"], out["o23_f1"])

    return out


def summarize(out: pd.DataFrame) -> dict:
    subsets = {
        "all": out,
        "non_empty": out[out["target_ratio"] > 0],
        "useful_5_75": out[(out["target_ratio"] >= 0.05) & (out["target_ratio"] < 0.75)],
        "small_1_5": out[(out["target_ratio"] >= 0.01) & (out["target_ratio"] < 0.05)],
        "very_small_0_1": out[(out["target_ratio"] > 0) & (out["target_ratio"] < 0.01)],
        "empty": out[out["target_ratio"] == 0],
        "almost_full_90_100": out[out["target_ratio"] >= 0.90],
    }

    row = {
        "use_b01_count": int((out["router_model"] == "b01").sum()),
        "use_o23_count": int((out["router_model"] == "o23").sum()),
    }

    for name, sub in subsets.items():
        row[f"{name}_count"] = int(len(sub))

        for metric in ["dice", "iou", "precision", "recall", "f1"]:
            row[f"{name}_{metric}"] = float(sub[f"router_{metric}"].mean()) if len(sub) else float("nan")

        row[f"{name}_o23_dice"] = float(sub["o23_dice"].mean()) if len(sub) else float("nan")
        row[f"{name}_b01_dice"] = float(sub["b01_dice"].mean()) if len(sub) else float("nan")

    return row


def score(row: dict, mode: str) -> float:
    if mode == "balanced":
        return (
            row["all_dice"]
            + 0.50 * row["useful_5_75_dice"]
            + 0.20 * row["non_empty_dice"]
            + 0.05 * row["small_1_5_dice"]
        )

    if mode == "global":
        return row["all_dice"]

    if mode == "useful":
        return (
            row["useful_5_75_dice"]
            + 0.20 * row["non_empty_dice"]
            + 0.10 * row["small_1_5_dice"]
        )

    raise ValueError(f"score_mode inválido: {mode}")


def tune_router(
    val_df: pd.DataFrame,
    small_grid: list[float],
    full_grid: list[float],
    score_mode: str,
) -> tuple[pd.DataFrame, dict]:
    rows = []

    for small_thr in small_grid:
        for full_thr in full_grid:
            if small_thr >= full_thr:
                continue

            out = apply_router(val_df, small_thr=small_thr, full_thr=full_thr)
            row = summarize(out)
            row["small_thr"] = small_thr
            row["full_thr"] = full_thr
            row["score"] = score(row, score_mode)
            rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["score", "all_dice", "useful_5_75_dice"],
        ascending=False,
    )

    best = ranking.iloc[0].to_dict()
    return ranking, best


def main():
    parser = argparse.ArgumentParser(description="Tuna roteador O23/B01 em VAL e aplica no TEST.")
    parser.add_argument("--o23-val-csv", type=Path, required=True)
    parser.add_argument("--b01-val-csv", type=Path, required=True)
    parser.add_argument("--o23-test-csv", type=Path, required=True)
    parser.add_argument("--b01-test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--small-grid", type=str, default="0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05")
    parser.add_argument("--full-grid", type=str, default="0.70,0.75,0.80,0.85,0.90,0.95,1.01")
    parser.add_argument("--score-mode", type=str, default="balanced", choices=["balanced", "global", "useful"])

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    small_grid = [float(x.strip()) for x in args.small_grid.split(",") if x.strip()]
    full_grid = [float(x.strip()) for x in args.full_grid.split(",") if x.strip()]

    val_df = load_pair(args.o23_val_csv, args.b01_val_csv)
    test_df = load_pair(args.o23_test_csv, args.b01_test_csv)

    ranking, best = tune_router(
        val_df=val_df,
        small_grid=small_grid,
        full_grid=full_grid,
        score_mode=args.score_mode,
    )

    ranking.to_csv(args.output_dir / "router_val_ranking.csv", index=False)

    small_thr = float(best["small_thr"])
    full_thr = float(best["full_thr"])

    val_best = apply_router(val_df, small_thr=small_thr, full_thr=full_thr)
    test_best = apply_router(test_df, small_thr=small_thr, full_thr=full_thr)

    val_summary = summarize(val_best)
    test_summary = summarize(test_best)

    val_best.to_csv(args.output_dir / "router_val_best_per_sample.csv", index=False)
    test_best.to_csv(args.output_dir / "router_test_final_per_sample.csv", index=False)

    final = {
        "chosen_on": "val",
        "score_mode": args.score_mode,
        "small_thr": small_thr,
        "full_thr": full_thr,
        "val_summary": val_summary,
        "test_summary": test_summary,
    }

    (args.output_dir / "router_final_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nMelhor regra escolhida na VAL:")
    print(f"small_thr = {small_thr}")
    print(f"full_thr  = {full_thr}")
    print(f"score_mode = {args.score_mode}")

    print("\nTop 20 VAL:")
    cols = [
        "small_thr",
        "full_thr",
        "score",
        "all_dice",
        "non_empty_dice",
        "useful_5_75_dice",
        "small_1_5_dice",
        "very_small_0_1_dice",
        "empty_dice",
        "almost_full_90_100_dice",
        "use_b01_count",
    ]
    print(ranking[cols].head(20).to_string(index=False))

    print("\nResumo TEST com regra fixa da VAL:")
    test_rows = []
    for subset in ["all", "non_empty", "useful_5_75", "small_1_5", "very_small_0_1", "empty", "almost_full_90_100"]:
        test_rows.append({
            "subset": subset,
            "count": test_summary[f"{subset}_count"],
            "router_dice": test_summary[f"{subset}_dice"],
            "o23_dice": test_summary[f"{subset}_o23_dice"],
            "b01_dice": test_summary[f"{subset}_b01_dice"],
            "router_iou": test_summary[f"{subset}_iou"],
            "router_f1": test_summary[f"{subset}_f1"],
        })

    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(args.output_dir / "router_test_final_by_subset.csv", index=False)
    print(test_table.to_string(index=False))

    print("\nArquivos salvos em:", args.output_dir)


if __name__ == "__main__":
    main()

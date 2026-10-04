from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


def pick_col(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]

    for c in cols:
        lc = c.lower()
        if kind == "prob" and ("prob" in lc or "score" in lc) and ("empty" in lc or "gate" in lc or "pred" in lc):
            return c
        if kind == "path" and ("image" in lc and "path" in lc):
            return c
        if kind == "label" and ("empty" in lc or "label" in lc or "target" in lc or "y_true" in lc):
            if "prob" not in lc and "score" not in lc and "pred" not in lc:
                return c

    raise ValueError(
        f"Não consegui detectar coluna {kind}. Colunas disponíveis: {cols}\n"
        f"Passe manualmente com --gate-{kind}-col."
    )


def normalize_path_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("\\", "/", regex=False)


def load_gate(
    path: Path,
    path_col: str | None,
    prob_col: str | None,
    label_col: str | None,
) -> tuple[pd.DataFrame, dict]:
    df = read_csv(path)

    detected = {}

    path_col = path_col or pick_col(
        df,
        ["image_path", "path", "file_path", "img_path", "filename", "image"],
        "path",
    )

    prob_col = prob_col or pick_col(
        df,
        ["prob_empty", "empty_prob", "p_empty", "gate_prob", "probability", "pred_prob", "y_prob", "score"],
        "prob",
    )

    if label_col is None:
        try:
            label_col = pick_col(
                df,
                ["is_empty", "target_empty", "empty_label", "label", "y_true", "target"],
                "label",
            )
        except Exception:
            label_col = None

    detected["path_col"] = path_col
    detected["prob_col"] = prob_col
    detected["label_col"] = label_col

    out = pd.DataFrame()
    out["image_path"] = normalize_path_key(df[path_col])
    out["gate_empty_prob"] = df[prob_col].astype(float)

    if label_col is not None:
        out["gate_label_raw"] = df[label_col]

    return out, detected


def load_seg_metrics(path: Path) -> pd.DataFrame:
    df = read_csv(path)
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
        raise ValueError(f"CSV de segmentação sem colunas necessárias: {missing}")

    df = df.copy()
    df["image_path"] = normalize_path_key(df["image_path"])
    return df


def zero_metrics(df: pd.DataFrame) -> pd.DataFrame:
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


def merge_seg_gate(seg: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    df = seg.merge(gate, on="image_path", how="left")

    if df["gate_empty_prob"].isna().mean() > 0.20:
        seg2 = seg.copy()
        gate2 = gate.copy()
        seg2["_base"] = seg2["image_path"].map(lambda x: Path(str(x)).name)
        gate2["_base"] = gate2["image_path"].map(lambda x: Path(str(x)).name)

        df2 = seg2.merge(
            gate2.drop(columns=["image_path"]).drop_duplicates("_base"),
            on="_base",
            how="left",
        ).drop(columns=["_base"])

        if df2["gate_empty_prob"].isna().mean() < df["gate_empty_prob"].isna().mean():
            df = df2

    missing_rate = df["gate_empty_prob"].isna().mean()
    if missing_rate > 0:
        print(f"[AVISO] {missing_rate:.2%} das amostras não encontraram probabilidade do gate. Elas serão mantidas.")

    return df


def apply_gate(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()

    gate_available = out["gate_empty_prob"].notna()
    gate_empty = gate_available & (out["gate_empty_prob"] >= threshold)

    out["gate_action"] = np.where(gate_empty, "zero_by_gate", "keep_segmentation")

    if gate_empty.any():
        zeroed = zero_metrics(out.loc[gate_empty].copy())
        for col in ["dice", "iou", "precision", "recall", "f1", "pred_pixels", "pred_ratio"]:
            out.loc[gate_empty, col] = zeroed[col].values

    return out


def summarize(df: pd.DataFrame) -> dict:
    subsets = {
        "all": df,
        "non_empty": df[df["target_ratio"] > 0],
        "useful_5_75": df[(df["target_ratio"] >= 0.05) & (df["target_ratio"] < 0.75)],
        "small_1_5": df[(df["target_ratio"] >= 0.01) & (df["target_ratio"] < 0.05)],
        "very_small_0_1": df[(df["target_ratio"] > 0) & (df["target_ratio"] < 0.01)],
        "empty": df[df["target_ratio"] == 0],
        "very_large_75_90": df[(df["target_ratio"] >= 0.75) & (df["target_ratio"] < 0.90)],
        "almost_full_90_100": df[df["target_ratio"] >= 0.90],
    }

    row = {}
    row["gate_zero_count"] = int((df["gate_action"] == "zero_by_gate").sum()) if "gate_action" in df.columns else 0
    row["gate_zero_non_empty_count"] = int(((df["gate_action"] == "zero_by_gate") & (df["target_ratio"] > 0)).sum()) if "gate_action" in df.columns else 0
    row["gate_zero_empty_count"] = int(((df["gate_action"] == "zero_by_gate") & (df["target_ratio"] == 0)).sum()) if "gate_action" in df.columns else 0
    row["false_empty_rate_non_empty"] = row["gate_zero_non_empty_count"] / max(int((df["target_ratio"] > 0).sum()), 1)

    for name, sub in subsets.items():
        row[f"{name}_count"] = int(len(sub))
        for metric in ["dice", "iou", "precision", "recall", "f1"]:
            row[f"{name}_{metric}"] = float(sub[metric].mean()) if len(sub) else float("nan")
        row[f"{name}_pred_ratio_mean"] = float(sub["pred_ratio"].mean()) if len(sub) else float("nan")

    return row


def score(row: dict, mode: str) -> float:
    if mode == "safe":
        return (
            row["all_dice"]
            + 0.35 * row["useful_5_75_dice"]
            + 0.20 * row["empty_dice"]
            + 0.10 * row["non_empty_dice"]
            + 0.05 * row["small_1_5_dice"]
            - 5.0 * row["false_empty_rate_non_empty"]
        )

    if mode == "global":
        return row["all_dice"] - 5.0 * row["false_empty_rate_non_empty"]

    if mode == "empty_safe":
        return (
            row["empty_dice"]
            + 0.50 * row["all_dice"]
            + 0.25 * row["useful_5_75_dice"]
            - 8.0 * row["false_empty_rate_non_empty"]
        )

    raise ValueError(f"score_mode inválido: {mode}")


def parse_thresholds(text: str) -> list[float]:
    if ":" in text:
        start, stop, step = [float(x) for x in text.split(":")]
        vals = []
        x = start
        while x <= stop + 1e-12:
            vals.append(round(x, 8))
            x += step
        return vals

    return [float(x.strip()) for x in text.split(",") if x.strip()]


def tune_threshold(
    val_df: pd.DataFrame,
    thresholds: list[float],
    max_false_empty_rate: float,
    score_mode: str,
) -> tuple[pd.DataFrame, dict]:
    rows = []

    base = val_df.copy()
    base["gate_action"] = "keep_segmentation"
    base_sum = summarize(base)
    base_sum["threshold"] = float("inf")
    base_sum["score"] = score(base_sum, score_mode)
    base_sum["valid_under_constraint"] = True
    rows.append(base_sum)

    for thr in thresholds:
        gated = apply_gate(val_df, threshold=thr)
        row = summarize(gated)
        row["threshold"] = thr
        row["score"] = score(row, score_mode)
        row["valid_under_constraint"] = row["false_empty_rate_non_empty"] <= max_false_empty_rate
        rows.append(row)

    ranking = pd.DataFrame(rows)

    valid = ranking[ranking["valid_under_constraint"]].copy()
    if len(valid) == 0:
        valid = ranking.copy()

    valid = valid.sort_values(
        ["score", "all_dice", "empty_dice", "useful_5_75_dice"],
        ascending=False,
    )

    return ranking.sort_values(["score"], ascending=False), valid.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser(description="Tuna empty gate S2 em VAL e aplica no TEST.")
    parser.add_argument("--seg-val-csv", type=Path, required=True)
    parser.add_argument("--seg-test-csv", type=Path, required=True)
    parser.add_argument("--gate-val-csv", type=Path, required=True)
    parser.add_argument("--gate-test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--thresholds", type=str, default="0.50:0.995:0.005")
    parser.add_argument("--max-false-empty-rate", type=float, default=0.005)
    parser.add_argument("--score-mode", type=str, default="safe", choices=["safe", "global", "empty_safe"])

    parser.add_argument("--gate-path-col", type=str, default=None)
    parser.add_argument("--gate-prob-col", type=str, default=None)
    parser.add_argument("--gate-label-col", type=str, default=None)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seg_val = load_seg_metrics(args.seg_val_csv)
    seg_test = load_seg_metrics(args.seg_test_csv)

    gate_val, detected_val = load_gate(
        args.gate_val_csv,
        path_col=args.gate_path_col,
        prob_col=args.gate_prob_col,
        label_col=args.gate_label_col,
    )
    gate_test, detected_test = load_gate(
        args.gate_test_csv,
        path_col=args.gate_path_col,
        prob_col=args.gate_prob_col,
        label_col=args.gate_label_col,
    )

    val_df = merge_seg_gate(seg_val, gate_val)
    test_df = merge_seg_gate(seg_test, gate_test)

    thresholds = parse_thresholds(args.thresholds)

    ranking, best = tune_threshold(
        val_df=val_df,
        thresholds=thresholds,
        max_false_empty_rate=args.max_false_empty_rate,
        score_mode=args.score_mode,
    )

    ranking.to_csv(args.output_dir / "s2_empty_gate_val_threshold_ranking.csv", index=False)

    best_thr = float(best["threshold"])

    val_best = apply_gate(val_df, threshold=best_thr)
    test_best = apply_gate(test_df, threshold=best_thr)

    val_summary = summarize(val_best)
    test_summary = summarize(test_best)

    val_best.to_csv(args.output_dir / "s2_empty_gate_val_best_per_sample.csv", index=False)
    test_best.to_csv(args.output_dir / "s2_empty_gate_test_final_per_sample.csv", index=False)

    final = {
        "chosen_on": "val",
        "threshold": best_thr,
        "max_false_empty_rate": args.max_false_empty_rate,
        "score_mode": args.score_mode,
        "detected_columns": {
            "val": detected_val,
            "test": detected_test,
        },
        "val_summary": val_summary,
        "test_summary": test_summary,
    }

    (args.output_dir / "s2_empty_gate_final_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nColunas detectadas:")
    print(json.dumps(final["detected_columns"], ensure_ascii=False, indent=2))

    print("\nMelhor threshold escolhido na VAL:")
    print(best_thr)

    cols = [
        "threshold",
        "valid_under_constraint",
        "score",
        "all_dice",
        "non_empty_dice",
        "useful_5_75_dice",
        "small_1_5_dice",
        "very_small_0_1_dice",
        "empty_dice",
        "gate_zero_count",
        "gate_zero_empty_count",
        "gate_zero_non_empty_count",
        "false_empty_rate_non_empty",
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
    table.to_csv(args.output_dir / "s2_empty_gate_test_final_by_subset.csv", index=False)
    print(table.to_string(index=False))

    print("\nArquivos salvos em:", args.output_dir)


if __name__ == "__main__":
    main()

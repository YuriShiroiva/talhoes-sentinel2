from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Script 36
# S2-FP01: false-positive suppressor tabular
#
# Objetivo:
#   aprender quando a máscara Sentinel-2 parece ser falso positivo.
#
# Usa apenas features operacionais:
#   - pred_ratio do ensemble S2
#   - prob_empty do empty gate existente
#   - combinações simples dessas duas variáveis
#
# Label de treino:
#   1 = target vazio e predição não vazia  -> empty false positive
#   0 = demais casos
#
# Aplicação:
#   se prob_fp >= threshold:
#       zera a máscara
#   senão:
#       mantém a segmentação
#
# Forma correta:
#   train: ajusta o classificador
#   val: escolhe threshold
#   test: aplica uma única vez
#
# Modo de diagnóstico:
#   se train não for fornecido, treina na própria VAL e aplica no TEST.
#   Esse modo é útil para testar potencial, mas não deve ser vendido como final.
# ============================================================


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


def normalize_path(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("\\", "/", regex=False)


def find_prob_col(df: pd.DataFrame) -> str:
    candidates = ["prob_empty", "empty_prob", "p_empty", "gate_prob", "probability", "pred_prob", "score"]
    lower = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    for c in df.columns:
        lc = c.lower()
        if ("prob" in lc or "score" in lc) and ("empty" in lc or "gate" in lc or "pred" in lc):
            return c

    raise ValueError(f"Não encontrei coluna de probabilidade no gate. Colunas: {list(df.columns)}")


def load_seg(path: Path) -> pd.DataFrame:
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
    df["image_path"] = normalize_path(df["image_path"])
    return df


def load_gate(path: Path, prob_col: str | None = None) -> pd.DataFrame:
    df = read_csv(path)

    if "image_path" not in df.columns:
        raise ValueError(f"CSV do gate precisa ter image_path. Colunas: {list(df.columns)}")

    prob_col = prob_col or find_prob_col(df)

    out = pd.DataFrame()
    out["image_path"] = normalize_path(df["image_path"])
    out["prob_empty"] = df[prob_col].astype(float)

    return out


def merge_seg_gate(seg: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    merged = seg.merge(gate, on="image_path", how="left")

    if merged["prob_empty"].isna().mean() > 0.20:
        seg2 = seg.copy()
        gate2 = gate.copy()

        seg2["_base"] = seg2["image_path"].map(lambda x: Path(str(x)).name)
        gate2["_base"] = gate2["image_path"].map(lambda x: Path(str(x)).name)

        merged2 = seg2.merge(
            gate2.drop(columns=["image_path"]).drop_duplicates("_base"),
            on="_base",
            how="left",
        ).drop(columns=["_base"])

        if merged2["prob_empty"].isna().mean() < merged["prob_empty"].isna().mean():
            merged = merged2

    missing = int(merged["prob_empty"].isna().sum())
    if missing:
        print(f"[AVISO] {missing} amostras sem prob_empty. Preenchendo com 0.")
        merged["prob_empty"] = merged["prob_empty"].fillna(0.0)

    return merged


def add_group_and_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["group"] = "other"
    out.loc[(out["target_ratio"] == 0) & (out["pred_pixels"] == 0), "group"] = "empty_correct"
    out.loc[(out["target_ratio"] == 0) & (out["pred_pixels"] > 0), "group"] = "empty_false_positive"
    out.loc[(out["target_ratio"] > 0) & (out["target_ratio"] < 0.01), "group"] = "very_small_0_1"
    out.loc[(out["target_ratio"] >= 0.01) & (out["target_ratio"] < 0.05), "group"] = "small_1_5"
    out.loc[(out["target_ratio"] >= 0.05) & (out["target_ratio"] < 0.75), "group"] = "useful_5_75"
    out.loc[(out["target_ratio"] >= 0.75) & (out["target_ratio"] < 0.90), "group"] = "very_large_75_90"
    out.loc[out["target_ratio"] >= 0.90, "group"] = "almost_full_90_100"

    out["label_empty_fp"] = ((out["target_ratio"] == 0) & (out["pred_pixels"] > 0)).astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    pred = df["pred_ratio"].astype(float).clip(lower=0.0, upper=1.0)
    prob = df["prob_empty"].astype(float).clip(lower=0.0, upper=1.0)

    eps = 1e-8

    X = pd.DataFrame(index=df.index)
    X["pred_ratio"] = pred
    X["prob_empty"] = prob
    X["log_pred_ratio"] = np.log1p(pred * 1000.0)
    X["prob_x_pred"] = prob * pred
    X["prob_div_pred"] = prob / (pred + eps)
    X["one_minus_pred"] = 1.0 - pred
    X["empty_conf_minus_pred"] = prob - pred
    X["small_pred_flag"] = (pred < 0.08).astype(float)
    X["tiny_pred_flag"] = (pred < 0.02).astype(float)
    X["mid_fp_suspect"] = ((pred >= 0.005) & (pred < 0.20)).astype(float)

    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return X


def zero_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    eps = 1e-7
    is_empty_target = out["target_pixels"] == 0
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


def apply_suppressor(df: pd.DataFrame, prob_fp: np.ndarray, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["prob_empty_fp"] = prob_fp

    suppress = out["prob_empty_fp"].values >= threshold
    out["suppress_action"] = np.where(suppress, "zero_by_fp_suppressor", "keep_segmentation")

    if suppress.any():
        zeroed = zero_metrics(out.loc[suppress].copy())
        for col in ["dice", "iou", "precision", "recall", "f1", "pred_pixels", "pred_ratio"]:
            out.loc[suppress, col] = zeroed[col].values

    return out


def summarize(df: pd.DataFrame) -> dict:
    subsets = {
        "all": df,
        "non_empty": df[df["target_ratio"] > 0],
        "useful_5_75": df[(df["target_ratio"] >= 0.05) & (df["target_ratio"] < 0.75)],
        "small_1_5": df[(df["target_ratio"] >= 0.01) & (df["target_ratio"] < 0.05)],
        "very_small_0_1": df[(df["target_ratio"] > 0) & (df["target_ratio"] < 0.01)],
        "empty": df[df["target_ratio"] == 0],
        "empty_false_positive_original": df[df["group"] == "empty_false_positive"],
        "very_large_75_90": df[(df["target_ratio"] >= 0.75) & (df["target_ratio"] < 0.90)],
        "almost_full_90_100": df[df["target_ratio"] >= 0.90],
    }

    row = {}

    row["zero_count"] = int((df["suppress_action"] == "zero_by_fp_suppressor").sum()) if "suppress_action" in df.columns else 0
    row["zero_non_empty_count"] = int(((df["suppress_action"] == "zero_by_fp_suppressor") & (df["target_ratio"] > 0)).sum()) if "suppress_action" in df.columns else 0
    row["zero_empty_fp_count"] = int(((df["suppress_action"] == "zero_by_fp_suppressor") & (df["group"] == "empty_false_positive")).sum()) if "suppress_action" in df.columns else 0

    non_empty_total = max(int((df["target_ratio"] > 0).sum()), 1)
    empty_fp_total = max(int((df["group"] == "empty_false_positive").sum()), 1)

    row["false_empty_rate_non_empty"] = row["zero_non_empty_count"] / non_empty_total
    row["empty_fp_recall"] = row["zero_empty_fp_count"] / empty_fp_total

    for name, sub in subsets.items():
        row[f"{name}_count"] = int(len(sub))

        for metric in ["dice", "iou", "precision", "recall", "f1"]:
            row[f"{name}_{metric}"] = float(sub[metric].mean()) if len(sub) else float("nan")

        row[f"{name}_pred_ratio_mean"] = float(sub["pred_ratio"].mean()) if len(sub) else float("nan")

    return row


def objective(row: dict, mode: str) -> float:
    if mode == "safe":
        return (
            row["all_dice"]
            + 0.35 * row["useful_5_75_dice"]
            + 0.20 * row["empty_dice"]
            + 0.10 * row["non_empty_dice"]
            + 0.05 * row["small_1_5_dice"]
            - 6.0 * row["false_empty_rate_non_empty"]
        )

    if mode == "global":
        return row["all_dice"] - 6.0 * row["false_empty_rate_non_empty"]

    if mode == "empty_gain":
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
            vals.append(round(x, 10))
            x += step
        return vals

    return [float(x.strip()) for x in text.split(",") if x.strip()]


def train_model(train_df: pd.DataFrame, model_type: str, seed: int):
    X_train = build_features(train_df)
    y_train = train_df["label_empty_fp"].astype(int).values

    if y_train.sum() == 0:
        raise RuntimeError("Não há exemplos positivos de empty_false_positive no conjunto de treino/dev.")

    try:
        if model_type == "logreg":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            )
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
        elif model_type == "gb":
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.03,
                max_depth=2,
                random_state=seed,
            )
        else:
            raise ValueError(f"model_type inválido: {model_type}")

        model.fit(X_train, y_train)

        return model

    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn não está instalado nesse ambiente. "
            "Instale com `pip install scikit-learn` ou rode em ambiente que já tenha sklearn."
        ) from exc


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    X = build_features(df)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def tune_threshold(
    val_df: pd.DataFrame,
    prob_val: np.ndarray,
    thresholds: list[float],
    max_false_empty_rate: float,
    max_small_kill_rate: float,
    score_mode: str,
) -> tuple[pd.DataFrame, dict]:
    rows = []

    base = val_df.copy()
    base["prob_empty_fp"] = prob_val
    base["suppress_action"] = "keep_segmentation"
    base_sum = summarize(base)
    base_sum["threshold"] = float("inf")
    base_sum["score"] = objective(base_sum, score_mode)
    base_sum["valid_under_constraint"] = True
    rows.append(base_sum)

    small_total = max(int(val_df["group"].isin(["very_small_0_1", "small_1_5"]).sum()), 1)

    for thr in thresholds:
        pred = apply_suppressor(val_df, prob_fp=prob_val, threshold=thr)
        row = summarize(pred)
        row["threshold"] = thr

        small_killed = int(
            (
                (pred["suppress_action"] == "zero_by_fp_suppressor")
                & pred["group"].isin(["very_small_0_1", "small_1_5"])
            ).sum()
        )
        row["small_kill_rate"] = small_killed / small_total

        row["score"] = objective(row, score_mode)
        row["valid_under_constraint"] = (
            row["false_empty_rate_non_empty"] <= max_false_empty_rate
            and row["small_kill_rate"] <= max_small_kill_rate
        )
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


def make_subset_table(summary: dict) -> pd.DataFrame:
    rows = []
    for subset in [
        "all",
        "non_empty",
        "useful_5_75",
        "small_1_5",
        "very_small_0_1",
        "empty",
        "empty_false_positive_original",
        "very_large_75_90",
        "almost_full_90_100",
    ]:
        rows.append({
            "subset": subset,
            "count": summary[f"{subset}_count"],
            "dice": summary[f"{subset}_dice"],
            "iou": summary[f"{subset}_iou"],
            "precision": summary[f"{subset}_precision"],
            "recall": summary[f"{subset}_recall"],
            "f1": summary[f"{subset}_f1"],
            "pred_ratio_mean": summary[f"{subset}_pred_ratio_mean"],
        })
    return pd.DataFrame(rows)


def save_feature_importance(model, output_dir: Path, feature_names: list[str]) -> None:
    rows = []

    if hasattr(model, "feature_importances_"):
        for name, val in zip(feature_names, model.feature_importances_):
            rows.append({"feature": name, "importance": float(val)})

    elif hasattr(model, "coef_"):
        for name, val in zip(feature_names, model.coef_.ravel()):
            rows.append({"feature": name, "coef": float(val), "abs_coef": float(abs(val))})

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "feature_importance.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="S2-FP01 false-positive suppressor tabular.")
    parser.add_argument("--train-seg-csv", type=Path, default=None)
    parser.add_argument("--train-gate-csv", type=Path, default=None)
    parser.add_argument("--val-seg-csv", type=Path, required=True)
    parser.add_argument("--val-gate-csv", type=Path, required=True)
    parser.add_argument("--test-seg-csv", type=Path, required=True)
    parser.add_argument("--test-gate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--gate-prob-col", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="logreg", choices=["logreg", "rf", "gb"])
    parser.add_argument("--thresholds", type=str, default="0.05:0.995:0.005")
    parser.add_argument("--max-false-empty-rate", type=float, default=0.005)
    parser.add_argument("--max-small-kill-rate", type=float, default=0.01)
    parser.add_argument("--score-mode", type=str, default="safe", choices=["safe", "global", "empty_gain"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_df = add_group_and_label(
        merge_seg_gate(
            load_seg(args.val_seg_csv),
            load_gate(args.val_gate_csv, prob_col=args.gate_prob_col),
        )
    )

    test_df = add_group_and_label(
        merge_seg_gate(
            load_seg(args.test_seg_csv),
            load_gate(args.test_gate_csv, prob_col=args.gate_prob_col),
        )
    )

    diagnostic_mode = args.train_seg_csv is None or args.train_gate_csv is None

    if diagnostic_mode:
        print("[AVISO] Rodando em modo diagnóstico: treinando no próprio VAL.")
        print("[AVISO] Use isso para avaliar potencial. Para resultado final, forneça train-seg-csv e train-gate-csv.")
        train_df = val_df.copy()
    else:
        train_df = add_group_and_label(
            merge_seg_gate(
                load_seg(args.train_seg_csv),
                load_gate(args.train_gate_csv, prob_col=args.gate_prob_col),
            )
        )

    config = {
        "diagnostic_mode_train_on_val": diagnostic_mode,
        "model_type": args.model_type,
        "score_mode": args.score_mode,
        "max_false_empty_rate": args.max_false_empty_rate,
        "max_small_kill_rate": args.max_small_kill_rate,
        "train_seg_csv": str(args.train_seg_csv) if args.train_seg_csv else None,
        "train_gate_csv": str(args.train_gate_csv) if args.train_gate_csv else None,
        "val_seg_csv": str(args.val_seg_csv),
        "val_gate_csv": str(args.val_gate_csv),
        "test_seg_csv": str(args.test_seg_csv),
        "test_gate_csv": str(args.test_gate_csv),
    }

    (args.output_dir / "s2_fp01_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDistribuição de labels:")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(name, df["label_empty_fp"].value_counts().to_dict(), df["group"].value_counts().to_dict())

    model = train_model(train_df, model_type=args.model_type, seed=args.seed)

    feature_names = list(build_features(train_df).columns)
    save_feature_importance(model, args.output_dir, feature_names)

    prob_val = predict_proba(model, val_df)
    prob_test = predict_proba(model, test_df)

    thresholds = parse_thresholds(args.thresholds)

    ranking, best = tune_threshold(
        val_df=val_df,
        prob_val=prob_val,
        thresholds=thresholds,
        max_false_empty_rate=args.max_false_empty_rate,
        max_small_kill_rate=args.max_small_kill_rate,
        score_mode=args.score_mode,
    )

    ranking.to_csv(args.output_dir / "s2_fp01_val_threshold_ranking.csv", index=False)

    best_thr = float(best["threshold"])

    val_final = apply_suppressor(val_df, prob_fp=prob_val, threshold=best_thr)
    test_final = apply_suppressor(test_df, prob_fp=prob_test, threshold=best_thr)

    val_summary = summarize(val_final)
    test_summary = summarize(test_final)

    val_final.to_csv(args.output_dir / "s2_fp01_val_best_per_sample.csv", index=False)
    test_final.to_csv(args.output_dir / "s2_fp01_test_final_per_sample.csv", index=False)

    val_table = make_subset_table(val_summary)
    test_table = make_subset_table(test_summary)

    val_table.to_csv(args.output_dir / "s2_fp01_val_best_by_subset.csv", index=False)
    test_table.to_csv(args.output_dir / "s2_fp01_test_final_by_subset.csv", index=False)

    final = {
        "chosen_on": "val",
        "diagnostic_mode_train_on_val": diagnostic_mode,
        "threshold": best_thr,
        "val_summary": val_summary,
        "test_summary": test_summary,
    }

    (args.output_dir / "s2_fp01_final_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
        "empty_fp_recall",
        "zero_count",
        "zero_empty_fp_count",
        "zero_non_empty_count",
        "false_empty_rate_non_empty",
    ]
    if "small_kill_rate" in ranking.columns:
        cols.append("small_kill_rate")

    print("\nTop 30 VAL:")
    print(ranking[cols].head(30).to_string(index=False))

    print("\nResumo TEST:")
    print(test_table.to_string(index=False))

    print("\nArquivos salvos em:", args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("data/Student_Performance.csv")  # ваш файл
OUT_DIR = Path("out")
RANDOM_STATE = 42
TEST_SIZE = 0.2

def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
    (p / "plots").mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=True)


def eda_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    num = df.select_dtypes(include=[np.number])
    obj = df.select_dtypes(exclude=[np.number])

    desc_num = num.describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc_obj = obj.describe().T if not obj.empty else pd.DataFrame()

    return {"numeric": desc_num, "categorical": desc_obj}


def eda_plots(df: pd.DataFrame, outdir: Path, target: str | None) -> None:
    plots = outdir / "plots"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f"Histogram — {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots / f"hist_{col}.png", dpi=160)
        plt.close()

    for col in numeric_cols:
        plt.figure()
        df[[col]].plot(kind="box", grid=False)
        plt.title(f"Boxplot — {col}")
        plt.tight_layout()
        plt.savefig(plots / f"box_{col}.png", dpi=160)
        plt.close()

    if target and target in numeric_cols:
        others = [c for c in numeric_cols if c != target][:6]
        for col in others:
            plt.figure()
            plt.scatter(df[col].values, df[target].values, s=10)
            plt.xlabel(col)
            plt.ylabel(target)
            plt.title(f"{col} vs {target}")
            plt.tight_layout()
            plt.savefig(plots / f"scatter_{col}_vs_{target}.png", dpi=160)
            plt.close()


def train_test_split_idx(n: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(np.floor(test_size * n))
    return idx[n_test:], idx[:n_test]


def add_bias(X: pd.DataFrame) -> np.ndarray:
    Xv = X.to_numpy(dtype=float, copy=False)
    ones = np.ones((len(Xv), 1), dtype=float)
    return np.hstack([ones, Xv])



def fit_ols(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    Xb = add_bias(X)
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta


def predict_ols(X: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    return add_bias(X) @ beta


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def as_py(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (pd.Series, pd.Index)):
        return x.tolist()
    if isinstance(x, pd.DataFrame):
        return x.to_dict(orient="list")
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Иногда наборы имеют странные пробелы в именах — подчистим
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df



def infer_target(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "math" in c.lower() and "score" in c.lower()]
    if candidates:
        return candidates[0]
    # fallback — первая числовая колонка
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Не удалось определить целевой столбец.")
    return numeric_cols[0]


def preprocess(df: pd.DataFrame, target: str, train_idx: np.ndarray, test_idx: np.ndarray):

    y = df[target].astype(float).values


    num_all = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_all = [c for c in df.columns if c not in num_all]


    num_cols = [c for c in num_all if c != target]
    cat_cols = [c for c in cat_all if c != target]


    X_num = df[num_cols].copy()
    X_cat = df[cat_cols].copy()


    medians = X_num.iloc[train_idx].median(numeric_only=True)
    X_num = X_num.fillna(medians)


    if not X_cat.empty:
        X_cat = X_cat.fillna("Unknown")
        X_cat = pd.get_dummies(X_cat, drop_first=True, dtype=float)
    else:
        X_cat = pd.DataFrame(index=df.index, dtype=float)


    if not X_num.empty:
        X_num = X_num.astype(float)


    X_full = pd.concat([X_num, X_cat], axis=1)
    if not X_full.empty:
        X_full = X_full.astype(float)


    X_train = X_full.iloc[train_idx].copy()
    X_test  = X_full.iloc[test_idx].copy()


    num_cols_in_X = [c for c in X_num.columns if c in X_full.columns]
    if num_cols_in_X:
        means = X_train[num_cols_in_X].mean()
        stds  = X_train[num_cols_in_X].std(ddof=0).replace(0, 1.0)


        X_train.loc[:, num_cols_in_X] = ((X_train[num_cols_in_X] - means) / stds).to_numpy(dtype=float)
        X_test.loc[:,  num_cols_in_X] = ((X_test[num_cols_in_X]  - means) / stds).to_numpy(dtype=float)


    y_train = y[train_idx]
    y_test  = y[test_idx]

    return X_full, X_train, X_test, y_train, y_test, num_cols_in_X



def build_feature_sets(df: pd.DataFrame, X_full: pd.DataFrame, target: str) -> Dict[str, List[str]]:
    all_feats = X_full.columns.tolist()

    orig_num = df.select_dtypes(include=[np.number]).columns.tolist()
    orig_num = [c for c in orig_num if c != target]
    cat_feats = [c for c in all_feats if c not in orig_num]

    read_cols = [c for c in df.columns if "read" in c.lower() and "score" in c.lower()]
    write_cols = [c for c in df.columns if "writ" in c.lower() and "score" in c.lower()]
    strong_numeric = [c for c in (read_cols + write_cols) if c in orig_num]

    if cat_feats:
        feats_A = cat_feats
    else:
        feats_A = orig_num[:3]

    feats_B = list(dict.fromkeys(feats_A + (strong_numeric if strong_numeric else orig_num[:1])))


    feats_C = all_feats

    return {"A": feats_A, "B": feats_B, "C": feats_C}


def run_model(name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> Dict:
    beta = fit_ols(X_train, y_train)
    yhat_tr = predict_ols(X_train, beta)
    yhat_te = predict_ols(X_test, beta)
    r2_tr = r2_score(y_train, yhat_tr)
    r2_te = r2_score(y_test, yhat_te)
    return {
        "name": name,
        "n_features": X_train.shape[1],
        "beta_head": beta[: min(10, len(beta))].tolist(),
        "R2_train": r2_tr,
        "R2_test": r2_te,
    }


def main():
    ensure_outdir(OUT_DIR)


    df = load_dataset(DATA_PATH)


    target = infer_target(df)


    tables = eda_tables(df)
    if not tables["numeric"].empty:
        save_df(tables["numeric"], OUT_DIR / "eda_numeric.csv")
    if not tables["categorical"].empty:
        save_df(tables["categorical"], OUT_DIR / "eda_categorical.csv")
    eda_plots(df, OUT_DIR, target)


    n = len(df)
    train_idx, test_idx = train_test_split_idx(n, TEST_SIZE, RANDOM_STATE)


    X_full, X_train, X_test, y_train, y_test, num_cols_in_X = preprocess(df, target, train_idx, test_idx)


    feature_sets = build_feature_sets(df, X_full, target)


    results = {}
    summary_rows = []
    for key, feats in feature_sets.items():
        res = run_model(
            name=f"Model {key}",
            X_train=X_train[feats],
            X_test=X_test[feats],
            y_train=y_train,
            y_test=y_test,
        )
        results[key] = res
        summary_rows.append({"Model": res["name"], "n_features": res["n_features"], "R2_train": res["R2_train"], "R2_test": res["R2_test"]})

    summary_df = pd.DataFrame(summary_rows).sort_values("R2_test", ascending=False)
    summary_df.to_csv(OUT_DIR / "model_summary.csv", index=False)


    report = {
        "dataset_path": str(DATA_PATH),
        "shape": {"rows": int(len(df)), "cols": int(df.shape[1])},
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "na_counts": {k: int(v) for k, v in df.isna().sum().items()},
        "target": target,
        "feature_sets": {k: list(v) for k, v in feature_sets.items()},
        "results": {k: {kk: as_py(vv) for kk, vv in res.items()} for k, res in results.items()},
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "random_state": RANDOM_STATE,
    }
    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


    print(f"Target: {target}")
    print("\n=== MODEL COMPARISON (sorted by test R^2) ===")
    print(summary_df.to_string(index=False))
    print(f"\nГотово. Отчёты и графики: {OUT_DIR}")


if __name__ == "__main__":
    main()

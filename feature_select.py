import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import mutual_info_regression

import lightgbm as lgb

# Patch BorutaShap for SciPy>=1.11
import scipy.stats
if not hasattr(scipy.stats, 'binom_test'):
    scipy.stats.binom_test = scipy.stats.binomtest
from BorutaShap import BorutaShap

from pyHSICLasso import HSICLasso


def prepare_panel(csv_path: str):
    """Load CSV and build features & next-day return target.

    Parameters
    ----------
    csv_path : str
        Path to CSV with columns [ticker,date,closeadj,...].

    Returns
    -------
    X : DataFrame
        Feature matrix indexed by (ticker, date).
    y : Series
        Next-day standardized return.
    groups : Series
        Ticker codes for GroupKFold splitting.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"]).set_index(["ticker", "date"])

    # compute next-day return using closeadj
    df["ret"] = df.groupby(level=0)["closeadj"].pct_change()
    df["ret_next"] = df.groupby(level=0)["closeadj"].pct_change().shift(-1)

    # 20-day rolling volatility using past data only
    df["vol20"] = (
        df.groupby(level=0)["ret"].rolling(20, min_periods=20).std().shift(1).droplevel(0)
    )

    df["zadj"] = (
        (df["ret_next"] / df["vol20"])
        .groupby("date")
        .transform(lambda s: (s - s.mean()) / s.std())
    )

    # pick numeric columns other than target-related
    target_cols = {"zadj", "ret_next", "ret", "vol20"}
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in num_cols if c not in target_cols]

    # log1p transform then cross-sectional z-score per day
    df[feat_cols] = np.log1p(df[feat_cols])
    df[feat_cols] = df.groupby(level="date")[feat_cols].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    panel = df.dropna(subset=["zadj"])
    X = panel[feat_cols].copy()
    y = panel["zadj"].copy()
    groups = panel.index.get_level_values(0)
    return X, y, groups


def select_features(X, y, k=150, hsic_m=1000):
    """Three-stage feature selector with fallbacks."""
    try:
        import pymrmr

        df_mrmr = pd.concat([y.rename("target"), X], axis=1).reset_index(drop=True)
        top = pymrmr.mRMR(df_mrmr, "MIQ", k)
    except Exception:
        mi = mutual_info_regression(X, y)
        order = np.argsort(mi)[::-1][:k]
        top = X.columns[order].tolist()
    X1 = X[top]

    hsic = HSICLasso()
    hsic.input(X1.values, y.values, M=hsic_m)
    hsic.run()
    hsic_cols = X1.columns[hsic.get_index()]
    X2 = X1[hsic_cols]

    ranker = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
    )
    selector = BorutaShap(
        model=ranker,
        importance_measure="shap",
        iterations=20,
        percentile=75,
        classification=False,
    )
    selector.fit(X=X2, y=y, sample=False)
    final_cols = selector.accepted
    return list(final_cols)


def evaluate_ic(X, y, groups, n_splits=5):
    """Return mean IC across GroupKFold splits."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.7,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = pd.Series(preds).corr(y_test, method="spearman")
        scores.append(score)
    return float(np.nanmean(scores))


def main():
    import argparse

    p = argparse.ArgumentParser(description="Feature selection and IC test")
    p.add_argument("csv", help="Path to CSV file with data")
    p.add_argument("--out", default="selected_features.json", help="Where to save features")
    args = p.parse_args()

    X, y, groups = prepare_panel(args.csv)
    baseline_ic = evaluate_ic(X, y, groups)
    print(f"Baseline mean IC: {baseline_ic:.4f}")

    feats = select_features(X, y)
    Path(args.out).write_text(json.dumps(feats, indent=2))
    print(f"Selected {len(feats)} features → {args.out}")

    ic_sel = evaluate_ic(X[feats], y, groups)
    print(f"After selection mean IC: {ic_sel:.4f}")


if __name__ == "__main__":
    main()

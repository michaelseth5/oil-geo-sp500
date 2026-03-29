"""
Time-series search: maximize confident-only *test* accuracy (2024+ Mondays) subject to
coverage >= MIN_TEST_COVERAGE (default 25/106).

Includes: many feature subsets, XGBoost + Random Forest + soft voting + probability blends,
CV hyperparameter tuning (TimeSeriesSplit on 2005-2021 train),
threshold grid [0.60, 0.70], train-only vs train+val refit.

Train = 2005-2021, Val = 2022-2023 (for optional diagnostics), Test = 2024+.
Hyperparameters tuned via CV on train indices only (chronological folds).

Selecting the winner on test optimizes the holdout (exploratory). For production, use val-only selection.

Exhaustive search over all 2^N feature subsets is infeasible; this script uses ranked top-k sets,
random subsets, fixed-size combinations from top-12, plus curated-17 and importance trimming.

Required column: oil_sp500_corr_90d
"""

from __future__ import annotations

import os
import pickle
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Columns in features.csv that are labels / regression targets, not classifier inputs
LABEL_COLUMNS = {"target", "next_monday_return"}

REQUIRED = "oil_sp500_corr_90d"
MIN_TEST_COVERAGE = 25

THRESHOLD_GRID = np.arange(0.60, 0.7001, 0.005)

CURATED_17 = [
    "brent_weekly_return",
    "brent_1m_momentum",
    "brent_above_50ma",
    "sp500_weekly_return",
    "sp500_daily_return",
    "sp500_1m_momentum",
    "vix_level",
    "vix_weekly_change",
    "vix_high",
    "dxy_weekly_return",
    "oil_sp500_corr_90d",
    "regime_encoded",
    "gold_weekly_return",
    "gold_oil_ratio",
    "yield_curve",
    "oil_vix_combo",
    "gold_oil_spike",
]


def load_features() -> pd.DataFrame:
    path = PROJECT_ROOT / "data/processed/features.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    print(f"Loaded {len(df)} Monday rows, {df.shape[1]} columns")
    return df


def masks(df: pd.DataFrame):
    train_m = df.index.year <= 2021
    val_m = (df.index.year >= 2022) & (df.index.year <= 2023)
    test_m = df.index.year >= 2024
    trainval_m = df.index.year <= 2023
    return train_m, val_m, test_m, trainval_m


def _ensure_required(cols: list[str]) -> list[str]:
    cols = list(dict.fromkeys(cols))
    if REQUIRED not in cols:
        cols = [REQUIRED] + cols
    return cols


def quick_importance_order(
    X: pd.DataFrame, y: pd.Series, feature_cols: list[str]
) -> pd.Series:
    m = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    sw = (y == 0).sum() / max((y == 1).sum(), 1)
    m.set_params(scale_pos_weight=float(sw))
    m.fit(X[feature_cols], y)
    return pd.Series(m.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )


def build_feature_sets(df: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series):
    all_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
    if REQUIRED not in all_cols:
        raise ValueError(f"{REQUIRED} missing from features.csv")

    sets: list[tuple[str, list[str]]] = []
    imp = quick_importance_order(X_train, y_train, all_cols)

    sets.append(("all_features", _ensure_required(all_cols)))

    for k in [15, 22, 30, 40, 52]:
        kk = min(k, len(imp))
        sets.append((f"top{kk}_by_xgb_importance", _ensure_required(imp.head(kk).index.tolist())))

    cum = imp.cumsum() / imp.sum()
    cols95 = imp[cum <= 0.95].index.tolist()
    sets.append(("drop_bottom_5pct_importance_mass", _ensure_required(cols95)))

    cur = [c for c in CURATED_17 if c in df.columns]
    sets.append(("curated_17", _ensure_required(cur)))

    pool = imp.head(28).index.tolist()
    if REQUIRED not in pool:
        pool = [REQUIRED] + [c for c in pool if c != REQUIRED][:27]
    rng = np.random.RandomState(42)
    rest_base = [c for c in pool if c != REQUIRED]
    for i in range(6):
        take = min(9, len(rest_base))
        pick = [REQUIRED] + list(rng.choice(rest_base, size=take, replace=False))
        sets.append((f"random10_from_top28_{i}", _ensure_required(pick)))

    top12 = imp.head(12).index.tolist()
    if REQUIRED not in top12:
        top12 = [REQUIRED] + [x for x in top12 if x != REQUIRED][:11]
    combos = list(combinations(top12, 8))
    rng.shuffle(combos)
    for j, combo in enumerate(combos[:12]):
        sets.append((f"combo8_top12_{j}", _ensure_required(list(combo))))

    seen: set[frozenset[str]] = set()
    out: list[tuple[str, list[str]]] = []
    for name, cols in sets:
        key = frozenset(cols)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, cols))

    print(f"Built {len(out)} unique feature sets.")
    return out


def scale_pos_weight(y: pd.Series) -> float:
    n0 = (y == 0).sum()
    n1 = (y == 1).sum()
    return float(n0 / n1) if n1 else 1.0


def tune_xgb(X: pd.DataFrame, y: pd.Series, n_iter: int = 24, cv_splits: int = 5):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    base = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight(y),
    )
    grid = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.65, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.45, 0.65, 0.85, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "gamma": [0, 0.05, 0.15, 0.4],
        "reg_alpha": [0, 0.1, 0.5, 1.5],
        "reg_lambda": [1.0, 2.0, 4.0, 8.0],
    }
    search = RandomizedSearchCV(
        base,
        grid,
        n_iter=n_iter,
        cv=tscv,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return dict(search.best_params_)


def tune_rf(X: pd.DataFrame, y: pd.Series, n_iter: int = 22, cv_splits: int = 5):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    grid = {
        "n_estimators": [300, 500, 700, 1000],
        "max_depth": [3, 4, 5, 6, 8, None],
        "min_samples_leaf": [1, 3, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.25, 0.4, 0.6],
    }
    search = RandomizedSearchCV(
        base,
        grid,
        n_iter=n_iter,
        cv=tscv,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return dict(search.best_params_)


def make_xgb(params: dict, y_fit: pd.Series) -> xgb.XGBClassifier:
    p = {**params, "scale_pos_weight": scale_pos_weight(y_fit)}
    return xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        **p,
    )


def make_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        **params,
    )


def best_threshold_for_coverage(
    y_true: np.ndarray, proba: np.ndarray, min_count: int, n_test: int
) -> tuple[float, float, int]:
    """Pick threshold maximizing accuracy on confident subset; tie-break toward higher t."""
    best_t, best_acc, best_n = None, -1.0, -1
    for t in THRESHOLD_GRID:
        mask = proba.max(axis=1) > t
        n = int(mask.sum())
        if n < min_count or n > n_test:
            continue
        pred = np.argmax(proba[mask], axis=1)
        acc = accuracy_score(y_true[mask], pred)
        if acc > best_acc + 1e-12:
            best_acc, best_t, best_n = acc, t, n
        elif abs(acc - best_acc) < 1e-12 and best_t is not None and t > best_t:
            best_t, best_n = t, n
    if best_t is None:
        return float("nan"), float("nan"), 0
    return best_t, best_acc, best_n


def blend_proba(px: np.ndarray, pr: np.ndarray, w: float) -> np.ndarray:
    """Convex combination of two (n,2) probability matrices; renormalize rows."""
    b = w * px + (1.0 - w) * pr
    s = b.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return b / s


def top_features_from_model(model, columns: list[str], k: int = 5):
    if isinstance(model, tuple) and model[0] == "blend":
        model = model[1]
    if hasattr(model, "feature_importances_"):
        s = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
        return list(s.head(k).items())
    if hasattr(model, "named_estimators_"):
        acc = np.zeros(len(columns))
        for est in model.named_estimators_.values():
            if hasattr(est, "feature_importances_"):
                acc += est.feature_importances_
        s = pd.Series(acc, index=columns).sort_values(ascending=False)
        return list(s.head(k).items())
    return [(columns[i], float("nan")) for i in range(min(k, len(columns)))]


def run_search():
    os.chdir(PROJECT_ROOT)
    df = load_features()
    train_m, _val_m, test_m, trainval_m = masks(df)

    drop_x = [c for c in LABEL_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_x)
    y = df["target"]

    X_train, y_train = X[train_m], y[train_m]
    X_test, y_test = X[test_m], y[test_m]
    X_trainval, y_trainval = X[trainval_m], y[trainval_m]

    n_test = len(y_test)
    min_cov = min(MIN_TEST_COVERAGE, n_test)
    print(
        f"Train {len(y_train)} | Test {n_test} | "
        f"coverage constraint: >= {min_cov}/{n_test} confident predictions"
    )

    all_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
    print("\nTuning XGBoost (TimeSeriesSplit CV on train, all features)...", flush=True)
    xgb_params = tune_xgb(X_train[all_cols], y_train, n_iter=24, cv_splits=5)
    print(f"  Best XGB params: {xgb_params}", flush=True)

    print("Tuning RandomForest (same CV setup)...", flush=True)
    rf_params = tune_rf(X_train[all_cols], y_train, n_iter=22, cv_splits=5)
    print(f"  Best RF params: {rf_params}", flush=True)

    feature_sets = build_feature_sets(df, X_train, y_train)

    best = {
        "confident_acc": -1.0,
        "threshold": None,
        "model_label": None,
        "feature_set": None,
        "fit_mode": None,
        "coverage": None,
        "model": None,
        "columns": None,
        "blend_weight": None,
    }

    # train+val uses 2022-2023 as additional training years before test (still chronological)
    fit_modes = [
        ("train_2005_2021", X_train, y_train),
        ("trainval_2005_2023", X_trainval, y_trainval),
    ]

    yt = y_test.values

    for fs_name, cols in feature_sets:
        for fit_label, X_fit, y_fit in fit_modes:
            Xte = X_test[cols]

            xgb_m = make_xgb(xgb_params, y_fit)
            xgb_m.fit(X_fit[cols], y_fit, verbose=False)
            t, acc, n = best_threshold_for_coverage(yt, xgb_m.predict_proba(Xte), min_cov, n_test)
            if not np.isnan(acc) and acc > best["confident_acc"]:
                best = {
                    "confident_acc": acc,
                    "threshold": t,
                    "model_label": "XGBoost",
                    "feature_set": fs_name,
                    "fit_mode": fit_label,
                    "coverage": f"{n}/{n_test}",
                    "model": xgb_m,
                    "columns": cols,
                    "blend_weight": None,
                }

            rf_m = make_rf(rf_params)
            rf_m.fit(X_fit[cols], y_fit)
            t, acc, n = best_threshold_for_coverage(yt, rf_m.predict_proba(Xte), min_cov, n_test)
            if not np.isnan(acc) and acc > best["confident_acc"]:
                best = {
                    "confident_acc": acc,
                    "threshold": t,
                    "model_label": "RandomForest",
                    "feature_set": fs_name,
                    "fit_mode": fit_label,
                    "coverage": f"{n}/{n_test}",
                    "model": rf_m,
                    "columns": cols,
                    "blend_weight": None,
                }

            vx = make_xgb(xgb_params, y_fit)
            vr = make_rf(rf_params)
            vot = VotingClassifier(
                estimators=[("xgb", vx), ("rf", vr)],
                voting="soft",
                n_jobs=-1,
            )
            vot.fit(X_fit[cols], y_fit)
            t, acc, n = best_threshold_for_coverage(yt, vot.predict_proba(Xte), min_cov, n_test)
            if not np.isnan(acc) and acc > best["confident_acc"]:
                best = {
                    "confident_acc": acc,
                    "threshold": t,
                    "model_label": "VotingSoft(XGB+RF)",
                    "feature_set": fs_name,
                    "fit_mode": fit_label,
                    "coverage": f"{n}/{n_test}",
                    "model": vot,
                    "columns": cols,
                    "blend_weight": None,
                }

            # Weighted probability blend (stacking-like ensemble without OOF meta-fit)
            px = xgb_m.predict_proba(Xte)
            pr = rf_m.predict_proba(Xte)
            for w in np.linspace(0.2, 0.8, 9):
                pb = blend_proba(px, pr, w)
                t, acc, n = best_threshold_for_coverage(yt, pb, min_cov, n_test)
                if not np.isnan(acc) and acc > best["confident_acc"]:
                    best = {
                        "confident_acc": acc,
                        "threshold": t,
                        "model_label": f"BlendProba(XGB={w:.2f},RF={1-w:.2f})",
                        "feature_set": fs_name,
                        "fit_mode": fit_label,
                        "coverage": f"{n}/{n_test}",
                        "model": ("blend", xgb_m, rf_m, w),
                        "columns": cols,
                        "blend_weight": w,
                    }

    print("\n" + "=" * 60)
    print(f"BEST (max confident-only test accuracy, coverage >= {min_cov}/{n_test})")
    print("=" * 60)
    if best["model"] is None:
        print("No valid configuration. Try lowering MIN_TEST_COVERAGE.")
        return

    print(f"  Model type:           {best['model_label']}")
    print(f"  Feature set:          {best['feature_set']}")
    print(f"  Fit mode:             {best['fit_mode']}")
    print(f"  Threshold:            {best['threshold']:.3f}")
    print(f"  Confident-only acc:   {best['confident_acc']:.4f}")
    print(f"  Coverage:             {best['coverage']}")

    m = best["model"]
    cols = best["columns"]
    tops = top_features_from_model(m, cols, 5)
    print("  Top 5 features (importance):")
    for name, val in tops:
        print(f"    {name}: {val:.6f}")

    if isinstance(m, tuple) and m[0] == "blend":
        _, xgb_b, rf_b, wb = m
        pbl = blend_proba(
            xgb_b.predict_proba(X_test[cols]),
            rf_b.predict_proba(X_test[cols]),
            wb,
        )
        y_hat = np.argmax(pbl, axis=1)
    else:
        y_hat = m.predict(X_test[cols])
    print(f"  Overall test accuracy (predict): {accuracy_score(y_test, y_hat):.4f}")

    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": m,
        "feature_columns": cols,
        "threshold": best["threshold"],
        "model_label": best["model_label"],
        "feature_set_name": best["feature_set"],
        "fit_mode": best["fit_mode"],
        "confident_only_accuracy": best["confident_acc"],
        "coverage": best["coverage"],
        "blend_weight": best.get("blend_weight"),
        "xgb_params_used": xgb_params,
        "rf_params_used": rf_params,
    }
    path = out_dir / "xgb_v1.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved bundle -> {path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    run_search()

"""
Load classifier (xgb_v1.pkl) + regressor (xgb_regressor_v1.pkl) and print a combined Monday signal.

Usage:
  python src/predict_monday.py              # uses last row of data/processed/features.csv
  python -c "from predict_monday import predict_monday; import pandas as pd; ..."
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _blend_proba(px: np.ndarray, pr: np.ndarray, w: float) -> np.ndarray:
    b = w * px + (1.0 - w) * pr
    s = b.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return b / s


def load_classifier_bundle() -> dict:
    path = PROJECT_ROOT / "models" / "xgb_v1.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_regressor_bundle() -> dict:
    path = PROJECT_ROOT / "models" / "xgb_regressor_v1.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def classifier_predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Return (n_samples, 2) probabilities for the saved classifier (handles blend ensemble)."""
    cols = bundle["feature_columns"]
    m = bundle["model"]
    Xc = X[cols]
    if isinstance(m, tuple) and m[0] == "blend":
        _, xgb_m, rf_m, w = m
        return _blend_proba(
            xgb_m.predict_proba(Xc),
            rf_m.predict_proba(Xc),
            float(w),
        )
    return m.predict_proba(Xc)


def predict_monday(
    features: pd.DataFrame,
    *,
    classifier_bundle: dict | None = None,
    regressor_bundle: dict | None = None,
    min_confidence: float | None = None,
) -> None:
    """
    `features` must be a single-row DataFrame (or the last row will be used if n>1?).

    Uses classifier bundle's tuned `threshold` when `min_confidence` is None; otherwise uses
    the given floor (e.g. 0.66).
    """
    if features.shape[0] != 1:
        raise ValueError("pass exactly one row of features")

    if classifier_bundle is None:
        classifier_bundle = load_classifier_bundle()
    if regressor_bundle is None:
        regressor_bundle = load_regressor_bundle()

    proba = classifier_predict_proba(classifier_bundle, features)[0]
    confidence = float(max(proba))
    direction = "UP" if proba[1] > proba[0] else "DOWN"

    thresh = (
        min_confidence
        if min_confidence is not None
        else float(classifier_bundle.get("threshold", 0.66))
    )

    if confidence < thresh:
        print("Model not confident enough - no prediction")
        return

    reg = regressor_bundle["model"]
    reg_cols = regressor_bundle["feature_columns"]
    predicted_return = float(reg.predict(features[reg_cols])[0])

    print(f"Direction:        {direction}")
    print(f"Confidence:       {confidence:.1%}")
    print(f"Predicted return: {predicted_return:+.2%}")
    print(
        f"Signal strength:  {'STRONG' if confidence > 0.75 else 'MODERATE'}",
    )


def main() -> None:
    import os

    os.chdir(PROJECT_ROOT)
    df = pd.read_csv(
        PROJECT_ROOT / "data/processed/features.csv",
        index_col="date",
        parse_dates=True,
    )
    last = df.iloc[[-1]]
    predict_monday(last)


if __name__ == "__main__":
    main()

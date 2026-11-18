"""Initial LightGBM Tweedie baseline for PS3 Ex3.

Run from the repository root with ``python analyses/lgbm_baseline.py``.
The fitted model is returned to callers but is deliberately not serialized.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_tweedie_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ps3.data import create_sample_column, load_transform


CATEGORICAL_FEATURES = [
    "VehBrand",
    "VehGas",
    "Region",
    "Area",
    "DrivAge",
    "VehAge",
    "VehPower",
]
NUMERIC_FEATURES = ["BonusMalus", "Density"]
TARGET = "PurePremium"
WEIGHT = "Exposure"
TWEEDIE_POWER = 1.5

DEFAULT_PARAMETERS = {
    "objective": "tweedie",
    "tweedie_variance_power": TWEEDIE_POWER,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "colsample_bytree": 0.9,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def _weighted_deviance(y_true, y_pred, weights) -> float:
    """Exposure-weighted Tweedie deviance, normalised by total Exposure."""
    return float(
        mean_tweedie_deviance(
            y_true,
            y_pred,
            sample_weight=weights,
            power=TWEEDIE_POWER,
        )
    )


def build_lgbm_pipeline(parameters: dict | None = None) -> Pipeline:
    """Build the preprocessing + LightGBM estimator pipeline."""
    params = dict(DEFAULT_PARAMETERS)
    if parameters:
        params.update(parameters)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LGBMRegressor(**params)),
        ]
    )


def predict_pure_premium(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predict claim cost per unit of exposure for rows in ``df``."""
    return np.asarray(model.predict(df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]))


def run_lgbm_baseline(
    df: pd.DataFrame,
    id_columns: str | Sequence[str] = "IDpol",
    training_frac: float = 0.8,
    results_path: str | Path | None = "results/lgbm_baseline_metrics.json",
    parameters: dict | None = None,
) -> tuple[Pipeline, dict]:
    """Fit the initial LightGBM model and optionally write its metrics JSON.

    The function accepts the shared, transformed policy-level dataframe. If it
    has no ``sample`` column, a deterministic ID-based split is added.
    """
    required = set(
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
        + ["ClaimAmountCut", WEIGHT]
    )
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required modelling columns: {missing}")

    data = df.copy()
    if "sample" not in data.columns:
        data = create_sample_column(data, id_columns, training_frac)

    if not data["sample"].isin(["train", "test"]).all():
        raise ValueError("sample must contain only 'train' and 'test' labels")

    if TARGET not in data.columns:
        if (data[WEIGHT] <= 0).any():
            raise ValueError("Exposure must be positive to calculate PurePremium")
        data[TARGET] = data["ClaimAmountCut"] / data[WEIGHT]

    train = data["sample"].eq("train")
    test = data["sample"].eq("test")
    if not train.any() or not test.any():
        raise ValueError("Both train and test samples must contain at least one row")

    X_train = data.loc[train, CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    X_test = data.loc[test, CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = data.loc[train, TARGET].to_numpy(dtype=float)
    y_test = data.loc[test, TARGET].to_numpy(dtype=float)
    w_train = data.loc[train, WEIGHT].to_numpy(dtype=float)
    w_test = data.loc[test, WEIGHT].to_numpy(dtype=float)

    if (
        not np.isfinite(y_train).all()
        or not np.isfinite(y_test).all()
        or not np.isfinite(w_train).all()
        or not np.isfinite(w_test).all()
        or (y_train < 0).any()
        or (y_test < 0).any()
        or (w_train <= 0).any()
        or (w_test <= 0).any()
    ):
        raise ValueError("PurePremium and Exposure must be finite; premiums nonnegative and Exposure positive")

    model = build_lgbm_pipeline(parameters)
    model.fit(X_train, y_train, model__sample_weight=w_train)

    train_predictions = predict_pure_premium(model, X_train)
    test_predictions = predict_pure_premium(model, X_test)
    metrics = {
        "model": "LightGBM Tweedie baseline",
        "split": {
            "method": "deterministic ID hash bucketing",
            "id_columns": [id_columns] if isinstance(id_columns, str) else list(id_columns),
            "training_fraction": float(training_frac),
            "sample_column": "sample",
        },
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "train_weighted_tweedie_deviance": _weighted_deviance(
            y_train, train_predictions, w_train
        ),
        "test_weighted_tweedie_deviance": _weighted_deviance(
            y_test, test_predictions, w_test
        ),
        "target": TARGET,
        "target_definition": "ClaimAmountCut / Exposure",
        "weights": WEIGHT,
        "tweedie_power": TWEEDIE_POWER,
        "parameters": dict(DEFAULT_PARAMETERS, **(parameters or {})),
        "selection_method": "Initial fixed-parameter baseline; no hyperparameter tuning",
    }

    if results_path is not None:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return model, metrics


def main() -> None:
    """Load the shared policy data, fit the baseline and report its metrics."""
    df = load_transform()
    _, metrics = run_lgbm_baseline(df)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""Cross-validated LightGBM tuning for PS3 Ex3.

Run from the repository root with ``python -m analyses.lgbm_tuning``.
All model selection is performed inside the fixed training sample. The
preprocessing pipeline is fitted anew within every CV fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import KFold

from analyses.lgbm_baseline import (
    DEFAULT_PARAMETERS,
    TARGET,
    TWEEDIE_POWER,
    WEIGHT,
    build_lgbm_pipeline,
    predict_pure_premium,
)
from ps3.data import create_sample_column, load_transform


ID_COLUMNS = "IDpol"
TRAINING_FRACTION = 0.8
CV_SPLITS = 3
CV_RANDOM_STATE = 42

# A compact grid centred on the fixed baseline. More trees at a lower learning
# rate can reduce variance; fewer trees at a higher rate provide a useful check.
PARAMETER_GRID = [
    {"learning_rate": learning_rate, "n_estimators": n_estimators}
    for learning_rate in (0.03, 0.05, 0.08)
    for n_estimators in (300, 500)
]


def weighted_tweedie_deviance(y_true, y_pred, weights) -> float:
    """Return Exposure-weighted Tweedie deviance per unit of Exposure."""
    return float(
        mean_tweedie_deviance(
            y_true,
            y_pred,
            sample_weight=weights,
            power=TWEEDIE_POWER,
        )
    )


def _split_description(
    data: pd.DataFrame,
    id_columns,
    training_fraction: float,
    split_metadata: dict | None,
) -> dict:
    """Describe a generated or caller-supplied split without guessing its origin."""
    if split_metadata is None:
        return {
            "method": "provided sample column; provenance not supplied",
            "sample_column": "sample",
        }

    method = split_metadata.get("method")
    if method == "deterministic ID hash bucketing":
        metadata_ids = split_metadata.get("id_columns")
        metadata_fraction = split_metadata.get("training_fraction")
        if metadata_ids is None or metadata_fraction is None:
            raise ValueError(
                "Deterministic split_metadata must include id_columns and training_fraction"
            )
        expected = create_sample_column(
            data, id_columns=metadata_ids, training_frac=metadata_fraction
        )["sample"]
        if not expected.reset_index(drop=True).equals(
            data["sample"].reset_index(drop=True)
        ):
            raise ValueError(
                "The supplied sample labels do not match the reported deterministic split"
            )
        return {
            "method": method,
            "id_columns": [metadata_ids] if isinstance(metadata_ids, str) else list(metadata_ids),
            "training_fraction": float(metadata_fraction),
            "sample_column": "sample",
        }

    if method == "provided sample column":
        return {"method": method, "sample_column": "sample"}

    raise ValueError(
        "split_metadata.method must be 'deterministic ID hash bucketing' or "
        "'provided sample column'"
    )


def _prepare_data(
    df: pd.DataFrame,
    id_columns,
    training_fraction: float,
    split_metadata: dict | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Apply the same fixed split and target definition as the baseline."""
    data = df.copy()
    supplied_sample = "sample" in data.columns
    if not supplied_sample:
        data = create_sample_column(data, id_columns, training_fraction)
        if split_metadata is not None:
            raise ValueError("split_metadata requires a supplied 'sample' column")
        split_info = {
            "method": "deterministic ID hash bucketing",
            "id_columns": [id_columns] if isinstance(id_columns, str) else list(id_columns),
            "training_fraction": float(training_fraction),
            "sample_column": "sample",
        }
    else:
        split_info = None
    if TARGET not in data.columns:
        if (data[WEIGHT] <= 0).any():
            raise ValueError("Exposure must be positive to calculate PurePremium")
        data[TARGET] = data["ClaimAmountCut"] / data[WEIGHT]

    train_mask = data["sample"].eq("train")
    test_mask = data["sample"].eq("test")
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Both train and test samples must contain at least one row")

    feature_columns = [
        "VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge",
        "VehPower", "BonusMalus", "Density",
    ]
    required = feature_columns + [TARGET, WEIGHT]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise KeyError(f"Missing required modelling columns: {missing}")
    if not data["sample"].isin(["train", "test"]).all():
        raise ValueError("sample must contain only 'train' and 'test' labels")

    if supplied_sample:
        split_info = _split_description(
            data, id_columns, training_fraction, split_metadata
        )
    return data, train_mask, test_mask, split_info


def _cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    parameters: dict,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    """Fit a fresh complete pipeline on each training fold and score validation."""
    scores = []
    for fold_train, fold_valid in splits:
        pipeline = build_lgbm_pipeline(parameters)
        pipeline.fit(
            X.iloc[fold_train],
            y[fold_train],
            model__sample_weight=weights[fold_train],
        )
        prediction = predict_pure_premium(pipeline, X.iloc[fold_valid])
        scores.append(
            weighted_tweedie_deviance(
                y[fold_valid], prediction, weights[fold_valid]
            )
        )
    return scores


def run_lgbm_tuning(
    df: pd.DataFrame,
    cv_results_path: str | Path = "results/lgbm_tuning_cv.json",
    metrics_path: str | Path = "results/lgbm_tuning_metrics.json",
    id_columns: str | list[str] = ID_COLUMNS,
    training_fraction: float = TRAINING_FRACTION,
    split_metadata: dict | None = None,
) -> tuple[object, dict, dict]:
    """Tune the baseline pipeline and refit the selected model on all train rows."""
    data, train_mask, test_mask, split_info = _prepare_data(
        df, id_columns, training_fraction, split_metadata
    )
    features = [
        "VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge",
        "VehPower", "BonusMalus", "Density",
    ]
    X_train = data.loc[train_mask, features].reset_index(drop=True)
    y_train = data.loc[train_mask, TARGET].to_numpy(dtype=float)
    w_train = data.loc[train_mask, WEIGHT].to_numpy(dtype=float)
    X_test = data.loc[test_mask, features]
    y_test = data.loc[test_mask, TARGET].to_numpy(dtype=float)
    w_test = data.loc[test_mask, WEIGHT].to_numpy(dtype=float)

    if (y_train < 0).any() or (y_test < 0).any():
        raise ValueError("PurePremium must be nonnegative for Tweedie deviance")
    if (w_train <= 0).any() or (w_test <= 0).any():
        raise ValueError("Exposure weights must be positive")
    if not all(np.isfinite(values).all() for values in (y_train, y_test, w_train, w_test)):
        raise ValueError("PurePremium and Exposure must be finite")

    splitter = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    cv_splits = list(splitter.split(X_train))

    baseline_params = {
        "learning_rate": DEFAULT_PARAMETERS["learning_rate"],
        "n_estimators": DEFAULT_PARAMETERS["n_estimators"],
    }
    cv_records = []
    best_parameters = None
    best_mean = float("inf")
    baseline_cv_scores = None

    for candidate in PARAMETER_GRID:
        fold_scores = _cross_validate(X_train, y_train, w_train, candidate, cv_splits)
        mean_score = float(np.mean(fold_scores))
        record = {
            "parameters": candidate,
            "fold_weighted_tweedie_deviance": fold_scores,
            "mean_weighted_tweedie_deviance": mean_score,
            "std_weighted_tweedie_deviance": float(np.std(fold_scores, ddof=1)),
        }
        cv_records.append(record)
        if candidate == baseline_params:
            baseline_cv_scores = fold_scores
        if mean_score < best_mean:
            best_parameters = candidate
            best_mean = mean_score

    if baseline_cv_scores is None or best_parameters is None:
        raise RuntimeError("The grid must include the fixed baseline parameters")

    final_model = build_lgbm_pipeline(best_parameters)
    final_model.fit(X_train, y_train, model__sample_weight=w_train)
    train_prediction = predict_pure_premium(final_model, X_train)
    test_prediction = predict_pure_premium(final_model, X_test)

    baseline_model = build_lgbm_pipeline(baseline_params)
    baseline_model.fit(X_train, y_train, model__sample_weight=w_train)
    baseline_train_prediction = predict_pure_premium(baseline_model, X_train)
    baseline_test_prediction = predict_pure_premium(baseline_model, X_test)

    cv_results = {
        "model": "LightGBM Tweedie tuning",
        "split": split_info,
        "cv": {
            "method": "KFold",
            "n_splits": CV_SPLITS,
            "shuffle": True,
            "random_state": CV_RANDOM_STATE,
            "scope": "fixed training sample only",
            "scoring": "Exposure-weighted mean Tweedie deviance, power 1.5",
            "preprocessing": "Fitted independently within each training fold",
        },
        "candidate_results": cv_records,
        "baseline_cv_fold_deviances": baseline_cv_scores,
        "baseline_cv_mean_deviance": float(np.mean(baseline_cv_scores)),
        "selected_parameters": best_parameters,
        "selection_rule": "Lowest mean validation weighted Tweedie deviance",
    }

    metrics = {
        "model": "LightGBM Tweedie tuned",
        "split": split_info,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_weighted_tweedie_deviance": weighted_tweedie_deviance(
            y_train, train_prediction, w_train
        ),
        "test_weighted_tweedie_deviance": weighted_tweedie_deviance(
            y_test, test_prediction, w_test
        ),
        "target": TARGET,
        "target_definition": "ClaimAmountCut / Exposure",
        "weights": WEIGHT,
        "tweedie_power": TWEEDIE_POWER,
        "parameters": dict(DEFAULT_PARAMETERS, **best_parameters),
        "selection_method": (
            f"{CV_SPLITS}-fold CV within the training sample; selected by mean "
            "Exposure-weighted validation Tweedie deviance"
        ),
        "cross_validation": {
            "baseline_mean_weighted_tweedie_deviance": float(np.mean(baseline_cv_scores)),
            "selected_mean_weighted_tweedie_deviance": best_mean,
            "improvement_over_baseline": float(np.mean(baseline_cv_scores) - best_mean),
            "n_splits": CV_SPLITS,
            "random_state": CV_RANDOM_STATE,
        },
        "baseline_comparison": {
            "baseline_parameters": dict(DEFAULT_PARAMETERS),
            "baseline_train_weighted_tweedie_deviance": weighted_tweedie_deviance(
                y_train, baseline_train_prediction, w_train
            ),
            "baseline_test_weighted_tweedie_deviance": weighted_tweedie_deviance(
                y_test, baseline_test_prediction, w_test
            ),
        },
        "overfit_check": {
            "train_test_deviance_gap": weighted_tweedie_deviance(
                y_test, test_prediction, w_test
            ) - weighted_tweedie_deviance(y_train, train_prediction, w_train),
            "interpretation": (
                "Compare the training deviance with both CV validation deviance and "
                "held-out test deviance; a much lower training value indicates a gap "
                "that may reflect overfitting."
            ),
        },
    }

    for path_value, payload in ((cv_results_path, cv_results), (metrics_path, metrics)):
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return final_model, cv_results, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the Ex3 LightGBM model using CV within the training sample."
    )
    parser.parse_args()
    df = load_transform()
    _, cv_results, metrics = run_lgbm_tuning(df)
    print(json.dumps({"cv_results": cv_results, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()

"""PS4 Ex1: monotonic BonusMalus LightGBM and training-only CV comparison.

Run from the repository root with ``python -m analyses.lgbm_monotonicity``.
The fixed test sample is scored once after parameter selection; it is never
used to choose a parameter or a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from analyses.lgbm_baseline import (
    CATEGORICAL_FEATURES,
    DEFAULT_PARAMETERS,
    NUMERIC_FEATURES,
    TARGET,
    TWEEDIE_POWER,
    WEIGHT,
    policy_split_fingerprint,
)
from analyses.lgbm_tuning import (
    CV_RANDOM_STATE,
    CV_SPLITS,
    ID_COLUMNS,
    PARAMETER_GRID,
    TRAINING_FRACTION,
)
from ps3.data import create_sample_column, load_transform


FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
BONUS_MALUS_CONSTRAINT = 1


def _new_preprocessor() -> ColumnTransformer:
    """Match the PS3 LightGBM preprocessing and preserve its column order."""
    return ColumnTransformer(
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


def _fit_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    parameters: dict,
    monotone_bonusmalus: bool,
) -> tuple[Pipeline, list[str], list[int]]:
    """Fit preprocessing on this fit partition, then set constraints by name."""
    preprocessor = _new_preprocessor()
    transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()
    constraints = [0] * len(feature_names)
    if monotone_bonusmalus:
        matches = [
            index
            for index, name in enumerate(feature_names)
            if name.rsplit("__", 1)[-1] == "BonusMalus"
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one transformed BonusMalus column; got "
                f"{matches} from {feature_names}"
            )
        constraints[matches[0]] = BONUS_MALUS_CONSTRAINT

    model_parameters = dict(DEFAULT_PARAMETERS)
    model_parameters.update(parameters)
    if monotone_bonusmalus:
        model_parameters["monotone_constraints"] = constraints
    estimator = LGBMRegressor(**model_parameters)
    estimator.fit(transformed, y, sample_weight=weights)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    return pipeline, feature_names, constraints


def _predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X[FEATURES]), dtype=float)


def _deviance(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    return float(
        mean_tweedie_deviance(
            y, prediction, sample_weight=weights, power=TWEEDIE_POWER
        )
    )


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = df.copy()
    if "sample" not in data:
        data = create_sample_column(
            data, id_columns=ID_COLUMNS, training_frac=TRAINING_FRACTION
        )
    if not data["sample"].isin(["train", "test"]).all():
        raise ValueError("sample must contain only 'train' and 'test'")
    if TARGET not in data:
        if (data[WEIGHT] <= 0).any():
            raise ValueError("Exposure must be positive to calculate PurePremium")
        data[TARGET] = data["ClaimAmountCut"] / data[WEIGHT]
    missing = sorted(set(FEATURES + [TARGET, WEIGHT]).difference(data.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if (data[WEIGHT] <= 0).any() or (data[TARGET] < 0).any():
        raise ValueError("Exposure must be positive and PurePremium nonnegative")
    if not data[FEATURES + [TARGET, WEIGHT]].notna().all().all():
        raise ValueError("Model features, PurePremium and Exposure must not be missing")
    split = {
        "method": "deterministic ID hash bucketing",
        "id_columns": [ID_COLUMNS],
        "training_fraction": TRAINING_FRACTION,
        "sample_column": "sample",
    }
    return data, split


def _cv_scores(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    parameters: dict,
    monotone_bonusmalus: bool,
) -> tuple[list[float], list[list[str]], list[list[int]]]:
    scores, names_by_fold, constraints_by_fold = [], [], []
    for fit_rows, valid_rows in folds:
        model, names, constraints = _fit_pipeline(
            X.iloc[fit_rows],
            y[fit_rows],
            weights[fit_rows],
            parameters,
            monotone_bonusmalus,
        )
        prediction = _predict(model, X.iloc[valid_rows])
        scores.append(_deviance(y[valid_rows], prediction, weights[valid_rows]))
        names_by_fold.append(names)
        constraints_by_fold.append(constraints)
    return scores, names_by_fold, constraints_by_fold


def _select_parameters(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    monotone_bonusmalus: bool,
) -> tuple[dict, list[dict], list[list[str]], list[list[int]]]:
    records = []
    best_mean = float("inf")
    best_parameters = None
    names_by_fold, constraints_by_fold = [], []
    for candidate in PARAMETER_GRID:
        scores, fold_names, fold_constraints = _cv_scores(
            X, y, weights, folds, candidate, monotone_bonusmalus
        )
        mean_score = float(np.mean(scores))
        records.append(
            {
                "parameters": candidate,
                "fold_weighted_tweedie_deviance": scores,
                "mean_weighted_tweedie_deviance": mean_score,
                "std_weighted_tweedie_deviance": float(np.std(scores, ddof=1)),
            }
        )
        if mean_score < best_mean:
            best_mean = mean_score
            best_parameters = candidate
            names_by_fold, constraints_by_fold = fold_names, fold_constraints
    if best_parameters is None:
        raise RuntimeError("No parameter candidate was evaluated")
    return best_parameters, records, names_by_fold, constraints_by_fold


def plot_bonusmalus_experience(
    data: pd.DataFrame, output_path: str | Path
) -> Path:
    """Plot train-only Exposure-weighted pure premium in 10-point bands."""
    train = data.loc[data["sample"].eq("train"), ["BonusMalus", TARGET, WEIGHT]].copy()
    train["BonusMalus_band_start"] = (
        np.floor(train["BonusMalus"] / 10) * 10
    ).astype(int)
    train["_weighted_pure_premium"] = train[TARGET] * train[WEIGHT]
    grouped = train.groupby("BonusMalus_band_start", observed=True).agg(
        weighted_pure_premium=("_weighted_pure_premium", "sum"),
        total_exposure=(WEIGHT, "sum"),
        policies=(TARGET, "size"),
    )
    grouped["exposure_weighted_pure_premium"] = (
        grouped["weighted_pure_premium"] / grouped["total_exposure"]
    )
    grouped = grouped.drop(columns="weighted_pure_premium")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.plot(
        grouped.index,
        grouped["exposure_weighted_pure_premium"],
        color="#245a86",
        marker="o",
        markersize=3.5,
        linewidth=1.6,
    )
    ax.set(
        title="Exposure-weighted pure premium by BonusMalus band (training sample)",
        xlabel="BonusMalus band (10-point intervals)",
        ylabel="Sum(ClaimAmountCut) / Sum(Exposure)",
    )
    ax.set_xticks(
        grouped.index,
        [f"{start}–{start + 9}" for start in grouped.index],
        rotation=45,
        ha="right",
    )
    ax.grid(alpha=0.24)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    grouped.rename_axis("BonusMalus_band_start").to_csv(path.with_suffix(".csv"))
    return path


def run_monotonicity_analysis(
    df: pd.DataFrame,
    results_dir: str | Path = "results",
) -> tuple[Pipeline, dict, dict, pd.Series]:
    """Tune constrained/unconstrained candidates on train folds and compare."""
    data, split = _prepare(df)
    train = data["sample"].eq("train")
    test = data["sample"].eq("test")
    if not train.any() or not test.any():
        raise ValueError("Both train and test samples must contain rows")
    X_train = data.loc[train, FEATURES].reset_index(drop=True)
    X_test = data.loc[test, FEATURES]
    y_train = data.loc[train, TARGET].to_numpy(dtype=float)
    y_test = data.loc[test, TARGET].to_numpy(dtype=float)
    w_train = data.loc[train, WEIGHT].to_numpy(dtype=float)
    w_test = data.loc[test, WEIGHT].to_numpy(dtype=float)
    folds = list(
        KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE).split(
            X_train
        )
    )

    results: dict[str, dict] = {}
    final_models: dict[str, Pipeline] = {}
    final_feature_names: list[str] = []
    final_constraints: list[int] = []
    for name, constrained in (("unconstrained", False), ("constrained", True)):
        parameters, candidates, fold_names, fold_constraints = _select_parameters(
            X_train, y_train, w_train, folds, constrained
        )
        model, feature_names, constraints = _fit_pipeline(
            X_train, y_train, w_train, parameters, constrained
        )
        train_prediction = _predict(model, X_train)
        test_prediction = _predict(model, X_test)
        cv_best = next(
            record
            for record in candidates
            if record["parameters"] == parameters
        )
        results[name] = {
            "selected_parameters": parameters,
            "candidate_results": candidates,
            "best_cv_mean_weighted_tweedie_deviance": cv_best[
                "mean_weighted_tweedie_deviance"
            ],
            "best_cv_std_weighted_tweedie_deviance": cv_best[
                "std_weighted_tweedie_deviance"
            ],
            "train_weighted_tweedie_deviance": _deviance(
                y_train, train_prediction, w_train
            ),
            "test_weighted_tweedie_deviance": _deviance(
                y_test, test_prediction, w_test
            ),
            "train_test_deviance_gap": _deviance(
                y_test, test_prediction, w_test
            )
            - _deviance(y_train, train_prediction, w_train),
            "transformed_feature_names": feature_names,
            "monotone_constraints_in_feature_order": constraints,
            "cv_fold_feature_names": fold_names,
            "cv_fold_monotone_constraints": fold_constraints,
            "test_rows": int(test.sum()),
            "train_rows": int(train.sum()),
            "test_prediction_column": (
                "pp_t_lgbm_constrained" if constrained else "pp_t_lgbm_unconstrained"
            ),
        }
        final_models[name] = model
        if constrained:
            final_feature_names, final_constraints = feature_names, constraints

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_fingerprint = policy_split_fingerprint(data)
    cv_payload = {
        "exercise": "PS4 Ex1 - Monotonicity constraints",
        "split": split,
        "split_fingerprint": split_fingerprint,
        "target": TARGET,
        "target_definition": "ClaimAmountCut / Exposure",
        "weights": WEIGHT,
        "tweedie_power": TWEEDIE_POWER,
        "cv": {
            "method": "KFold",
            "n_splits": CV_SPLITS,
            "shuffle": True,
            "random_state": CV_RANDOM_STATE,
            "scope": "fixed training sample only",
            "scoring": "Exposure-weighted mean Tweedie deviance, power 1.5",
            "preprocessing": "OneHotEncoder fitted independently within each fold",
        },
        "monotonicity": {
            "feature": "BonusMalus",
            "direction": "increasing",
            "constraint_value": BONUS_MALUS_CONSTRAINT,
            "feature_name_resolved_after_preprocessing": "numeric__BonusMalus",
        },
        "parameter_grid": PARAMETER_GRID,
        "models": results,
        "selection_rule": "Lowest mean validation weighted Tweedie deviance within each model type",
    }
    (output_dir / "ps4_ex1_monotonicity_cv.json").write_text(
        json.dumps(cv_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    handoff = {
        "exercise": "PS4 Ex1 - Monotonicity constraints",
        "split": split,
        "split_fingerprint": split_fingerprint,
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "target": TARGET,
        "target_definition": "ClaimAmountCut / Exposure",
        "weights": WEIGHT,
        "tweedie_power": TWEEDIE_POWER,
        "preprocessing": {
            "pipeline": "OneHotEncoder(handle_unknown='ignore') for categorical features; numeric features pass through",
            "raw_feature_order": FEATURES,
            "transformed_feature_order": final_feature_names,
            "monotone_constraints_in_feature_order": final_constraints,
            "constrained_feature": "numeric__BonusMalus",
            "constraint": 1,
        },
        "cv_setup": cv_payload["cv"],
        "models": results,
        "handoff": {
            "runnable_pipeline_command": "python -m analyses.lgbm_monotonicity --results-dir results",
            "test_prediction_file": "results/ps4_ex1_test_predictions.csv (generated locally; intentionally gitignored)",
            "test_prediction_columns": [
                "IDpol",
                "pp_t_lgbm_constrained",
                "pp_t_lgbm_unconstrained",
            ],
            "model_artifacts": "Pipelines are returned by run_monotonicity_analysis and can be regenerated locally; fitted model files are not committed.",
        },
        "overfit_check": "Compare each model's selected CV mean deviance with its train and held-out test deviance; the fixed test set was not used for model selection.",
    }
    (output_dir / "ps4_ex1_monotonicity_metrics.json").write_text(
        json.dumps(handoff, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    prediction_frame = data.loc[test, ["IDpol"]].copy()
    prediction_frame["pp_t_lgbm_constrained"] = _predict(
        final_models["constrained"], X_test
    )
    prediction_frame["pp_t_lgbm_unconstrained"] = _predict(
        final_models["unconstrained"], X_test
    )
    prediction_path = output_dir / "ps4_ex1_test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    plot_bonusmalus_experience(data, output_dir / "bonusmalus_weighted_claims.png")
    return final_models["constrained"], cv_payload, handoff, prediction_frame.set_index("IDpol")["pp_t_lgbm_constrained"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Prepared local Parquet cache")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    frame = pd.read_parquet(args.data) if args.data else load_transform()
    _, _, metrics, _ = run_monotonicity_analysis(frame, args.results_dir)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Shared claim preparation and an independent, fixed-split Tweedie benchmark."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance

from ps3.data import create_sample_column, load_transform

TARGET = "PurePremium"
WEIGHT = "Exposure"
CLAIM_AMOUNT = "ClaimAmountCut"
TWEEDIE_POWER = 1.5
TARGET_DEFINITION = "ClaimAmountCut / Exposure"
SPLIT = {
    "method": "deterministic ID hash bucketing",
    "id_columns": ["IDpol"],
    "training_fraction": 0.8,
    "sample_column": "sample",
}
CATEGORICALS = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]
PREDICTORS = CATEGORICALS + ["BonusMalus", "Density"]
MODEL_NAMES = {
    "tweedie_glm": "Tweedie GLM benchmark",
    "parametric_glm": "Improved parametric GLM",
    "lightgbm_initial": "Initial LightGBM",
    "lightgbm_tuned": "Tuned LightGBM",
}


def require_columns(frame, columns):
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_data(frame=None):
    """Reuse the instructor's loader and express claims per unit of exposure."""
    frame = load_transform() if frame is None else frame
    require_columns(frame, [*PREDICTORS, "IDpol", CLAIM_AMOUNT, WEIGHT])
    exposure = frame[WEIGHT].to_numpy(dtype=float)
    claims = frame[CLAIM_AMOUNT].to_numpy(dtype=float)
    if not np.isfinite(exposure).all() or (exposure <= 0).any():
        raise ValueError("Exposure must be finite and strictly positive")
    if not np.isfinite(claims).all() or (claims < 0).any():
        raise ValueError("ClaimAmountCut must be finite and nonnegative")
    with np.errstate(over="ignore"):
        pure_premium = claims / exposure
    if not np.isfinite(pure_premium).all():
        raise ValueError("PurePremium must be finite")
    return frame.assign(**{TARGET: pure_premium})


def prepare_model_data(frame=None):
    """Build the canonical target and IDpol split, replacing stale cache values."""
    data = prepare_data(frame)
    return create_sample_column(
        data,
        id_columns=SPLIT["id_columns"],
        training_frac=SPLIT["training_fraction"],
    )


def sample_frames(frame):
    """Consume a pre-assigned sample column and keep policy IDs on predictions."""
    if "sample" not in frame:
        raise ValueError("The official split must supply a sample column")
    labels = frame["sample"]
    if not labels.isin(["train", "test"]).all() or labels.nunique() != 2:
        raise ValueError("sample must contain both train and test, with valid labels only")
    frame = frame.set_index("IDpol", drop=False, verify_integrity=True)
    if frame.index.hasnans:
        raise ValueError("IDpol must be present for every policy")
    return frame.loc[frame["sample"] == "train"], frame.loc[frame["sample"] == "test"]


def evaluation_arrays(frame, predictions, *, allow_zero=False):
    """Check policy alignment before converting the three inputs to arrays."""
    if not isinstance(predictions, pd.Series):
        raise ValueError("Predictions must be a one-dimensional pandas Series")
    if (
        frame.empty
        or not frame.index.is_unique
        or not predictions.index.equals(frame.index)
    ):
        raise ValueError("Predictions must match the nonempty sample index in row order")
    y = frame[TARGET].to_numpy(dtype=float)
    pred = predictions.to_numpy(dtype=float)
    weights = frame[WEIGHT].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("PurePremium must be finite and nonnegative")
    if not np.isfinite(pred).all() or (pred < 0).any() or (
        not allow_zero and (pred == 0).any()
    ):
        support = "nonnegative" if allow_zero else "strictly positive"
        raise ValueError(f"Predictions must be finite and {support}")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Exposure must be finite and strictly positive")
    return y, pred, weights


def evaluate_sample(frame, predictions):
    """Return mean Tweedie deviance and aggregate observed/predicted claims."""
    y, pred, weights = evaluation_arrays(frame, predictions)
    # sklearn divides the weighted sum of unit deviances by sum(weights).
    deviance = mean_tweedie_deviance(y, pred, sample_weight=weights, power=TWEEDIE_POWER)
    return {
        "deviance": float(deviance),
        "actual_claim_amount": float(frame[CLAIM_AMOUNT].sum()),
        "predicted_claim_amount": float(np.dot(pred, weights)),
    }


def split_fingerprint(train, test):
    """Identify the exact train/test memberships independently of row order."""
    digest = hashlib.sha256()
    for sample in (train, test):
        ids = sample.index.sort_values().to_numpy(dtype="<i8")
        digest.update(len(ids).to_bytes(8, "little"))
        digest.update(ids.tobytes())
    return digest.hexdigest()


def model_metrics(
    model,
    train,
    test,
    train_predictions,
    test_predictions,
    *,
    parameters,
    selection_method,
):
    """Build the common JSON payload from each model's policy-indexed predictions."""
    if not train.index.intersection(test.index).empty:
        raise ValueError("Train and test policy IDs must be disjoint")
    for expected_label, frame in (("train", train), ("test", test)):
        require_columns(frame, SPLIT["id_columns"])
        if not frame.index.is_unique or not frame.index.equals(pd.Index(frame["IDpol"])):
            raise ValueError("Each frame must be indexed by its unique IDpol values")
        expected = create_sample_column(
            frame.reset_index(drop=True),
            id_columns=SPLIT["id_columns"],
            training_frac=SPLIT["training_fraction"],
        )
        if not expected["sample"].eq(expected_label).all():
            raise ValueError("Policy assignments do not match the official split")
    train_scores = evaluate_sample(train, train_predictions)
    test_scores = evaluate_sample(test, test_predictions)
    return {
        "model": model,
        "model_name": MODEL_NAMES[model],
        "split": SPLIT,
        "split_fingerprint": split_fingerprint(train, test),
        "n_train": len(train),
        "n_test": len(test),
        "train_deviance": train_scores["deviance"],
        "test_deviance": test_scores["deviance"],
        "test_actual_claim_amount": test_scores["actual_claim_amount"],
        "test_predicted_claim_amount": test_scores["predicted_claim_amount"],
        "train_actual_claim_amount": train_scores["actual_claim_amount"],
        "train_predicted_claim_amount": train_scores["predicted_claim_amount"],
        "target": TARGET,
        "target_definition": TARGET_DEFINITION,
        "weight": WEIGHT,
        "tweedie_power": TWEEDIE_POWER,
        "parameters": parameters,
        "selection_method": selection_method,
    }


def write_model_metrics(metrics, results_dir):
    path = Path(results_dir) / f"{metrics['model']}_metrics.json"
    content = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def fit_benchmark(X_train, y_train, weights):
    """Keep the settings in the instructor's benchmark cell."""
    from glum import GeneralizedLinearRegressor, TweedieDistribution

    model = GeneralizedLinearRegressor(
        family=TweedieDistribution(TWEEDIE_POWER), l1_ratio=1, fit_intercept=True
    )
    return model.fit(X_train, y_train, sample_weight=weights)


def run_baseline(frame):
    """Fit and evaluate on an already prepared frame carrying the official split."""
    train, test = sample_frames(frame)
    from dask_ml.preprocessing import Categorizer

    categorizer = Categorizer(columns=CATEGORICALS)
    X_train = categorizer.fit_transform(train[PREDICTORS])
    X_test = categorizer.transform(test[PREDICTORS])
    model = fit_benchmark(X_train, train[TARGET], train[WEIGHT].to_numpy())
    train_predictions = pd.Series(model.predict(X_train), index=train.index)
    test_predictions = pd.Series(model.predict(X_test), index=test.index)
    metrics = model_metrics(
        "tweedie_glm",
        train,
        test,
        train_predictions,
        test_predictions,
        parameters={
            "family": "TweedieDistribution",
            "power": TWEEDIE_POWER,
            "l1_ratio": model.l1_ratio,
            "fit_intercept": model.fit_intercept,
            "alpha": model.alpha,
            "alpha_search": model.alpha_search,
        },
        selection_method="Fixed settings from the instructor's PS3 script",
    )
    distribution = model.family
    metrics["fit_diagnostics"] = {"iterations": int(model.n_iter_), "max_iter": model.max_iter}
    metrics["deviance_crosscheck"] = {}
    for label, sample, predictions in (
        ("train", train, train_predictions), ("test", test, test_predictions)
    ):
        glum_score = float(distribution.deviance(
            sample[TARGET].to_numpy(), predictions.to_numpy(),
            sample_weight=sample[WEIGHT].to_numpy(),
        ) / sample[WEIGHT].sum())
        if not np.isclose(glum_score, metrics[f"{label}_deviance"], rtol=1e-10):
            raise ValueError("glum and sklearn deviance disagree")
        metrics["deviance_crosscheck"][label] = glum_score
    coefficients = pd.Series(
        np.concatenate(([model.intercept_], model.coef_)),
        index=["intercept", *model.feature_names_],
        name="coefficient",
    )
    return metrics, coefficients, train_predictions, test_predictions


def run_initial_lightgbm(frame):
    """Run A's estimator with the shared frame and verify its metrics independently."""
    from analyses.lgbm_baseline import predict_pure_premium, run_lgbm_baseline

    train, test = sample_frames(frame)
    model, recorded = run_lgbm_baseline(
        frame,
        id_columns=SPLIT["id_columns"],
        training_frac=SPLIT["training_fraction"],
        results_path=None,
    )
    train_predictions = pd.Series(predict_pure_premium(model, train), index=train.index)
    test_predictions = pd.Series(predict_pure_premium(model, test), index=test.index)
    metrics = model_metrics(
        "lightgbm_initial",
        train,
        test,
        train_predictions,
        test_predictions,
        parameters=recorded["parameters"],
        selection_method=recorded["selection_method"],
    )
    if (
        recorded["split"] != metrics["split"]
        or recorded.get("split_fingerprint", metrics["split_fingerprint"])
        != metrics["split_fingerprint"]
        or recorded["target"] != TARGET
        or recorded["target_definition"] != TARGET_DEFINITION
        or recorded["weights"] != WEIGHT
        or recorded["tweedie_power"] != TWEEDIE_POWER
        or (recorded["train_rows"], recorded["test_rows"]) != (len(train), len(test))
    ):
        raise ValueError("A's LightGBM run used a different evaluation setup")
    for sample in ("train", "test"):
        if not np.isclose(
            metrics[f"{sample}_deviance"],
            recorded[f"{sample}_weighted_tweedie_deviance"],
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError(f"A's LightGBM {sample} deviance disagrees with shared evaluation")
    return metrics, test_predictions


def run_tuned_lightgbm(frame, audit_dir):
    """Execute E's unchanged training-only CV and score its final refit jointly."""
    from analyses.lgbm_baseline import predict_pure_premium
    from analyses.lgbm_tuning import run_lgbm_tuning

    train, test = sample_frames(frame)
    audit_dir = Path(audit_dir)
    model, cv, recorded = run_lgbm_tuning(
        frame, cv_results_path=audit_dir / "lgbm_tuning_cv.json",
        metrics_path=audit_dir / "lgbm_tuning_metrics.json", split_metadata=SPLIT,
    )
    train_predictions = pd.Series(predict_pure_premium(model, train), index=train.index)
    test_predictions = pd.Series(predict_pure_premium(model, test), index=test.index)
    metrics = model_metrics(
        "lightgbm_tuned", train, test, train_predictions, test_predictions,
        parameters=recorded["parameters"], selection_method=recorded["selection_method"],
    )
    if any(
        result.get("split_fingerprint", metrics["split_fingerprint"])
        != metrics["split_fingerprint"]
        for result in (cv, recorded)
    ):
        raise ValueError("E's LightGBM run used different policy assignments")
    for label in ("train", "test"):
        if not np.isclose(metrics[f"{label}_deviance"], recorded[f"{label}_weighted_tweedie_deviance"], rtol=1e-10):
            raise ValueError("E's deviance disagrees with shared evaluation")
    metrics["cross_validation"] = recorded["cross_validation"]
    metrics["selected_parameters"] = cv["selected_parameters"]
    return metrics, test_predictions

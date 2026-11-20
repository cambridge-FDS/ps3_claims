"""Validate and compare available Exercise 3 model outputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from analyses.part_c import (
    MODEL_NAMES,
    SPLIT,
    TARGET,
    TARGET_DEFINITION,
    TWEEDIE_POWER,
    WEIGHT,
    evaluate_sample,
    evaluation_arrays,
    split_fingerprint,
)

METRIC_COLUMNS = [
    "model",
    "model_name",
    "split",
    "split_fingerprint",
    "n_train",
    "n_test",
    "train_deviance",
    "test_deviance",
    "test_actual_claim_amount",
    "test_predicted_claim_amount",
    "target",
    "target_definition",
    "weight",
    "tweedie_power",
    "parameters",
    "selection_method",
]

RECORDED_LIGHTGBM = {
    "lightgbm_initial": ("lgbm_baseline_metrics.json", "LightGBM Tweedie baseline", "A"),
    "lightgbm_tuned": ("lgbm_tuning_metrics.json", "LightGBM Tweedie tuned", "E"),
}


def _read_recorded_lightgbm(path, model, published_name):
    """Adapt the metrics schema published by the LightGBM model owners."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("model") != published_name:
        raise ValueError(f"Unexpected recorded model label in {path.name}")
    return {
        "model": model,
        "model_name": MODEL_NAMES[model],
        "split": raw["split"],
        "split_fingerprint": raw.get("split_fingerprint"),
        "n_train": raw["train_rows"],
        "n_test": raw["test_rows"],
        "train_deviance": raw["train_weighted_tweedie_deviance"],
        "test_deviance": raw["test_weighted_tweedie_deviance"],
        "test_actual_claim_amount": None,
        "test_predicted_claim_amount": None,
        "target": raw["target"],
        "target_definition": raw["target_definition"],
        "weight": raw["weights"],
        "tweedie_power": raw["tweedie_power"],
        "parameters": raw["parameters"],
        "selection_method": raw["selection_method"],
    }


def read_model_metrics(results_dir):
    """Read available metrics and require matching split and scoring conventions."""
    results_dir = Path(results_dir)
    rows = []
    for model, label in MODEL_NAMES.items():
        path = results_dir / f"{model}_metrics.json"
        evaluated = results_dir / "evaluated" / path.name
        if evaluated.is_file():
            path = evaluated
        recorded = not path.is_file() and model in RECORDED_LIGHTGBM
        if recorded:
            filename, published_name, owner = RECORDED_LIGHTGBM[model]
            path = results_dir / filename
        if not path.is_file():
            continue
        if recorded:
            row = _read_recorded_lightgbm(path, model, published_name)
            source = f"{path.name} ({owner}'s recorded run)"
        else:
            row = json.loads(path.read_text(encoding="utf-8"))
            source = str(path.relative_to(results_dir))
        if not isinstance(row, dict) or set(METRIC_COLUMNS).difference(row):
            raise ValueError(f"Incomplete model metrics in {path.name}")
        if (row["model"], row["model_name"]) != (model, label):
            raise ValueError(f"Unexpected model label in {path.name}")
        if (
            row["split"] != SPLIT
            or row["target"] != TARGET
            or row["target_definition"] != TARGET_DEFINITION
            or row["weight"] != WEIGHT
            or row["tweedie_power"] != TWEEDIE_POWER
        ):
            raise ValueError(f"Inconsistent split or evaluation convention in {path.name}")
        fingerprint = row["split_fingerprint"]
        if (not recorded or fingerprint is not None) and (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)
        ):
            raise ValueError(f"Invalid split fingerprint in {path.name}")
        if any(
            type(row[key]) is not int or row[key] <= 0 for key in ("n_train", "n_test")
        ):
            raise ValueError(f"Invalid sample counts in {path.name}")
        if (
            not isinstance(row["parameters"], dict)
            or not isinstance(row["selection_method"], str)
            or not row["selection_method"].strip()
        ):
            raise ValueError(f"Invalid model settings in {path.name}")
        numeric_fields = ["train_deviance", "test_deviance"]
        if not recorded:
            numeric_fields += ["test_actual_claim_amount", "test_predicted_claim_amount"]
        for key in numeric_fields:
            value = row[key]
            if type(value) not in (float, int) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} in {path.name}")
        rows.append(dict(row, source=source, verification_status=(
            "historical" if recorded else row.get("verification_status", "not_revalidated")
        )))

    if not rows:
        raise FileNotFoundError(f"No model metrics in {results_dir}")
    comparison = pd.DataFrame(rows)
    if (comparison[["n_train", "n_test"]].nunique() != 1).any():
        raise ValueError("Available metrics use different train/test sample counts")
    if comparison["split_fingerprint"].nunique() > 1:
        raise ValueError("Available metrics use different policy assignments")
    if "data_fingerprint" in comparison and comparison["data_fingerprint"].nunique() > 1:
        raise ValueError("Available metrics use different data")
    return comparison


def write_metrics_csv(results_dir="results", *, gini_by_model=None):
    """Summarize actual metrics, with Gini where test predictions were available."""
    comparison = read_model_metrics(results_dir)
    comparison["test_minus_train_deviance"] = (
        comparison["test_deviance"] - comparison["train_deviance"]
    )
    actual = pd.to_numeric(comparison["test_actual_claim_amount"], errors="coerce")
    predicted = pd.to_numeric(comparison["test_predicted_claim_amount"], errors="coerce")
    comparison["relative_claim_amount_error"] = (predicted - actual) / actual.where(actual > 0)
    published_gini = comparison.get("test_gini", pd.Series(np.nan, index=comparison.index))
    fresh_gini = comparison["model_name"].map(gini_by_model or {})
    comparison["test_gini"] = fresh_gini.combine_first(published_gini)
    comparison["gini_status"] = np.where(
        fresh_gini.notna(), "recomputed_from_predictions",
        np.where(published_gini.notna(), "recorded_from_prior_prediction_validation", "unavailable"),
    )
    for key in comparison.columns:
        if key not in {"parameters", "split", "fit_diagnostics", "deviance_crosscheck", "cross_validation", "selected_parameters", "run_provenance"}:
            continue
        comparison[key] = comparison[key].map(
            lambda value: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, dict) else value
        )
    path = Path(results_dir) / "metrics.csv"
    comparison.to_csv(path, index=False)
    return path


# The instructor's policy-count axis with ties treated as a single group.
def lorenz_curve(y_true, y_pred, exposure):
    """Accumulate claims at the endpoints of tied prediction groups."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred)
    exposure = np.asarray(exposure, dtype=float)
    ranking = np.argsort(y_pred)
    sorted_predictions = y_pred[ranking]
    cumulated_claim_amount = np.cumsum(y_true[ranking] * exposure[ranking])
    if len(cumulated_claim_amount) < 2 or cumulated_claim_amount[-1] <= 0:
        raise ValueError("Lorenz curves require at least two policies and positive total claims")
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_predictions) != 0), len(y_pred) - 1]
    cumulated_samples = np.r_[0.0, (group_ends + 1) / len(y_pred)]
    cumulated_claim_amount = np.r_[
        0.0, cumulated_claim_amount[group_ends] / cumulated_claim_amount[-1]
    ]
    return cumulated_samples, cumulated_claim_amount


def comparison_curves(test, predictions_by_model):
    """Compute Lorenz curves and unnormalized Gini on the common test policies."""
    if not predictions_by_model:
        raise ValueError("At least one model's test predictions are required")
    curves = {}
    for model, predictions in predictions_by_model.items():
        y, pred, weights = evaluation_arrays(test, predictions)
        x, claims = lorenz_curve(y, pred, weights)
        curves[MODEL_NAMES[model]] = (x, claims, 1 - 2 * auc(x, claims))

    y, _, weights = evaluation_arrays(test, test[TARGET], allow_zero=True)
    x, claims = lorenz_curve(y, y, weights)
    curves["Oracle"] = (x, claims, 1 - 2 * auc(x, claims))
    return curves


def plot_comparison(curves):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    for label, (x, claims, gini) in curves.items():
        style = "-." if label == "Oracle" else "-"
        ax.plot(x, claims, linestyle=style, label=f"{label} (Gini: {gini:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random (Gini: 0)")
    ax.set(
        title="Test-set Lorenz curves",
        xlabel="Fraction of policies ordered from lowest to highest predicted risk",
        ylabel="Fraction of total claim amount",
    )
    ax.legend(loc="upper left")
    return fig


def save_comparison(train, test, results_dir):
    """Validate files against the official test data before plotting."""
    results_dir = Path(results_dir)
    metrics = read_model_metrics(results_dir)
    fingerprint = split_fingerprint(train, test)
    predictions = {}
    for row in metrics.itertuples(index=False):
        if (row.n_train, row.n_test) != (len(train), len(test)) or (
            pd.notna(row.split_fingerprint) and row.split_fingerprint != fingerprint
        ):
            raise ValueError(f"{row.model_name} metrics do not match the official split")
        path = results_dir / f"{row.model}_predictions.csv"
        if not path.is_file():
            continue
        series = pd.read_csv(path, index_col="IDpol")["PurePremiumPredicted"]
        if not series.index.is_unique or not series.index.sort_values().equals(
            test.index.sort_values()
        ):
            raise ValueError(f"{path.name} does not cover the official test policies")
        series = series.reindex(test.index)
        score = evaluate_sample(test, series)
        if not np.isclose(score["deviance"], row.test_deviance, rtol=1e-8, atol=1e-8):
            raise ValueError(f"{path.name} disagrees with the recorded test deviance")
        for field in ("actual_claim_amount", "predicted_claim_amount"):
            recorded_value = getattr(row, f"test_{field}")
            if pd.notna(recorded_value) and not np.isclose(score[field], recorded_value, rtol=1e-8, atol=1e-8):
                raise ValueError(f"{path.name} disagrees with recorded claim totals")
        predictions[row.model] = series

    available_models = set(metrics["model"])
    missing_metrics = [
        name for model, name in MODEL_NAMES.items() if model not in available_models
    ]
    missing_predictions = metrics.loc[~metrics["model"].isin(predictions), "model_name"]
    if missing_metrics:
        print("Metrics unavailable for " + ", ".join(missing_metrics))
    if not missing_predictions.empty:
        print("Test predictions unavailable for " + ", ".join(missing_predictions))
    if not predictions:
        print(f"Saved {write_metrics_csv(results_dir)} without a figure")
        return

    curves = comparison_curves(test, predictions)
    gini = {name: values[2] for name, values in curves.items()}
    # Persist only our evaluated records, after verifying the actual predictions.
    # This keeps an offline JSON summary from erasing the published Gini values.
    for model in predictions:
        evaluated = results_dir / "evaluated" / f"{model}_metrics.json"
        if evaluated.is_file():
            record = json.loads(evaluated.read_text(encoding="utf-8"))
            record["test_gini"] = gini[MODEL_NAMES[model]]
            record["gini_definition"] = "policy-count axis; tied predictions grouped; 1 - 2*AUC; unnormalized"
            evaluated.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fig = plot_comparison(curves)
    path = results_dir / "lorenz_comparison.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"Saved {path}")
    for name, value in gini.items():
        print(f"{name} test Gini: {value:.6f}")
    print(f"Saved {write_metrics_csv(results_dir, gini_by_model=gini)}")

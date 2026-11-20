import json

import numpy as np
import pandas as pd
import pytest

from analyses import part_c
from analyses.model_comparison import (
    comparison_curves,
    lorenz_curve,
    read_model_metrics,
    save_comparison,
    write_metrics_csv,
)
from analyses.ps3_script import load_model_data
from ps3.data import create_sample_column


@pytest.fixture
def policies():
    return pd.DataFrame(
        {
            **{column: ["A", "A", "B", "B"] for column in part_c.CATEGORICALS},
            "BonusMalus": [70, 80, 85, 90],
            "Density": [10, 100, 20, 25],
            "IDpol": [0, 1, 7, 12],
            "Exposure": [0.25, 1.0, 0.5, 1.0],
            "ClaimAmountCut": [0.0, 4.0, 0.5, 2.0],
        }
    )


@pytest.fixture
def samples(policies):
    frame = part_c.prepare_data(policies.assign(sample=["train", "train", "test", "test"]))
    return part_c.sample_frames(frame)


def test_prepare_data_uses_loader_and_preserves_features(policies, monkeypatch):
    monkeypatch.setattr(part_c, "load_transform", lambda: policies)
    frame = part_c.prepare_data()
    pd.testing.assert_frame_equal(frame[policies.columns], policies)
    assert frame["PurePremium"].tolist() == [0.0, 4.0, 1.0, 2.0]
    assert "PurePremium" not in policies


def test_prepare_data_checks_required_columns(policies):
    with pytest.raises(ValueError, match="Density"):
        part_c.prepare_data(policies.drop(columns="Density"))


@pytest.mark.parametrize("exposure", [0.0, -1.0, np.nan, np.inf])
def test_invalid_exposure_is_rejected(policies, exposure):
    policies.loc[0, "Exposure"] = exposure
    with pytest.raises(ValueError, match="Exposure"):
        part_c.prepare_data(policies)


def test_shared_membership_uses_a_split(policies):
    data = load_model_data(policies)
    expected = create_sample_column(policies, id_columns="IDpol", training_frac=0.8)
    pd.testing.assert_series_equal(data["sample"], expected["sample"])
    shuffled = load_model_data(policies.iloc[::-1])
    pd.testing.assert_series_equal(data["sample"], shuffled["sample"].reindex(data.index))


def test_invalid_sample_labels_are_rejected(policies):
    frame = part_c.prepare_data(policies)
    with pytest.raises(ValueError, match="valid labels"):
        part_c.sample_frames(frame.assign(sample=["train", "test", None, "test"]))


def test_no_baseline_without_official_sample_column(policies):
    with pytest.raises(ValueError, match="official split"):
        part_c.run_baseline(part_c.prepare_data(policies))


def test_weighted_deviance_and_claim_totals(samples):
    train, _ = samples
    preds = pd.Series([1.0, 4.0], index=train.index)
    scores = part_c.evaluate_sample(train, preds)
    # At p=1.5 the individual deviances are 4 and 0, with weights 0.25 and 1.
    assert scores == pytest.approx(
        {"deviance": 0.8, "actual_claim_amount": 4.0, "predicted_claim_amount": 4.25}
    )


def test_prediction_validation(samples):
    train, _ = samples
    with pytest.raises(ValueError, match="one-dimensional"):
        part_c.evaluate_sample(train, np.ones((2, 1)))
    with pytest.raises(ValueError, match="match"):
        part_c.evaluate_sample(train, pd.Series([4.0, 1.0], index=train.index[::-1]))
    with pytest.raises(ValueError, match="match"):
        part_c.evaluate_sample(train, pd.Series([1.0], index=train.index[:1]))
    for value in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="strictly positive"):
            part_c.evaluate_sample(train, pd.Series([1.0, value], index=train.index))


def test_train_test_metrics_and_aggregation(samples, tmp_path):
    train, test = samples
    train_predictions = pd.Series([1.0, 4.0], index=train.index)
    test_predictions = pd.Series([1.0, 2.0], index=test.index)
    with pytest.raises(FileNotFoundError, match="No model metrics"):
        write_metrics_csv(tmp_path)
    assert not (tmp_path / "metrics.csv").exists()

    for model in part_c.MODEL_NAMES:
        metrics = part_c.model_metrics(
            model,
            train,
            test,
            train_predictions,
            test_predictions,
            parameters={"example": 1},
            selection_method="synthetic test",
        )
        part_c.write_model_metrics(metrics, tmp_path)
    comparison = pd.read_csv(write_metrics_csv(tmp_path))
    assert list(comparison["model"]) == list(part_c.MODEL_NAMES)
    assert list(comparison["train_deviance"]) == pytest.approx([0.8] * 4)
    assert list(comparison["test_deviance"]) == pytest.approx([0.0] * 4)
    assert json.loads(comparison["parameters"].iloc[-1]) == {"example": 1}

    metrics["split_fingerprint"] = "0" * 64
    part_c.write_model_metrics(metrics, tmp_path)
    with pytest.raises(ValueError, match="different policy assignments"):
        read_model_metrics(tmp_path)


def test_overlapping_samples_are_rejected(samples):
    train, _ = samples
    predictions = pd.Series([1.0, 4.0], index=train.index)
    with pytest.raises(ValueError, match="disjoint"):
        part_c.model_metrics(
            "tweedie_glm", train, train, predictions, predictions,
            parameters={}, selection_method="synthetic test",
        )


def test_lorenz_curves_keep_initial_and_tuned_predictions_separate(samples):
    _, test = samples
    predictions = {
        model: pd.Series([1.0, 2.0], index=test.index) for model in part_c.MODEL_NAMES
    }
    predictions["lightgbm_initial"] = pd.Series([2.0, 1.0], index=test.index)
    curves = comparison_curves(test, predictions)
    x, claims, gini = curves["Tuned LightGBM"]
    np.testing.assert_allclose(x, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(claims, [0.0, 0.2, 1.0])
    assert gini == pytest.approx(0.3)
    assert curves["Initial LightGBM"][2] == pytest.approx(-0.3)
    assert curves["Oracle"][2] == pytest.approx(gini)

    partial = comparison_curves(test, {"lightgbm_tuned": predictions["lightgbm_tuned"]})
    assert set(partial) == {"Tuned LightGBM", "Oracle"}
    with pytest.raises(ValueError, match="At least one"):
        comparison_curves(test, {})


def test_benchmark_with_real_glum_and_categorizer(policies):
    glum = pytest.importorskip("glum")
    pytest.importorskip("dask_ml.preprocessing")
    rng = np.random.default_rng(18)
    frame = pd.concat([policies] * 20, ignore_index=True)
    frame["IDpol"] = np.arange(len(frame))
    frame["BonusMalus"] = rng.uniform(70, 95, len(frame))
    frame["Density"] = rng.uniform(20, 250, len(frame))
    frame["ClaimAmountCut"] = rng.uniform(1, 10, len(frame))
    frame = load_model_data(frame)
    metrics, coefficients, train_predictions, test_predictions = part_c.run_baseline(
        frame
    )
    assert metrics["n_train"] + metrics["n_test"] == 80
    assert "intercept" in coefficients
    assert "Density" in coefficients
    train, test = part_c.sample_frames(frame)
    distribution = glum.TweedieDistribution(1.5)
    for sample, predictions, key in (
        (train, train_predictions, "train_deviance"),
        (test, test_predictions, "test_deviance"),
    ):
        expected = distribution.deviance(
            sample["PurePremium"].to_numpy(),
            predictions.to_numpy(),
            sample_weight=sample["Exposure"].to_numpy(),
        ) / sample["Exposure"].sum()
        assert predictions.index.equals(sample.index)
        assert metrics[key] == pytest.approx(expected)


def test_reads_distinct_recorded_a_and_e_metrics(tmp_path):
    source = {
        "split": part_c.SPLIT,
        "train_rows": 2,
        "test_rows": 3,
        "train_weighted_tweedie_deviance": 1.0,
        "test_weighted_tweedie_deviance": 1.2,
        "target": part_c.TARGET,
        "target_definition": part_c.TARGET_DEFINITION,
        "weights": part_c.WEIGHT,
        "tweedie_power": part_c.TWEEDIE_POWER,
        "parameters": {"example": 1},
        "selection_method": "synthetic test",
    }
    baseline = dict(source, model="LightGBM Tweedie baseline")
    tuned = dict(
        source,
        model="LightGBM Tweedie tuned",
        test_weighted_tweedie_deviance=1.1,
    )
    (tmp_path / "lgbm_baseline_metrics.json").write_text(json.dumps(baseline))
    path = tmp_path / "lgbm_tuning_metrics.json"
    path.write_text(json.dumps(tuned))
    rows = read_model_metrics(tmp_path)
    assert rows["model"].tolist() == ["lightgbm_initial", "lightgbm_tuned"]
    assert rows["test_deviance"].tolist() == [1.2, 1.1]
    assert rows["split_fingerprint"].isna().all()
    assert rows["source"].tolist() == [
        "lgbm_baseline_metrics.json (A's recorded run)",
        "lgbm_tuning_metrics.json (E's recorded run)",
    ]

    tuned["tweedie_power"] = 1.8
    path.write_text(json.dumps(tuned))
    with pytest.raises(ValueError, match="Inconsistent split or evaluation convention"):
        read_model_metrics(tmp_path)


def test_save_partial_comparison_validates_predictions_and_writes_figure(samples, tmp_path):
    train, test = samples
    train_predictions = pd.Series([1.0, 4.0], index=train.index)
    test_predictions = pd.Series([1.0, 2.0], index=test.index)
    model = "tweedie_glm"
    metrics = part_c.model_metrics(
        model, train, test, train_predictions, test_predictions,
        parameters={}, selection_method="synthetic test",
    )
    part_c.write_model_metrics(metrics, tmp_path)
    path = tmp_path / f"{model}_predictions.csv"
    save_comparison(train, test, tmp_path)
    assert not (tmp_path / "lorenz_comparison.png").exists()

    test_predictions.rename("PurePremiumPredicted").to_csv(path)
    save_comparison(train, test, tmp_path)
    assert (tmp_path / "lorenz_comparison.png").stat().st_size > 0
    [row] = pd.read_csv(tmp_path / "metrics.csv").to_dict("records")
    assert row["test_gini"] == pytest.approx(0.3)

    (test_predictions * 1.1).rename("PurePremiumPredicted").to_csv(path)
    with pytest.raises(ValueError, match="recorded test deviance"):
        save_comparison(train, test, tmp_path)


def test_initial_lightgbm_with_real_estimator(policies):
    pytest.importorskip("lightgbm")
    frame = pd.concat([policies] * 20, ignore_index=True)
    frame["IDpol"] = np.arange(len(frame))
    frame = load_model_data(frame)
    metrics, predictions = part_c.run_initial_lightgbm(frame)
    train, test = part_c.sample_frames(frame)
    assert predictions.index.equals(test.index)
    assert metrics["split_fingerprint"] == part_c.split_fingerprint(train, test)
    assert metrics["test_deviance"] == part_c.evaluate_sample(test, predictions)["deviance"]


def test_lorenz_ties_are_order_independent_and_constant_prediction_is_random():
    from sklearn.metrics import auc

    claims = np.array([0.0, 0.0, 0.0, 100.0])
    exposure = np.ones(4)
    prediction = np.ones(4)
    x, y = lorenz_curve(claims, prediction, exposure)
    reversed_x, reversed_y = lorenz_curve(claims[::-1], prediction[::-1], exposure)
    np.testing.assert_allclose(x, [0, 1])
    np.testing.assert_allclose(y, [0, 1])
    np.testing.assert_allclose(reversed_x, x)
    np.testing.assert_allclose(reversed_y, y)
    assert 1 - 2 * auc(x, y) == pytest.approx(0)

    scores = np.array([1.0, 1.0, 2.0, 2.0])
    expected_x, expected_y = lorenz_curve(claims, scores, exposure)
    order = np.array([1, 0, 3, 2])
    actual_x, actual_y = lorenz_curve(claims[order], scores[order], exposure[order])
    np.testing.assert_allclose(actual_x, expected_x)
    np.testing.assert_allclose(actual_y, expected_y)


def test_model_metrics_rejects_alternate_split(policies):
    frame = pd.concat([policies] * 10, ignore_index=True)
    frame["IDpol"] = np.arange(len(frame))
    frame = create_sample_column(part_c.prepare_data(frame), "IDpol", training_frac=0.5)
    train, test = part_c.sample_frames(frame)
    with pytest.raises(ValueError, match="do not match the official split"):
        part_c.model_metrics(
            "parametric_glm", train, test,
            pd.Series(1.0, index=train.index), pd.Series(1.0, index=test.index),
            parameters={}, selection_method="fixture",
        )


def test_comparison_allows_provisional_record_with_verified_predictions(samples, tmp_path):
    train, test = samples
    train_predictions = pd.Series([1.0, 4.0], index=train.index)
    test_predictions = pd.Series([1.0, 2.0], index=test.index)
    metrics = part_c.model_metrics(
        "tweedie_glm", train, test, train_predictions, test_predictions,
        parameters={}, selection_method="fixture",
    )
    part_c.write_model_metrics(metrics, tmp_path)
    test_predictions.rename("PurePremiumPredicted").to_csv(
        tmp_path / "tweedie_glm_predictions.csv"
    )
    recorded = {
        "model": "LightGBM Tweedie tuned", "split": part_c.SPLIT,
        "train_rows": len(train), "test_rows": len(test),
        "train_weighted_tweedie_deviance": 1.0,
        "test_weighted_tweedie_deviance": 1.2,
        "target": part_c.TARGET, "target_definition": part_c.TARGET_DEFINITION,
        "weights": part_c.WEIGHT, "tweedie_power": part_c.TWEEDIE_POWER,
        "parameters": {}, "selection_method": "recorded",
    }
    (tmp_path / "lgbm_tuning_metrics.json").write_text(json.dumps(recorded))
    save_comparison(train, test, tmp_path)
    assert (tmp_path / "lorenz_comparison.png").is_file()
    comparison = pd.read_csv(tmp_path / "metrics.csv").set_index("model")
    assert pd.isna(comparison.loc["lightgbm_tuned", "split_fingerprint"])
    assert "E's recorded run" in comparison.loc["lightgbm_tuned", "source"]
    assert pd.isna(comparison.loc["lightgbm_tuned", "test_gini"])

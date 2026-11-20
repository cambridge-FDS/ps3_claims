"""Fit and compare Part D's Tweedie GLMs on the prepared policy data."""

import argparse

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution
from sklearn.base import OneToOneFeatureMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)

from ps3.data import create_sample_column, load_transform
from ps3.preprocessing import Winsorizer


class SplineWinsorizer(OneToOneFeatureMixin, Winsorizer):
    """Enable pandas output for the shared Winsorizer in the spline pipeline."""

    def fit(self, X, y=None):
        super().fit(X, y)
        self.feature_names_in_ = X.columns.to_numpy()
        return self


CATEGORICALS = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge"]
TWEEDIE = TweedieDistribution(1.5)

QUADRATIC = PolynomialFeatures(degree=2, include_bias=False)
CUBIC = make_pipeline(
    StandardScaler(), PolynomialFeatures(degree=3, include_bias=False)
)
QUARTIC = make_pipeline(
    StandardScaler(), PolynomialFeatures(degree=4, include_bias=False)
)
# Uniform knots avoid duplicate quantiles at the most common observed values.
BONUS_SPLINE = SplineTransformer(
    n_knots=4, degree=3, knots="uniform", include_bias=False
)
CAPPED_BONUS_SPLINE = make_pipeline(
    SplineWinsorizer(lower_quantile=0, upper_quantile=0.99), clone(BONUS_SPLINE)
)
POWER_SPLINE = SplineTransformer(
    n_knots=3, degree=3, knots="uniform", include_bias=False
)

VARIANTS = {
    "quadratic": (QUADRATIC, QUADRATIC),
    "spline": (BONUS_SPLINE, POWER_SPLINE),
    "capped_spline": (CAPPED_BONUS_SPLINE, POWER_SPLINE),
    "spline_quadratic": (BONUS_SPLINE, QUADRATIC),
    "quadratic_spline": (QUADRATIC, POWER_SPLINE),
    "spline_cubic": (BONUS_SPLINE, CUBIC),
    "cubic_spline": (CUBIC, POWER_SPLINE),
    "cubic": (CUBIC, CUBIC),
    "quartic": (QUARTIC, QUARTIC),
}


def make_model(variant="quadratic"):
    bonus_transform, power_transform = VARIANTS[variant]
    preprocessor = ColumnTransformer(
        transformers=[
            ("bonus", clone(bonus_transform), ["BonusMalus"]),
            ("power", clone(power_transform), ["VehPower"]),
            ("cat", OneHotEncoder(sparse_output=False, drop="first"), CATEGORICALS),
            ("density", "passthrough", ["Density"]),
        ]
    ).set_output(transform="pandas")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "estimate",
                GeneralizedLinearRegressor(
                    family=TWEEDIE, alpha=0.0, solver="irls-ls", fit_intercept=True
                ),
            ),
        ]
    )


def fit_model(train, variant):
    model = make_model(variant)
    return model.fit(
        train, train["PurePremium"], estimate__sample_weight=train["Exposure"]
    )


def run_parametric_glm(frame):
    """Fit the quadratic handoff model and return metrics and IDpol-indexed predictions."""
    from analyses.part_c import model_metrics, sample_frames

    train, test = sample_frames(frame)
    model = fit_model(train, "quadratic")
    estimate = model.named_steps["estimate"]
    train_predictions = pd.Series(model.predict(train), index=train.index)
    test_predictions = pd.Series(model.predict(test), index=test.index)
    metrics = model_metrics(
        "parametric_glm",
        train,
        test,
        train_predictions,
        test_predictions,
        parameters={
            "family": "TweedieDistribution",
            "polynomial_degree": {"BonusMalus": 2, "VehPower": 2},
            "alpha": estimate.alpha,
            "solver": estimate.solver,
            "fit_intercept": estimate.fit_intercept,
        },
        selection_method=(
            "Quadratic terms selected for stable predictions in the sparse BonusMalus tail."
        ),
    )
    return metrics, train_predictions, test_predictions


def weighted_deviance(frame, predictions):
    exposure = frame["Exposure"]
    return TWEEDIE.deviance(
        frame["PurePremium"], predictions, sample_weight=exposure
    ) / exposure.sum()


def compare_variants(train):
    splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(splitter.split(train))
    results = []
    for variant in VARIANTS:
        fold_scores = []
        for fit_rows, validation_rows in folds:
            model = fit_model(train.iloc[fit_rows], variant)
            validation = train.iloc[validation_rows]
            fold_scores.append(weighted_deviance(validation, model.predict(validation)))
        results.append((variant, *fold_scores, np.mean(fold_scores)))

    scores = pd.DataFrame(
        results, columns=["variant", "fold_1", "fold_2", "fold_3", "mean_cv_deviance"]
    ).sort_values("mean_cv_deviance", ignore_index=True)
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return scores.loc[0, "variant"]


def prepare_model_data(frame, id_columns="IDpol", training_fraction=0.8):
    """Apply the shared splitter and reject a cache with obsolete assignments."""
    prepared = create_sample_column(frame, id_columns, training_fraction)
    if "sample" in frame and not np.array_equal(frame["sample"], prepared["sample"]):
        raise ValueError("Supplied sample labels differ from the current shared split")
    prepared["PurePremium"] = prepared["ClaimAmountCut"] / prepared["Exposure"]
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group()
    models.add_argument(
        "--compare",
        action="store_true",
        help="Compare variants by training-only CV and test the selected model.",
    )
    models.add_argument(
        "--cv-only",
        action="store_true",
        help="Compare variants by CV using only rows labelled train.",
    )
    models.add_argument(
        "--handoff",
        action="store_true",
        help="Fit the quadratic specification using the shared data and write metrics.",
    )
    models.add_argument("--variant", choices=VARIANTS, default="quadratic")
    parser.add_argument("--data", help="Optional prepared Parquet input.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="With --handoff, also write test predictions for the shared comparison.",
    )
    args = parser.parse_args()
    if args.write_predictions and not args.handoff:
        parser.error("--write-predictions requires --handoff")

    input_frame = pd.read_parquet(args.data) if args.data else load_transform()
    if args.handoff:
        from analyses.part_c import SPLIT, prepare_data, write_model_metrics

        frame = prepare_model_data(
            prepare_data(input_frame), SPLIT["id_columns"], SPLIT["training_fraction"]
        )
        metrics, _, test_predictions = run_parametric_glm(frame)
        metrics_path = write_model_metrics(metrics, args.results_dir)
        print(metrics_path)
        if args.write_predictions:
            prediction_path = metrics_path.with_name("parametric_glm_predictions.csv")
            test_predictions.rename("PurePremiumPredicted").to_csv(prediction_path)
            print(prediction_path)
        return

    frame = prepare_model_data(input_frame)
    if args.cv_only:
        compare_variants(frame.loc[frame["sample"].eq("train")])
        return

    train = frame.loc[frame["sample"].eq("train")]
    test = frame.loc[frame["sample"].eq("test")]
    variant = compare_variants(train) if args.compare else args.variant

    model = fit_model(train, variant)
    test_predictions = model.predict(test)
    print(f"Model: {variant}")
    print(f"Training deviance: {weighted_deviance(train, model.predict(train)):.6f}")
    print(f"Test deviance: {weighted_deviance(test, test_predictions):.6f}")
    print(f"Observed test claims: {test['ClaimAmountCut'].sum():,.2f}")
    print(f"Predicted test claims: {np.dot(test['Exposure'], test_predictions):,.2f}")


if __name__ == "__main__":
    main()

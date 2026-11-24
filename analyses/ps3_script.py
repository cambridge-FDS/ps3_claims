# %%
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data._load_transform import load_transform
from ps3.data._sample_split import create_sample_split
from ps3.data.plots import plot_weighted_avg_claims

# %%
# load data
df = load_transform()

# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?
## We want to make policies with different duration/exposure comparable.
# Exposure is then used as a weight as longer-exposed policies give less noisey observations.


# TODO: use your create_sample_split function here
df = create_sample_split(df, "IDpol", 0.8)
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
# Create train and test dataframes
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
# Function to categorize the categorical variables. This is needed for the GLM model.
glm_categorizer = Categorizer(columns=categoricals)

# Split the data into train and test sets, and apply the categorizer to the categorical variables.
X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

# Define the GLM model with a Tweedie distribution and fit it to the training data.
TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

# Use fitted model to predict on the test set
# and calculate the deviance loss for both training and testing sets.
df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps:
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer.
#    Choose knots="quantile" for the SplineTransformer and make sure, we
#    are only including one intercept in the final GLM.
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]

# Column transformer for numeric columns
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "spline",
            SplineTransformer(
                knots="quantile", degree=3, include_bias=False, extrapolation="constant"
            ),
        ),
    ]
)

# Feature preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)

preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimate", t_glm1)])

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

# Fit pipeline to training data
model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.

# Model pipeline with LGBMRegressor
model_pipeline = Pipeline(
    steps=[
        (
            "estimate",
            LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5),
        )
    ]
)

# Fit on the data and predict on the test set
model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param.

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators

## GridSearchCV uses the earlier LGBM pipeline as a template, makes copies of it
# and fits each copy with a different combination of the hyperparameters specified in the param_grid.
# It then evaluates the performance of each model using cross-validation
# and selects the best combination of hyperparameters based on the specified scoring metric.

cv = GridSearchCV(
    model_pipeline,
    param_grid={
        "estimate__learning_rate": [0.01, 0.05, 0.1],
        "estimate__n_estimators": [50, 100, 150, 200],
    },
    cv=5,
    verbose=2,
)
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

######## PS4 #############
# %%
# Exercise 1

# Observed average claims per BonusMalus group, weighted by exposure.
plot_weighted_avg_claims(
    df,
    group_col="BonusMalus",
    claims_col="PurePremium",
    weight_col="Exposure",
)
plt.show()
## We can see that the average claims are increasing with increasing BonusMalus, which is what we would expect.
## Hence we need to impose a monotonicity constraint on the BonusMalus variable in our model to ensure
## that the predicted pure premium is increasing with increasing BonusMalus.


# %%
# New LGBM model pipeline with monotonicity constraint on BonusMalus

# One entry per feature, in the column order of X_train_t:
# 1 = increasing, 0 = unconstrained.
monotone_constraints = [1 if col == "BonusMalus" else 0 for col in X_train_t.columns]

# Hyperparameters are tuned with 3-fold cross-validation on the training data.
# The same folds are used for every trial, and the test set is not used for tuning.
cv_folds = list(KFold(n_splits=3, shuffle=True, random_state=0).split(X_train_t))


def make_constrained_lgbm(**params):
    """Tweedie LGBM pipeline with an increasing constraint on BonusMalus."""
    return Pipeline(
        steps=[
            (
                "estimate",
                LGBMRegressor(
                    objective="tweedie",
                    tweedie_variance_power=1.5,
                    monotone_constraints=monotone_constraints,
                    verbose=-1,
                    **params,
                ),
            )
        ]
    )


def objective(trial):
    """Cross-validated Tweedie deviance per unit of exposure (lower is better)."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
    }
    fold_scores = []
    for fit_idx, val_idx in cv_folds:
        candidate = make_constrained_lgbm(**params)
        candidate.fit(
            X_train_t.iloc[fit_idx],
            y_train_t.iloc[fit_idx],
            estimate__sample_weight=w_train_t[fit_idx],
        )
        # Exposure-weighted deviance on the held-out fold, per unit of exposure
        fold_scores.append(
            TweedieDist.deviance(
                y_train_t.iloc[val_idx],
                candidate.predict(X_train_t.iloc[val_idx]),
                sample_weight=w_train_t[val_idx],
            )
            / np.sum(w_train_t[val_idx])
        )
    return np.mean(fold_scores)


study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=0)
)
study.optimize(objective, n_trials=20)
# The tuned hyperparameters are saved in `study.best_params`
print("best parameters:", study.best_params)

# Refit the best configuration on the full training data
# The LGBM with the tuned hyperparameters is built here: `**study.best_params`
# unpacks the dictionary into make_constrained_lgbm. It is stored in
# `constrained_lgbm`, and fitted on the next line.
constrained_lgbm = make_constrained_lgbm(**study.best_params)
# This is the fit of the tuned, constrained LGBM on the full training data.
constrained_lgbm.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm_constrained"] = constrained_lgbm.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = constrained_lgbm.predict(X_train_t)

print(
    "training loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(
            y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t
        )
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(
            y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t
        )
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)

# %%
# Exercise 2
#
# What is done here:
# 1. The best hyperparameters from the Optuna cross-validation (`study.best_params`)
#    are used again, so this is the same tuned, constrained model as above.
# 2. The model is re-fitted on the training data. This time the train and test sets
#    are also passed as `eval_set`, a list of (X, y) tuples. After every boosting
#    round (every added tree), LightGBM records the loss (how far predictions are from the actual values)
# on each of them.
#    (Newer LightGBM versions warn that `eval_set` is deprecated in favour of
#    `eval_X` and `eval_y`; `eval_set` still works.)
# 3. `eval_sample_weight` gives the exposure weights for the two eval sets, so the
#    recorded losses are exposure-weighted, like the training itself.
# 4. The test set is only monitored here. It is not used to train the model or to
#    choose any setting (there is no early stopping), so the model is unchanged.
constrained_lgbm = make_constrained_lgbm(**study.best_params)
constrained_lgbm.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
    estimate__eval_sample_weight=[w_train_t, w_test_t],
    estimate__eval_names=["train", "test"],
)

# The loss after each boosting round is stored on the fitted LGBM (the "estimate"
# step of the pipeline): {"train": {metric: [...]}, "test": {metric: [...]}}.
evals_result = constrained_lgbm.named_steps["estimate"].evals_result_
metric = next(iter(evals_result["train"]))  # LightGBM's Tweedie metric
print("recorded metric:", metric)

# Learning curves: loss against the number of trees for the train and test set,
# drawn with LightGBM's own `lgb.plot_metric`. It needs the fitted LGBM and not the
# pipeline, so we pass the last step of the pipeline (`constrained_lgbm[-1]`).
# A test curve that flattens or rises while the train curve keeps falling signals
# overfitting.
ax = lgb.plot_metric(constrained_lgbm[-1], title="Learning curves: constrained LGBM")
plt.show()

## the test loss plateaus without rising, so the model is close to optimally tuned.
# The widening train–test gap and the flat test curve after about 100 iterations shows
# diminishing returns in the number of trees.
#  A simpler model with fewer trees or more regularisation would do about as well

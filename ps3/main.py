import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.stats
from dask_ml.preprocessing import Categorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ShuffleSplit
from glum import GeneralizedLinearRegressor
from glum import TweedieDistribution
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance

from data import load_transform, create_sample_split

# read data
df = load_transform()
print(df.head())

# add sample column
df = create_sample_split(df, id_column="IDpol", training_frac=0.8)

#define X and Y
categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge"]
num_linear = ["Density"]                     # 線形のまま使う
num_poly   = ["BonusMalus", "VehPower"]  

predictors = categoricals + num_linear + num_poly
X = df[predictors]

weight = df['Exposure'].values
df["PurePremium"] = df["ClaimAmountCut"] / weight
y = df["PurePremium"].values

train_mask = df["sample"] == "train"
test_mask = df["sample"] == "test"

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
w_train, w_test = weight[train_mask], weight[test_mask]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricals),
        ("num_linear", "passthrough", num_linear),
        ("num_poly", PolynomialFeatures(degree=2, include_bias=False), num_poly),
    ]
)

model = TweedieRegressor(power=1.5, alpha=1.0, link="log")

glm_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", model),
])

# 3. 学習
glm_pipe.fit(X_train, y_train, model__sample_weight=w_train)

# 4. 損失（deviance）で性能を見る
y_pred_train = glm_pipe.predict(X_train)
y_pred_test = glm_pipe.predict(X_test)

train_loss = mean_tweedie_deviance(y_train, y_pred_train, power=1.5, sample_weight=w_train)
test_loss = mean_tweedie_deviance(y_test, y_pred_test, power=1.5, sample_weight=w_test)

print("Tweedie pure premium – train deviance:", train_loss)
print("Tweedie pure premium – test deviance :", test_loss)

###　LGBM追加
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import make_scorer

# 1. LightGBM のベースモデル（Tweedie 用）
lgbm_base = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    boosting_type="gbdt",
    random_state=42,
    n_estimators=500,      # 後で調整されるかもしれない
)

# 2. GLM と同じ前処理をそのまま再利用
#    preprocess はすでに定義済みなので、そのまま使える
lgbm_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", lgbm_base),
])

# 3. Tweedie deviance を最小化する scorer を作る
tweedie_scorer = make_scorer(
    mean_tweedie_deviance,
    greater_is_better=False,  # deviance は小さい方が良いので False
    power=1.5,
)

# 4. ハイパーパラメータ探索の範囲（過学習を抑えるパラメータ中心）
param_dist = {
    "model__num_leaves": [15, 31, 63],         # 木の複雑さ
    "model__min_child_samples": [50, 100, 200],# 各葉の最小サンプル数（大きいほど過学習抑制）
    "model__subsample": [0.7, 0.8, 0.9],       # 行サブサンプリング
    "model__colsample_bytree": [0.7, 0.8, 0.9],# 列サブサンプリング
    "model__learning_rate": [0.03, 0.05, 0.1], # 学習率
    "model__n_estimators": [300, 500, 800],    # 木の本数
}

# 5. K-fold CV の設定
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 6. ランダムサーチでチューニング
search = RandomizedSearchCV(
    estimator=lgbm_pipe,
    param_distributions=param_dist,
    n_iter=15,              # 試す組み合わせの数（多すぎると時間がかかる）
    scoring=tweedie_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

# 7. 学習（GLM と同じく exposure を sample_weight に）
search.fit(X_train, y_train, model__sample_weight=w_train)

#print("Best params (LGBM):", search.best_params_)
#print("Best CV score (negative deviance):", search.best_score_)

# 8. チューニング済みのベストモデルで train/test deviance を計算
best_lgbm = search.best_estimator_

y_pred_train_lgbm = best_lgbm.predict(X_train)
y_pred_test_lgbm  = best_lgbm.predict(X_test)

train_loss_lgbm = mean_tweedie_deviance(
    y_train, y_pred_train_lgbm, power=1.5, sample_weight=w_train
)
test_loss_lgbm = mean_tweedie_deviance(
    y_test, y_pred_test_lgbm, power=1.5, sample_weight=w_test
)

print("GLM  Tweedie – train deviance:", train_loss)
print("GLM  Tweedie – test  deviance:", test_loss)
print("LGBM Tweedie – train deviance:", train_loss_lgbm)
print("LGBM Tweedie – test  deviance:", test_loss_lgbm)

# 3. 特徴量の選び方
#    ★ BonusMalus を 先頭列 に置く（単調制約をかけやすくするため）
num_cols = ["BonusMalus", "VehPower", "Density"]
cat_cols = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge"]

feature_cols = num_cols + cat_cols

# カテゴリカルは LightGBM 用に category 型にしておく
df[cat_cols] = df[cat_cols].astype("category")

X_lgbm_con = df[feature_cols]

# 4. train / test 分割 ---------------------------------------------
X_train_con, X_test_con = X_lgbm_con[train_mask], X_lgbm_con[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
w_train, w_test = weight[train_mask], weight[test_mask]

# 5. 単調性制約付き LGBM モデル -------------------------------------
#    feature_cols の並び：
#    0: BonusMalus（ここだけ単調増加） → 1
#    1: VehPower → 0
#    2: Density  → 0
#    3以降のカテゴリカル → 0
monotone_constraints = [1] + [0] * (len(feature_cols) - 1) # [1,0,0,0,…]

lgbm_constrained = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    learning_rate=0.05,
    n_estimators=500,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    monotone_constraints=monotone_constraints,
)

# 6. 学習（BonusMalus に単調増加制約が入った LGBM）
#    categorical_feature に列名を渡すと、カテゴリカルとして扱われます
lgbm_constrained.fit(
    X_train_con,
    y_train,
    sample_weight=w_train,
    categorical_feature=cat_cols,
)

# 7. deviance で性能を見る -----------------------------------------
y_pred_train_c = lgbm_constrained.predict(X_train_con)
y_pred_test_c = lgbm_constrained.predict(X_test_con)

train_loss_c = mean_tweedie_deviance(
    y_train, y_pred_train_c, power=1.5, sample_weight=w_train
)
test_loss_c = mean_tweedie_deviance(
    y_test, y_pred_test_c, power=1.5, sample_weight=w_test
)

print("Constrained LGBM – train deviance:", train_loss_c)
print("Constrained LGBM – test  deviance:", test_loss_c)

# ==========================
# Explainer の作成 <説明用の便利関数をまとめて提供してくれる窓口>
# ==========================
import dalex as dx
import matplotlib
matplotlib.use("Agg")

# 1. GLM（Tweedie）パイプライン
exp_glm = dx.Explainer(
    glm_pipe,      # ← すでに定義済みの GLM パイプライン
    X_train,
    y_train,
    label="GLM Tweedie"
)

# 2. unconstrained LGBM（前処理込みのベスト推定器）
exp_lgbm_uncon = dx.Explainer(
    best_lgbm,     # ← RandomizedSearchCV の best_estimator_
    X_train,
    y_train,
    label="LGBM unconstrained"
)

# 3. constrained LGBM（カテゴリカルを直接受けるモデル）
exp_lgbm_con = dx.Explainer(
    lgbm_constrained,  # ← 単調性制約付き LGBM
    X_train_con,
    y_train,
    label="LGBM constrained"
)

# ==========================
# Ex4: Partial Dependence Plot for BonusMalus
# ==========================

pdp_glm      = exp_glm.model_profile(variables=["BonusMalus"], type="partial")
pdp_lgbm_u   = exp_lgbm_uncon.model_profile(variables=["BonusMalus"], type="partial")
pdp_lgbm_con = exp_lgbm_con.model_profile(variables=["BonusMalus"], type="partial")

# ★ それぞれ別々に描画する（重ねなくてOK）
pdp_glm.plot(geom="profiles")
pdp_lgbm_u.plot(geom="profiles")
pdp_lgbm_con.plot(geom="profiles")


# ==========================
# Ex5: SHAP (Shapley values) の比較
# ==========================

# 説明したい観測を1つ選ぶ（例：test の先頭）
x0_glm = X_test.iloc[[0]]        # GLM / unconstrained LGBM 用
x0_con = X_test_con.iloc[[0]]    # constrained LGBM 用

# GLM の SHAP
shap_glm = exp_glm.predict_parts(
    new_observation=x0_glm,
    type="shap"
)
shap_glm.plot()

# constrained LGBM の SHAP
shap_con = exp_lgbm_con.predict_parts(
    new_observation=x0_con,
    type="shap"
)
shap_con.plot()
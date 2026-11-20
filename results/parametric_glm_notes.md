# Part D · Quadratic Tweedie GLM

The quadratic GLM uses linear and squared terms for both `BonusMalus` and `VehPower`. It models pure premium (`ClaimAmountCut / Exposure`) with Tweedie power 1.5, an intercept, and exposure weights. In C's benchmark, `BonusMalus` is linear and `VehPower` is categorical.

## Comparison on the normalized ID split

All models use the normalized `IDpol` split from PR #10 (merged as `c5b3b6b`), with 542,286 training policies and 135,727 test policies. Scores are exposure-weighted mean Tweedie deviances, for which lower is better.

| Specification | Train | Test |
|---|---:|---:|
| C's Tweedie benchmark | 73.4855 | 74.8970 |
| **Quadratic for both variables** | **73.4660** | **74.8145** |
| Uncapped splines for both variables | 73.3371 | 74.6672 |
| Uncapped spline for `BonusMalus`, quadratic for `VehPower` | 73.3561 | 74.7278 |
| Fourth-degree polynomials for both variables | 73.3414 | 74.6549 |
| Capped `BonusMalus` spline (exploratory) | 73.1470 | 74.7218 |

The quadratic handoff reduces test deviance by 0.11% relative to C's benchmark. Among the nine Part D variants, the uncapped spline improves on the quadratic in both training-only CV (74.0698 versus 74.1664) and observed test deviance. The quartic has the lowest test deviance of the variants, but it has the highest mean CV deviance (74.4156). We retain the agreed quadratic model for the handoff and report the alternatives as exploratory results.

## Checking overfitting

Compare train and test deviance, assess variation across the validation folds, and inspect the fitted curves alongside training counts. On the earlier split the quartic's test deviance was 245.1259, driven by a near-zero prediction for a positive claim above the training maximum. This split has four training policies with `BonusMalus` of at least 200 and no test policy above the training maximum of 230. The new test score therefore measures a different tail and does not settle the extrapolation concern. The [sensitivity plot](parametric_glm_bonus_malus.png) holds the other predictors fixed.

The capped spline uses the training 99th percentile (106) and has the best training-only CV mean (73.9228) among the D variants. The earlier test-tail findings prompted the cap and influenced the handoff choice. Repartitioning this same dataset does not establish independent evidence for choosing a different specification.

## Reproduce and integrate

With A's shared split and C's `analyses.part_c` available, run from the repository root:

```bash
python -m analyses.parametric_glm --handoff
```

The command loads the course data and writes `results/parametric_glm_metrics.json`. For C's combined comparison, `run_parametric_glm(frame)` accepts the prepared shared frame and returns `(metrics, train_predictions, test_predictions)`. Both prediction series are indexed by `IDpol`.

For the file-based comparison, run `python -m analyses.parametric_glm --handoff --write-predictions --results-dir data/part_d_results`. This writes the metrics and `parametric_glm_predictions.csv` with columns `IDpol` and `PurePremiumPredicted`. Keep the per-policy file local. A prepared Parquet input supplied via `--data` is checked against the current shared splitter before fitting.

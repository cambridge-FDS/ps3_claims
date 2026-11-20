# Part D · Quadratic Tweedie GLM

The quadratic GLM uses linear and squared terms for both `BonusMalus` and `VehPower`. It models pure premium (`ClaimAmountCut / Exposure`) with Tweedie power 1.5, an intercept, and exposure weights. In C's benchmark, `BonusMalus` is linear and `VehPower` is categorical.

## Comparison on the current shared split

These provisional results use A's split from PR #11 (`83f6f88`) and C's shared evaluation from PR #13 (`34bd343`), with 542,605 training policies and 135,408 test policies. All scores are exposure-weighted mean Tweedie deviances, for which lower is better.

| Specification | Train | Test |
|---|---:|---:|
| C's Tweedie benchmark | 73.5260 | 74.7351 |
| **Quadratic for both variables** | **73.5302** | **74.5342** |
| Uncapped splines for both variables | 73.3440 | 74.5819 |
| Uncapped spline for `BonusMalus`, quadratic for `VehPower` | 73.3993 | 74.4882 |
| Fourth-degree polynomials for both variables | 73.3279 | 245.1259 |

We selected the quadratic specification. It reduced test deviance by 0.27% relative to C's benchmark. The spline–quadratic model scored 0.046 points lower on the test set, but it predicted about 624,409 in combined claim amounts for two policies above the maximum training `BonusMalus`, where the observed total was 1,128.12. The quadratic model predicted 622.99 for those two policies.

## Checking overfitting

Compare each model's train and test deviance using a common split, then inspect the fitted curves alongside training counts. The quartic model fit the training data better than the quadratic model and had slightly better mean three-fold validation deviance. Its fitted premium fell to nearly zero in the sparse upper tail, where the training set contained only one policy with `BonusMalus` of 200 or higher. Two test policies lay above the training maximum. Their contribution was 170.41 points of test deviance, almost entirely from the policy with a positive claim and a prediction near zero. The [sensitivity plot](parametric_glm_bonus_malus.png) holds the other predictors fixed.

As an exploratory check prompted by the test-set diagnostics, we capped `BonusMalus` at its training 99th percentile (106) before fitting the spline. Train and test deviance were 73.2155 and 74.3988. An independent sample would be needed to validate the improvement.

## Reproduce and integrate

With A's shared split and C's `analyses.part_c` available, run from the repository root:

```bash
python -m analyses.parametric_glm --handoff
```

The command loads the course data and writes `results/parametric_glm_metrics.json`. For C's combined comparison, `run_parametric_glm(frame)` accepts the prepared shared frame and returns `(metrics, train_predictions, test_predictions)`. Both prediction series are indexed by `IDpol`.

For the file-based comparison, run `python -m analyses.parametric_glm --handoff --write-predictions --results-dir data/part_d_results`. This writes the metrics and `parametric_glm_predictions.csv` with columns `IDpol` and `PurePremiumPredicted`. Keep the per-policy file local. Rerun on the agreed team split before publishing the combined results.

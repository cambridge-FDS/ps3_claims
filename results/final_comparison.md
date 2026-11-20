# Four-model comparison on the fixed train/test split

All four models use the same 678,013 instructor-transformed policies and the normalized `IDpol` split from Exercise 1, with 542,286 training and 135,727 test policies. Test clipped claims total 10,041,931.48. The target is `ClaimAmountCut / Exposure`, weights are `Exposure` and Tweedie power is 1.5. [Split implementation](../ps3/data/_sample_split.py)

| Model | Train deviance | Test deviance | Test − train | Test Gini | Predicted test claims | Relative claim error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tweedie GLM benchmark | 73.485505 | 74.897049 | 1.411544 | 0.256255 | 9,988,382.48 | -0.53% |
| Improved parametric GLM | 73.466040 | 74.814510 | 1.348469 | 0.257554 | 9,957,587.03 | -0.84% |
| Initial LightGBM | 64.577274 | 74.751332 | 10.174058 | 0.273634 | 8,324,414.59 | -17.10% |
| Tuned LightGBM | 66.412966 | 74.291639 | 7.878672 | 0.274197 | 8,605,090.30 | -14.31% |

## Interpretation and limitations

Tuned LightGBM has the lowest observed test deviance. Its Gini is only slightly above the initial LightGBM’s; no uncertainty or significance analysis was performed. D’s quadratic GLM improves slightly over the benchmark GLM in observed deviance, while both have very similar risk ranking. These differences should not be interpreted as statistically established superiority.

The initial LightGBM train–test gap is 10.174058, compared with 7.878672 after tuning. E’s training-only three-fold CV selected learning_rate=0.03 and n_estimators=300, with mean validation deviance 73.831160 versus 74.509440 for the initial parameters. Each fold fits a fresh complete preprocessing/model pipeline; the selected model is refitted on all current training policies. [Tuning notes](lgbm_tuning_notes.md)

The GLMs have much smaller aggregate test claim bias (approximately −0.53% benchmark and −0.84% quadratic) than the LightGBMs (−17.10% initial, −14.31% tuned). Aggregate totals are only one aspect of calibration and do not prove calibration within risk groups. The GLMs retain explicit coefficients/transforms; boosted trees offer nonlinearities and interactions with less direct interpretation. Large train–test gaps are diagnostics, not proof of overfitting by themselves.

**D selection limitation:** D chose the quadratic specification after inspecting test-tail behaviour on the earlier split of this same dataset. Repartitioning and refitting does not make this an independent model-selection test. D’s score remains exploratory and would need genuinely independent validation for an unbiased selection comparison. C preserves D’s handed-over method instead of choosing another specification using the new test scores. The actual preprocessing and E’s CV remain training-only.

## Lorenz/Gini conventions

Each model is ranked by predicted pure premium. The horizontal axis is fraction of policies; the vertical axis accumulates observed PurePremium × Exposure, i.e. ClaimAmountCut. Curves start at (0,0), group tied predictions and end at (1,1). Gini is unnormalized 1−2×AUC. Random is the diagonal. The teacher’s Oracle ranks observed PurePremium; with unequal exposures it is a reference, not necessarily the ordering that maximizes policy-count Gini. The final figure includes all four verified models plus Oracle and Random.

`verification_status` identifies locally refitted records. `gini_status` distinguishes newly recomputed values from previously prediction-validated values retained by an offline summary. The common comparison uses the [evaluated model JSON files](evaluated/). The [initial LightGBM](lgbm_baseline_metrics.json), [quadratic GLM](parametric_glm_metrics.json) and [tuned LightGBM](lgbm_tuning_metrics.json) handoff files use the same final split.

## Reproduce and verify

Follow the [README setup and run instructions](../README.md#install-and-run) to regenerate the comparison in a separate output directory. The published run used data from the instructor loader with the shared splitter applied to the prepared policies. Per-policy predictions, data caches and logs are local only.

GLM converged in 3 iterations (limit 100), with convergence warnings treated as errors. Both GLM sample scores agree between sklearn and glum. Independent test rescoring checked all four models against both deviance implementations, the exact policy-ID order/membership, positive weights/predictions, target equality, fingerprints and aggregate claim totals. An offline summary retained all four verified Ginis.

The full suite passed **57 tests**, including the modelling integration, Winsorizer and split tests. Pre-commit checks passed. Matplotlib emitted dependency deprecation warnings.

Split fingerprint: `48634e41e66f74a0a62ae7e54de1da7299d8abf8e78ce5f9ca941ce055e0e881`.

Data fingerprint: `fefd1e42c487594b88df12032cef35d3599b189b4ff0b06ae3ddab48006b432e`.

Each evaluated JSON records source-file hashes, versions, timestamp and source commit `350f055db87ed8c399307c81f1bc4dbd3c49ebb0`. Later report-only commits do not change the model specification.

## Model handoffs

The integrated run uses the [GLM benchmark and common evaluation](../analyses/part_c.py), the [initial LightGBM](../analyses/lgbm_baseline.py), the [quadratic GLM handoff](parametric_glm_notes.md) and the [LightGBM tuning handoff](lgbm_tuning_notes.md). The [team assignment record](../TEAM_ASSIGNMENTS.md) documents the division of work.

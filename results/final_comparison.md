# Final common model comparison — 20 November 2026

All four member implementations were refitted locally using the instructor OpenML loader, the shared PurePremium target, Exposure weights and Tweedie power 1.5. There are 542,605 training and 135,408 test policies. The test clipped claim amount is 9,978,366.81. No member model method was changed by C.

| Model | Train deviance | Test deviance | Test − train | Test Gini | Predicted test claims | Relative claim error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tweedie GLM benchmark | 73.525995 | 74.735057 | 1.209062 | 0.232849 | 9,922,993.11 | -0.55% |
| Improved parametric GLM | 73.530198 | 74.534158 | 1.003960 | 0.238912 | 9,840,562.74 | -1.38% |
| Initial LightGBM | 64.545789 | 74.669387 | 10.123598 | 0.256496 | 8,193,846.90 | -17.88% |
| Tuned LightGBM | 66.425334 | 74.138050 | 7.712716 | 0.267243 | 8,496,582.42 | -14.85% |

## Interpretation

Tuned LightGBM has the lowest observed test deviance and highest Gini in this comparison. Initial LightGBM and the benchmark GLM have very similar test deviances. No confidence interval or significance test was computed, so small differences should not be interpreted as established superiority.

The initial LightGBM train–test gap is 10.123598 versus 7.712716 after tuning. E selected learning_rate=0.03 and n_estimators=300 using training-only three-fold CV: mean validation deviance 73.723996 versus 74.264281 for initial parameters. The larger tree-model gaps are consistent with greater in-sample flexibility; a gap alone does not prove overfitting. Every fold fitted a fresh full preprocessing/model pipeline. All 18 reproduced fold scores match E’s original results within 5.7e-14.

The benchmark GLM has the smallest absolute aggregate claim error here (−0.55%); the quadratic GLM is also close (−1.38%). Both LightGBMs underpredict total claims substantially (−17.88% and −14.85%). Aggregate totals measure only one aspect of calibration and do not establish calibration within risk groups. The GLMs provide coefficients and explicit feature transforms; trees capture nonlinearities/interactions with less direct interpretation.

**D selection limitation:** D’s handover notes describe choosing the quadratic specification after inspecting test-tail behaviour and alternatives. Its test score is exploratory, not untouched validation of the model-selection procedure. No test leakage was found in the actual training preprocessing, but this selection-stage reuse cannot be undone by refitting. Independent validation would be needed for an unbiased model-selection comparison. C preserves D’s handed-over specification and reports this limitation rather than selecting another model from the test scores.

## Curves and metric conventions

The figure ranks each model’s predicted pure premiums from low to high; the horizontal axis is fraction of policies, and the vertical axis is cumulative observed PurePremium × Exposure, i.e. ClaimAmountCut. Curves start at (0,0), group tied predictions and finish at (1,1). Gini is unnormalized 1−2×AUC. Random is the diagonal. The teacher’s Oracle ranks by observed PurePremium; with unequal exposures it is a reference curve, not necessarily the mathematical maximum of a policy-count Gini.

`verification_status` labels newly fitted outputs. `gini_status` distinguishes values recomputed from policy predictions from prior prediction-validated values retained in offline summaries. E’s published historical outputs remain unchanged; this report uses the independently fitted common outputs in `results/evaluated/`.

## Reproduction and evidence

```bash
python -m analyses.ps3_script all --save-data data/shared_final.parquet
python -m analyses.ps3_script compare --data data/shared_final.parquet
python -m pytest
pre-commit run --all-files
```

Choose a new cache filename if it exists. The successful local run used `data/final_shared_20261120.parquet` and the independent compare used that cache. An earlier stalled OpenML read was interrupted; the retry succeeded with the bounded-read entry point. The loader and its transformations are unchanged. Raw data, caches, logs, per-policy predictions and previous-result archives are local only. Prior outputs are archived under `data/run_backups/` before replacement.

GLM converged in 3 iterations (limit 100), with convergence warnings treated as errors. Both sample deviances were checked against glum; the independent test rescore checked all four models against glum as well as sklearn, exact ID alignment, target equality and positive weights/predictions. The separate cached `compare` run reproduced the report and figure. An offline JSON summary also retained all four verified Ginis.

Full suite: **53 passed, no skipped tests** (including 27 C tests and B’s 12 Winsorizer tests). Pre-commit checks pass. Matplotlib emits dependency deprecation warnings, not model convergence warnings. The previously failing placeholder tests were replaced through integration of B’s existing main commits, without modifying B’s implementation or deleting tests.

Split fingerprint: `2fac3d388008e7f94443efa1f2e2fa3df35480d27f6fc1626dfd8486de79bb5e`.

Data fingerprint: `985313e2c0d33b3c75a831940d45f3321008e7a3e109eb6fc7e8eb55afca7def`.

Run provenance in each evaluated JSON records source-file hashes, package versions, timestamp and source commit `f8c03f9`. This distinguishes the executed implementation from later documentation-only commits.

## Ownership and integration dependencies

A: split and initial LightGBM (#11, 83f6f88). B: Winsorizer already on main (23569fd). E: CV/tuning (#12, 8e9340f). D: quadratic GLM (#14, 54558d1). These source files are byte-for-byte identical to their owner branches. C supplies shared orchestration, adapters, validation, tests and the common results.

`integration/part-c-dependencies` is a review base containing those histories. PR #13 targets that base to keep C’s diff focused while member PRs remain open. It is not a merge into main. The repository owner must coordinate member PR merges and later retarget C to finalized main, then rerun if data/split/model code changes. No new D handover is required for this comparison; D’s delivered model and its selection limitation are now integrated.

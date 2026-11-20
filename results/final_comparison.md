# Four-model comparison on the merged normalized-ID split — 20 November 2026

All four models were refitted together after Ex1 PR #10 merged the normalized-ID split into main (76d9970; main merge c5b3b6b). The former PR #11 split is superseded for this report: 217,325 policy assignments changed. No metric or curve from the former split is mixed into the current comparison. The same 678,013 instructor-transformed policies are used, with 542,286 training and 135,727 test policies. Test clipped claims total 10,041,931.48. The target is ClaimAmountCut / Exposure, weights are Exposure and Tweedie power is 1.5.

| Model | Train deviance | Test deviance | Test − train | Test Gini | Predicted test claims | Relative claim error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tweedie GLM benchmark | 73.485505 | 74.897049 | 1.411544 | 0.256255 | 9,988,382.48 | -0.53% |
| Improved parametric GLM | 73.466040 | 74.814510 | 1.348469 | 0.257554 | 9,957,587.03 | -0.84% |
| Initial LightGBM | 64.577274 | 74.751332 | 10.174058 | 0.273634 | 8,324,414.59 | -17.10% |
| Tuned LightGBM | 66.412966 | 74.291639 | 7.878672 | 0.274197 | 8,605,090.30 | -14.31% |

## Interpretation and limitations

Tuned LightGBM has the lowest observed test deviance. Its Gini is only slightly above the initial LightGBM’s; no uncertainty or significance analysis was performed. D’s quadratic GLM improves slightly over the benchmark GLM in observed deviance, while both have very similar risk ranking. These differences should not be interpreted as statistically established superiority.

The initial LightGBM train–test gap is 10.174058, compared with 7.878672 after tuning. E’s unchanged full training-only three-fold CV selected learning_rate=0.03 and n_estimators=300, with mean validation deviance 73.831160 versus 74.509440 for the initial parameters. Each fold fits a fresh complete preprocessing/model pipeline; the selected model is refitted on all current training policies. These CV numbers differ from E’s published historical report because the training membership changed. Historical scores were not reused or represented as the present run.

The GLMs have much smaller aggregate test claim bias (approximately −0.53% benchmark and −0.84% quadratic) than the LightGBMs (−17.10% initial, −14.31% tuned). Aggregate totals are only one aspect of calibration and do not prove calibration within risk groups. The GLMs retain explicit coefficients/transforms; boosted trees offer nonlinearities and interactions with less direct interpretation. Large train–test gaps are diagnostics, not proof of overfitting by themselves.

**D selection limitation:** D chose the quadratic specification after inspecting test-tail behaviour on the earlier split of this same dataset. Repartitioning and refitting does not make this an independent model-selection test. D’s score remains exploratory and would need genuinely independent validation for an unbiased selection comparison. C preserves D’s handed-over method instead of choosing another specification using the new test scores. The actual preprocessing and E’s CV remain training-only.

## Lorenz/Gini conventions

Each model is ranked by predicted pure premium. The horizontal axis is fraction of policies; the vertical axis accumulates observed PurePremium × Exposure, i.e. ClaimAmountCut. Curves start at (0,0), group tied predictions and end at (1,1). Gini is unnormalized 1−2×AUC. Random is the diagonal. The teacher’s Oracle ranks observed PurePremium; with unequal exposures it is a reference, not necessarily the ordering that maximizes policy-count Gini. The final figure includes all four verified models plus Oracle and Random.

`verification_status` identifies locally refitted records. `gini_status` distinguishes newly recomputed values from previously prediction-validated values retained by an offline summary. The old member-published JSONs remain historical records; the authoritative current outputs are in `results/evaluated/`.

## Reproduce and verify

```bash
python -m analyses.ps3_script all --save-data data/shared_final.parquet
python -m analyses.ps3_script compare --data data/shared_final.parquet
python -m pytest
pre-commit run --all-files
```

Choose a new cache filename if it exists. The instructor loader was used successfully in this session after a stalled connection and retry. The final normalized-split run reused exactly that prepared data, reconstructed membership with the new official function and saved `data/final_shared_normalized_20261120.parquet`. Its separate cached compare run independently verified the final table and figure. Prior results are archived under gitignored `data/run_backups/`; per-policy predictions, caches, raw data and logs are local only.

GLM converged in 3 iterations (limit 100), with convergence warnings treated as errors. Both GLM sample scores agree between sklearn and glum. Independent test rescoring checked all four models against both deviance implementations, the exact policy-ID order/membership, positive weights/predictions, target equality, fingerprints and aggregate claim totals. An offline summary retained all four verified Ginis.

Full suite after the official split update: **57 passed, no skipped tests**, including 27 C tests, 12 Winsorizer tests and 17 split tests. Pre-commit checks pass. Matplotlib emits dependency deprecation warnings, not model convergence warnings. B’s original main implementation/tests replace the old placeholders through their existing Git history. The integration additionally fixes loss of verified Ginis during offline or partial summaries.

Split fingerprint: `48634e41e66f74a0a62ae7e54de1da7299d8abf8e78ce5f9ca941ce055e0e881`.

Data fingerprint: `fefd1e42c487594b88df12032cef35d3599b189b4ff0b06ae3ddab48006b432e`.

Each evaluated JSON records source-file hashes, versions, timestamp and source commit `350f055db87ed8c399307c81f1bc4dbd3c49ebb0`. Later report-only commits do not change the model specification.

## Ownership and PR dependencies

A: split and initial LightGBM (#10/#11). B: Winsorizer on main. E: unchanged tuning/CV (#12, 8e9340f). D: unchanged quadratic GLM (#14, 54558d1). All member implementations match their owner branches; the splitter matches merged main’s normalized version. C owns the shared orchestration, evaluation adapters, validation, tests and combined outputs.

`integration/part-c-dependencies` is the review base containing the member histories and the merged Ex1 fix. PR #13 targets it to keep C’s diff focused while member PRs remain open. It is not a merge into main. The repository owner must coordinate the pending member PRs and later retarget C to finalized main; rerun if the data, split or model sources change. The D handover is available and integrated, with its selection limitation explicit.

# LightGBM tuning notes

## Selection

This run uses the normalized `IDpol` split merged in PR #10, with 542,286
training and 135,727 test policies. Both JSON files record the same policy
split fingerprint.

The fixed baseline uses 300 estimators and a learning rate of 0.05. I compared six
combinations of learning rate (0.03, 0.05, 0.08) and estimator count (300, 500).
Selection used the lowest mean Exposure-weighted Tweedie deviance across three
shuffled folds of the fixed training sample (random state 42). The target, Exposure
weights, Tweedie power, ID-based train/test split, and remaining LightGBM parameters
were held constant. Each fold fits its own categorical encoder using only that fold's
training rows.

The selected setting is learning rate 0.03 with 300 estimators. It had mean CV
deviance 73.831, compared with 74.509 for the baseline setting (a reduction of
0.678, about 0.9%). Its CV standard deviation was 2.474 across folds. The next best
candidate, 0.03 with 500 estimators, had mean deviance 74.424.

## Held-out comparison and overfitting check

| Model | Train deviance | Test deviance |
| --- | ---: | ---: |
| Fixed baseline (0.05, 300) | 64.577 | 74.751 |
| Tuned (0.03, 300) | 66.413 | 74.292 |

The tuned model improves the fixed test deviance by 0.460 (about 0.6%). Its train to
test gap is 7.879, compared with 10.174 for the baseline. The lower training score
alone should not be used to select a model: compare it with the training-only CV
scores and the held-out test score. Here CV and test both favour the tuned setting,
while the gap remains a sign of generalisation error. Only the selected setting was
evaluated on the test set after tuning.

The full fold-level table and run metadata are in `lgbm_tuning_cv.json`; model metrics
and the baseline comparison are in `lgbm_tuning_metrics.json`.

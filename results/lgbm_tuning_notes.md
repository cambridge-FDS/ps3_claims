# LightGBM tuning notes

## Selection

The fixed baseline uses 300 estimators and a learning rate of 0.05. I compared six
combinations of learning rate (0.03, 0.05, 0.08) and estimator count (300, 500).
Selection used the lowest mean Exposure-weighted Tweedie deviance across three
shuffled folds of the fixed training sample (random state 42). The target, Exposure
weights, Tweedie power, ID-based train/test split, and remaining LightGBM parameters
were held constant. Each fold fits its own categorical encoder using only that fold's
training rows.

The selected setting is learning rate 0.03 with 300 estimators. It had mean CV
deviance 73.724, compared with 74.264 for the baseline setting (a reduction of
0.540, about 0.7%). Its CV standard deviation was 1.430 across folds. The next best
candidate, 0.03 with 500 estimators, had mean deviance 74.150.

## Held-out comparison and overfitting check

| Model | Train deviance | Test deviance |
| --- | ---: | ---: |
| Fixed baseline (0.05, 300) | 64.546 | 74.669 |
| Tuned (0.03, 300) | 66.425 | 74.138 |

The tuned model improves the fixed test deviance by 0.531 (about 0.7%). Its train to
test gap is 7.713, compared with 10.124 for the baseline. The lower training score
alone should not be used to select a model: compare it with the training-only CV
scores and the held-out test score. Here CV and test both favour the tuned setting,
while the gap remains a sign of generalisation error. Only the selected setting was
evaluated on the test set after tuning.

The full fold-level table and run metadata are in `lgbm_tuning_cv.json`; model metrics
and the baseline comparison are in `lgbm_tuning_metrics.json`.

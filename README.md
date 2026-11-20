# PS3 — Feature engineering and claims modelling

This submission covers the fixed train/test split, the Winsorizer transformer and the motor claims model comparison. The [official problem set](https://languid-citipati-d0d.notion.site/PS-3-Feature-engineering-modelling-2ac32dea82c1814dad25e6d3c2af3b4e) gives the three exercise prompts.

## Install and run

```bash
git clone https://github.com/starries-creator/ps3_claims.git
cd ps3_claims
conda env create -f environment.yml
conda activate ps3
python -m pip install --no-build-isolation -e .
python -m pytest
python -m analyses.ps3_script all --save-data data/shared_final.parquet --results-dir data/reproduced
```

The last command loads the instructor's OpenML data, fits the four models and writes the comparison table and Lorenz figure to `data/reproduced/`. Once the cache exists, rerun using it as input.

```bash
python -m analyses.ps3_script all --data data/shared_final.parquet --results-dir data/reproduced
```

## Exercise 1 — Fixed train/test split

`create_sample_split` labels rows as `train` or `test` by hashing one or more normalized ID columns. A policy keeps its assignment when rows are shuffled or new policies are added. For this dataset, splitting on `IDpol` with a target training fraction of 80% gives **542,286 training policies** and **135,727 test policies**. [Implementation](ps3/data/_sample_split.py) · [Split tests](tests/data/test_sample_split.py) · [Exercise 1 PR](https://github.com/starries-creator/ps3_claims/pull/10)

## Exercise 2 — Winsorizer

`Winsorizer.fit` learns the lower and upper quantile cutoffs from the training data. `transform` clips values to those same cutoffs when applied to new data. In the unit test, fitting the 25th and 75th percentiles on `[0, 10, 20, 30, 100]` gives cutoffs of 10 and 30. The transformer then clips `[-100, 50, 1000]` to `[10, 30, 30]`. [Implementation](ps3/preprocessing/_winsorizer.py) · [Transformer tests](tests/preprocessing/test_winsorizer.py) · [Exercise 2 PR](https://github.com/starries-creator/ps3_claims/pull/8)

## Exercise 3 — Improve the model

All four models use the instructor's data transformation and the same fixed split from Exercise 1. The table reports mean Exposure-weighted Tweedie deviance at power 1.5. Lower is better. [Combined results](results/metrics.csv) · [Data transformation](ps3/data/_load_transform.py) · [Evaluation code](analyses/part_c.py) · [Integrated runner](analyses/ps3_script.py)

| Model | Train deviance | Test deviance |
| --- | ---: | ---: |
| [Tweedie GLM benchmark](results/evaluated/tweedie_glm_metrics.json) | 73.4855 | 74.8970 |
| [Quadratic Tweedie GLM](results/evaluated/parametric_glm_metrics.json) | 73.4660 | 74.8145 |
| [Initial LightGBM](results/evaluated/lightgbm_initial_metrics.json) | 64.5773 | 74.7513 |
| [Tuned LightGBM](results/evaluated/lightgbm_tuned_metrics.json) | 66.4130 | **74.2916** |

### Why divide by Exposure?

`PurePremium = ClaimAmountCut / Exposure` expresses clipped claims per unit of coverage, so policies with different coverage durations are comparable. Fitting and evaluation weight each policy by `Exposure`. To get the predicted portfolio claim total, multiply each policy's predicted pure premium by its exposure and sum the amounts. [Shared evaluation](analyses/part_c.py) · [Comparison report](results/final_comparison.md)

### What changed in the GLM?

The benchmark uses a linear term for `BonusMalus` and treats `VehPower` as categorical. The submitted GLM uses linear and squared terms for both variables. Train and test deviance fall slightly, with test deviance moving from 74.8970 to 74.8145. [Benchmark coefficients](results/tweedie_glm_coefficients.csv) · [Quadratic implementation](analyses/parametric_glm.py) · [GLM notes and variant results](results/parametric_glm_notes.md)

### How did we check for overfitting?

The GLM analysis compares train and test deviance, checks cross-validation results and inspects fitted curves where training policies are sparse. The double spline scored lower than the quadratic on the fixed test set, but after refitting without two high-`BonusMalus` training policies, the spline's predictions for those policies rose to about 145 and 64 times their original values. The quadratic predictions changed by less than 2%, so the team retained it. Earlier test-tail diagnostics informed that choice. An independent sample would be needed to validate the choice without that influence. [Tail stress test](results/parametric_glm_notes.md#two-policy-tail-stress-test) · [Sensitivity plot](results/parametric_glm_bonus_malus.png)

### What did LightGBM tuning change?

Initial LightGBM used a learning rate of 0.05 and 300 estimators. Tuning compared six configurations using three folds within the training sample. The selected learning rate of 0.03 with 300 estimators lowered mean CV deviance from 74.509 to 73.831. It also lowered fixed-test deviance from 74.7513 to 74.2916 and narrowed the train–test gap from 10.174 to 7.879. [Baseline code](analyses/lgbm_baseline.py) · [Tuning code](analyses/lgbm_tuning.py) · [Tuning notes](results/lgbm_tuning_notes.md) · [Fold-level scores](results/lgbm_tuning_cv.json)

### What does the four-model comparison show?

Tuned LightGBM has the lowest test deviance and the highest observed Gini, with a small Gini improvement over initial LightGBM. Both GLMs predict portfolio totals within 1% of observed clipped test claims, while the initial and tuned LightGBM totals are about 17% and 14% below observed claims. Agreement on the total alone leaves calibration across risk groups open. [Full comparison and limitations](results/final_comparison.md) · [Test Lorenz curves](results/lorenz_comparison.png)

## Repository map

| Path | Contents |
| --- | --- |
| [`ps3/`](ps3/) | Data preparation, fixed split and preprocessing |
| [`analyses/`](analyses/) | Model implementations and run commands |
| [`tests/`](tests/) | Unit and integration tests |
| [`results/`](results/) | Submitted metrics, figures and analysis notes |

The original division of work is recorded in the [team assignments](TEAM_ASSIGNMENTS.md).

# PS3 — Feature engineering & modelling: A–E assignments

- [Official problem set](https://languid-citipati-d0d.notion.site/PS-3-Feature-engineering-modelling-2ac32dea82c1814dad25e6d3c2af3b4e) · [Team repository](https://github.com/starries-creator/ps3_claims) · [Setup and running instructions](README.md)
- [Instructor-specified glum tutorial](https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html)
- Submission deadline: **Friday, 20 November 2026 at 12 noon**. One team member should email the public GitHub repository link to Vahan (vg400@cam.ac.uk).
- **A–E are roles awaiting assignment, not predefined names.** All three exercises share this repository. The plan below describes the team's working arrangements; the official problem set governs the coursework requirements.

## Shared conventions

First agree on the Ex1 split interface, primary key(s) and training fraction. All models must use the same `sample` column. The agreed interface is `create_sample_column(df, id_columns, training_frac=0.8)`, supporting one or multiple columns and returning a dataframe with `sample` (`train` / `test`). A must also retain the existing `create_sample_split` entry point for the scaffold and make the parameter mapping clear.

For Ex3, retain the scaffold's pure premium target, Exposure weights and Tweedie power=1.5. Report exposure-weighted train/test deviance normalised by total weight consistently. Agree any changes with the whole team first and rerun all models being compared. Tune using cross-validation within the training portion only; fit preprocessing within each training fold, and do not use the test set to select hyperparameters.

Each model should save a small metrics file at `results/<model>_metrics.json`, recording the model name, split convention, train/test row counts and deviance, target/weights/power, parameters and selection method. C will consolidate these into `results/metrics.csv`. Each contributor should save the necessary figures and explanations; do not push large per-policy predictions, raw data or model files to Git.

## Ex1 — Create fixed train/test split

[Exercise](https://languid-citipati-d0d.notion.site/Ex-1-Create-fixed-train-test-split-2ac32dea82c181e082a6c7b76a79f71a)

**Goal:** Create a stable split based on one or more primary keys, so that the same observation always belongs to the same set.

**Dependencies across exercises:** Work can start immediately. Once complete, Ex1 supplies the fixed split for all Ex3 models. Ex2 can proceed independently in parallel. Ex3 data preparation, model code and evaluation interfaces can be prepared in advance; formal comparisons must wait until the split is merged.

| Stage / order | Task and exercise item | Owner | Start condition | Deliverable |
| --- | --- | --- | --- | --- |
| 1, start immediately | All of Ex1: implement single/multiple primary keys, training fraction and stable bucketing; reconcile the function names in the instructions and scaffold | A | README environment is available | `ps3/data/_sample_split.py` and public exports; retain a compatible entry point |
| 1, same work package | Verify that existing observations keep their assignments across repeated calls, row reordering and new observations; test multiple keys, fractions and boundary inputs; explain the bucketing choice and interpretation of the fraction | A | Develop alongside the implementation | `tests/data/test_sample_split.py`, interface documentation and limitations |
| 2, connect to Ex3 | Ex3 item 2: use the agreed split in the shared data entry point and record the primary key(s) and fraction | A | Ex1 is merged and C's data entry point is agreed | A reusable split call for all models; documentation of the data split |

## Ex2 — Write your own Transformer

[Exercise](https://languid-citipati-d0d.notion.site/Ex-2-Write-your-own-Transformer-2ac32dea82c181679356c02e5fc4436e)

**Goal:** Implement and test a sklearn-compatible `Winsorizer` for use in subsequent model pipelines.

**Dependencies across exercises:** Run in parallel with Ex1. Unit tests do not depend on the real insurance data. B can agree input shapes and return formats with D/E in advance; the merged component will then be available for Ex3 integration.

| Stage / order | Task and exercise item | Owner | Start condition | Deliverable |
| --- | --- | --- | --- | --- |
| 1, start immediately | Ex2 implementation: `__init__(lower_quantile, upper_quantile)`, `fit` storing `lower_quantile_` / `upper_quantile_`, and clipping in `transform` | B | README environment is available | Complete `ps3/preprocessing/_winsorizer.py` |
| 1, same work package | Ex2 tests: complete the instructor's pytest placeholders; check different quantiles, fit/transform behaviour, the sklearn interface and pipeline compatibility | B | Develop alongside the implementation | `tests/preprocessing/test_winsorizer.py`, test commands and necessary explanations |
| 2, agree interfaces in advance | Own the custom Winsorizer's Ex3 integration: agree with D/E which numeric branch will use it, training-fold fitting rules and the feature output interface | B | Interfaces can be agreed first; full integration waits for Ex2 and the relevant pipeline to be ready | Reusable preprocessing components and usage instructions, called by D/E from their model entry points |

## Ex3 — Improve the model

[Exercise](https://languid-citipati-d0d.notion.site/Ex-3-Improve-the-model-2ac32dea82c18123a6e1e926a62721a9)

**Goal:** Compare the Tweedie benchmark, an improved parametric model and tuned LightGBM using a fixed split and consistent evaluation conventions.

**Dependencies across exercises:** Formal runs depend on Ex1. Pipelines using the custom Winsorizer also depend on Ex2. C can confirm the data entry point immediately. A/D/E can write independent model modules in advance, then connect them to the shared data and evaluation functions.

| Stage / order | Task and exercise item | Owner | Start condition | Deliverable |
| --- | --- | --- | --- | --- |
| 1, start immediately | Item 1: obtain data with `load_transform`, confirm the required columns and data access method, and agree shared data/evaluation interfaces | C | Environment is available and OpenML is accessible | Data entry point in the main script and shared interface documentation; retain the instructor's data transformation conventions |
| 1, prepare in advance; 2, formal run | Item 3: run the initial Tweedie benchmark, explain why the outcome is divided by Exposure, and provide shared evaluation and result output | C | Read the tutorial/prepare code first; formal results wait for A's split | Shared entry point in `analyses/ps3_script.py`, benchmark metrics and explanation |
| 1, prepare in advance; 2, formal run | Item 4: add flexibility for **BonusMalus and VehPower** using polynomials or splines; compare deviance and explain how to detect overfitting | D | Prepare transformations first; formal comparison waits for shared data/evaluation interfaces and Ex1; Winsorizer integration waits for B | Independent parametric model module, metrics, necessary figures and explanations |
| 2, parallel with the parametric model | Initial LGBM for item 5: establish a suitable initial model and prediction entry point, then hand it to E for tuning | A | Ex1 and shared interfaces are agreed; model interfaces can be prepared in advance | `analyses/lgbm_baseline.py`, initial model metrics, parameters and running instructions |
| 1, prepare the search; 3, tuning and explanation | Tuning for item 5: select parameters within the training set, control overfitting, compare with the initial LGBM/GLM and explain the selection | E | Prepare the search/evaluation first; formal runs wait for A's initial LGBM and shared interfaces; coordinate preprocessing integration with B | `analyses/lgbm_tuning.py`, CV results, final parameters, tuned model metrics and explanation |
| 4, consolidate analysis outputs | Consolidate the benchmark, parametric model, initial and tuned LGBM; connect the runnable entry point and organise comparative conclusions and the scaffold's Lorenz/Gini figures | C | Model modules and metrics are ready; the summary template can be prepared in advance | `results/metrics.csv`, small figures, a unified results explanation and final running instructions in README |

**Differences between the instructions and scaffold:** Official item 4 specifies `BonusMalus` and `VehPower`, whereas the script comments say `BonusMalus` and `Density`; the initial code also treats `VehPower` as categorical. D must explicitly adjust column assignments to avoid including the same feature twice. Additional Density experiments may be documented, but cannot replace the features specified in the instructions.

**File ownership:** C primarily edits `analyses/ps3_script.py` and the final results explanation. D owns the independent parametric model module; A/E own the initial LGBM/tuning modules respectively. B primarily edits `ps3/preprocessing/` and its tests. Document shared function parameters, return values and sample weight passing in the PR before calling them from the main script. Coordinate before editing shared files owned by someone else.

**Optional work, listed separately:** Item 6, frequency + severity LGBM and its comparison, is optional and is not included in the required workload above. Claim it after the required work is complete, assign one primary owner at that point, and use the same split and evaluation conventions.

All five roles include implementation, necessary tests of their own results and explanations. A moves from the stable split to the initial LGBM; B moves from the Transformer to preprocessing integration. C's workload includes shared evaluation and the final analysis summary, while D/E each own substantial modelling experiments. Once roles are claimed, include PR review/merge time in the relevant person's workload and check that balance again; do not assume that person is A.

Start with the README, set up the environment and install the hook in your own clone. Work on individual branches and submit PRs describing completed tasks and necessary validation. The repository owner normally coordinates PR review/merging; no separate routine reviewer or merger is assigned. Update from `main` after prerequisites are merged before connecting downstream work. Add the A–E name/account mapping once roles have been claimed.

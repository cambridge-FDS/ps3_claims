# PS3: A–E team assignments

[Problem set](https://languid-citipati-d0d.notion.site/PS-3-Feature-engineering-modelling-2ac32dea82c1814dad25e6d3c2af3b4e) · [Team repository](https://github.com/starries-creator/ps3_claims) · [Setup instructions](README.md) · [glum tutorial](https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html)

**Deadline: Friday, 20 November 2026 at 12 noon.** One team member should email the public GitHub repository link to Vahan (vg400@cam.ac.uk).

A–E have not been assigned to people yet. Use this overview to choose a role; the details below explain what each role includes. All three exercises use the same repository, and the official problem set remains the source of coursework requirements.

| Role | Main responsibilities |
| --- | --- |
| A | Implement the fixed data split, then build the initial LightGBM model for E to tune |
| B | Implement and test the Winsorizer, then help D/E connect it to their model pipelines |
| C | Set up the shared data and evaluation code, run the Tweedie benchmark, and bring the final results together |
| D | Improve the parametric model using polynomials or splines, and explain the results |
| E | Tune LightGBM using the training data, compare it with the other models, and explain the parameter choice |

Each person is responsible for the code, necessary tests and explanation for their own work. The person coordinating PR review and merging will also need time for that; once roles are claimed, check their overall workload. Do not assume that person is A.

## Ex1: A builds the fixed train/test split

[Exercise instructions](https://languid-citipati-d0d.notion.site/Ex-1-Create-fixed-train-test-split-2ac32dea82c181e082a6c7b76a79f71a)

A can start immediately after setting up the README environment. The split should use one or more primary keys so that the same observation always goes into the same set.

A's work includes:

- Implement the split and training fraction in `ps3/data/_sample_split.py`, and expose the function for the other models to import.
- Test repeated calls, reordered rows and newly added observations: existing observations should keep their assignments. Also cover multiple keys, fractions and boundary inputs. Put the tests in `tests/data/test_sample_split.py` and explain the bucketing choice, what the fraction means and any limitations.
- Complete Ex3 item 2 by connecting this function to C's shared data entry point once that interface is agreed and the split is merged. Record the keys and fraction so that everyone uses the same split.

The instructions call the function `create_sample_column`; the scaffold currently imports `create_sample_split`. A should reconcile these names and retain the existing entry point for compatibility. The shared function signature is listed under “Agreements for implementation” below.

## Ex2: B builds and tests the Winsorizer

[Exercise instructions](https://languid-citipati-d0d.notion.site/Ex-2-Write-your-own-Transformer-2ac32dea82c181679356c02e5fc4436e)

B can work in parallel with A. These unit tests do not need the real insurance data.

B's work includes:

- Complete `ps3/preprocessing/_winsorizer.py`: accept `lower_quantile` and `upper_quantile` in `__init__`, store the fitted values in `lower_quantile_` and `upper_quantile_` during `fit`, and clip the data in `transform`.
- Complete the instructor's tests in `tests/preprocessing/test_winsorizer.py`. Cover different quantiles, fit/transform behaviour, sklearn compatibility and use in a pipeline; include the test command and any necessary explanation.
- Own the custom Winsorizer's integration into Ex3. Agree input shapes, return formats, the numeric branch that will use it and training-fold fitting rules with D/E. Provide reusable preprocessing components and usage instructions; D/E will call them from their own model code.

B can agree these interfaces with D/E in advance. The actual integration needs both the Winsorizer and the relevant model pipeline to be ready.

## Ex3: build and compare the models

[Exercise instructions](https://languid-citipati-d0d.notion.site/Ex-3-Improve-the-model-2ac32dea82c18123a6e1e926a62721a9)

C can start by checking data access and agreeing the shared data/evaluation interfaces. D can prepare transformations, A can prepare the LightGBM interface, and E can plan the tuning search. Final model runs and comparisons need A's fixed split; pipelines using the Winsorizer also need B's component.

| Who | Work to complete | What to hand over |
| --- | --- | --- |
| C | Items 1 and 3: load the data with `load_transform`, check the required columns, retain the instructor's data transformations, and provide shared evaluation/output code. Run the initial Tweedie benchmark and explain why the outcome is divided by Exposure. | The shared entry point in `analyses/ps3_script.py`, interface/data-access instructions, benchmark metrics and explanation |
| D | Item 4: use polynomials or splines to add flexibility for **BonusMalus and VehPower**. Compare train/test deviance and explain how to detect overfitting. | A separate parametric model module, metrics, necessary figures and explanation |
| A | The initial model in item 5: build a suitable initial LGBM model and prediction entry point for E to tune. This follows A's work on the fixed split. | `analyses/lgbm_baseline.py`, initial metrics, parameters and running instructions |
| E | Tuning in item 5: select parameters using only the training data, control overfitting, and compare with the initial LGBM and GLM. Explain the selection. | `analyses/lgbm_tuning.py`, CV results, final parameters, tuned model metrics and explanation |
| C | Bring the benchmark, parametric model, initial LGBM and tuned LGBM together. Connect the runnable entry point, explain the comparison and produce the scaffold's Lorenz/Gini figures. | `results/metrics.csv`, small figures, a unified results explanation and final running instructions in README |

The main handovers are:

- A connects the split with C's data entry point. Once the split and shared interfaces are ready, D's parametric model and A's initial LGBM can run in parallel.
- E's final tuning runs need A's initial LGBM and the shared interfaces. E coordinates preprocessing with B and can prepare the search/evaluation code beforehand.
- C can prepare the summary template early, but the final comparison needs everyone's model modules and metrics.

**A detail for D:** item 4 asks for `BonusMalus` and `VehPower`, but the script comments say `BonusMalus` and `Density`, and the initial code treats `VehPower` as categorical. Adjust the column assignments explicitly so a feature is not included twice. Extra Density experiments can be documented, but do not replace the required features.

**Optional:** item 6, frequency + severity LGBM and its comparison, is outside the required workload above. Claim it after the required work is complete, give it one primary owner, and use the same split and evaluation conventions.

## Agreements for implementation

These details keep everyone's results comparable; agree them before connecting the model code.

- Use `create_sample_column(df, id_columns, training_frac=0.8)` for single or multiple key columns. It returns a dataframe with `sample` values of `train` / `test`. A also retains `create_sample_split` and documents the parameter mapping. All models use the same keys, fraction and `sample` column.
- Retain the scaffold's pure premium target, Exposure weights and Tweedie power=1.5. Report exposure-weighted train/test deviance normalised by total weight. If these choices change, agree it with the team and rerun every model being compared.
- Tune by cross-validation within the training data. Fit preprocessing within each training fold; do not select hyperparameters using the test set.
- Save a small `results/<model>_metrics.json` for each model, recording its name, split convention, train/test row counts and deviance, target/weights/power, parameters and selection method. C combines these into `results/metrics.csv`. Save necessary figures and explanations, but keep large per-policy predictions, raw data and model files out of Git.
- Keep model code separate: C mainly edits `analyses/ps3_script.py` and the final results explanation; D owns the parametric module; A/E own the initial LGBM/tuning modules; B owns `ps3/preprocessing/` and its tests. Describe shared function parameters, return values and sample weight passing in the PR before the main script calls them. Coordinate before editing someone else's shared files.

Start from the README and install the hook in your own clone. Work on your own branch and submit a PR explaining what you completed and checked. The repository owner normally coordinates review and merging; no separate routine reviewer or merger is assigned. Update from `main` after prerequisite changes are merged. Add names and GitHub accounts to A–E once roles are claimed.

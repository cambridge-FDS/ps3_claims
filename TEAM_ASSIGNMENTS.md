# PS3 team assignments

[Problem set](https://languid-citipati-d0d.notion.site/PS-3-Feature-engineering-modelling-2ac32dea82c1814dad25e6d3c2af3b4e) · [Team repository](https://github.com/starries-creator/ps3_claims) · [Setup instructions](README.md) · [glum tutorial](https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html)

**Deadline — Friday, 20 November 2026 at 12 noon.** One team member emails the public GitHub repository link to Vahan (vg400@cam.ac.uk).

| Role to claim | Main work |
| --- | --- |
| A | Fixed train/test split and initial LightGBM model |
| B | Winsorizer implementation, tests and pipeline integration |
| C | Shared data and evaluation code, Tweedie benchmark and final comparison |
| D | Parametric model with polynomials or splines |
| E | LightGBM tuning and comparison |

Set up the environment and install the hook in your own clone using the README. Work on your own branch and open a PR with what you did and checked. The repository owner normally coordinates review and merging. Pull the latest `main` after changes you depend on are merged.

## Ex1 · Fixed train/test split

[Exercise instructions](https://languid-citipati-d0d.notion.site/Ex-1-Create-fixed-train-test-split-2ac32dea82c181e082a6c7b76a79f71a)

1. **After setup**, A writes `create_sample_column(df, id_columns, training_frac=0.8)` in `ps3/data/_sample_split.py`. Use one or more key columns, return `sample` with `train` or `test`, and keep `create_sample_split` available for the existing script. In `tests/data/test_sample_split.py`, check that repeated calls, reordered rows and new records leave existing assignments stable. Cover multiple keys, fractions and edge cases. Document the bucketing choice and what the fraction means.
2. **Once the split is merged and C's data interface is agreed**, A plugs it into the shared data entry point for Ex3 item 2 and records the keys and training fraction. C, D and E can then use the same split for their models, as can A for the initial LightGBM.

## Ex2 · Winsorizer

[Exercise instructions](https://languid-citipati-d0d.notion.site/Ex-2-Write-your-own-Transformer-2ac32dea82c181679356c02e5fc4436e)

1. **Alongside Ex1**, B completes `ps3/preprocessing/_winsorizer.py` and `tests/preprocessing/test_winsorizer.py`. Accept `lower_quantile` and `upper_quantile` in `__init__`, save fitted quantiles as `lower_quantile_` and `upper_quantile_`, and clip in `transform`. Test different quantiles, fit/transform behaviour and sklearn pipeline compatibility. Include the test command.
2. **Before Ex3 integration**, B agrees with D/E on input shape, output format, which numeric branch uses the Winsorizer and how to fit it within each training fold. Once the component is merged and the relevant pipeline is ready, B provides the reusable preprocessing and usage instructions for D/E to call.

## Ex3 · Train and compare the models

[Exercise instructions](https://languid-citipati-d0d.notion.site/Ex-3-Improve-the-model-2ac32dea82c18123a6e1e926a62721a9)

1. **After setup, start in parallel.**
   - **A** works on the fixed split and can prepare the initial LightGBM interface.
   - **B** builds the Winsorizer and agrees how D/E will use it.
   - **C** checks OpenML access and required columns with `load_transform` (item 1), keeps the supplied transformations, and prepares shared data and evaluation code in `analyses/ps3_script.py`.
   - **D** prepares the feature transformations. **E** prepares the search and evaluation code.
2. **With the fixed split and shared interfaces in place, run the first models in parallel.**
   - **C** runs the Tweedie benchmark (item 3), saves its metrics and explains why the target is divided by Exposure.
   - **D** uses polynomials or splines for `BonusMalus` and `VehPower` (item 4). Use `VehPower` as required by the problem set even though the script comments mention `Density`, and avoid also feeding `VehPower` in as a categorical feature. Use B's Winsorizer once ready wherever the pipeline calls for it. Hand over the separate model module, train/test deviance, relevant figures and an explanation of how to spot overfitting.
   - **A** runs the initial LightGBM (item 5). Hand `analyses/lgbm_baseline.py`, the prediction entry point, parameters, metrics and run instructions to E.
3. **Once A hands over the initial LightGBM, E tunes it** (item 5). Use cross-validation within the training data and fit preprocessing inside each training fold. Coordinate Winsorizer use with B as needed. Compare with the initial LightGBM and GLM, and hand C `analyses/lgbm_tuning.py`, CV results, chosen parameters, final metrics and the reasoning behind the selection and checks for overfitting.
4. **Once the model code and metrics are ready, C brings them together.** Connect the runnable entry point, collect all four sets of metrics in `results/metrics.csv`, explain the comparisons and include the Lorenz/Gini figures. Update the README with the final run instructions. C can prepare the summary template while others work.

Use the same split, pure premium target, Exposure weights and Tweedie power (1.5) for all runs. Report train/test deviance weighted by Exposure and divided by total weight. If a convention changes, agree it together and rerun the comparison. Each model owner saves `results/<model>_metrics.json` with the model name, split, train/test counts and deviance, target, weights, power, parameters and selection method. Commit small figures and explanations alongside the metrics. Keep raw data, model files and large per-policy predictions local.

**Optional after the required work is complete.** Claim item 6 separately with one owner. Use the same split and evaluation when comparing frequency + severity LightGBM.

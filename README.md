# PS3 — Claims modelling

This repository is shared by the team for all three PS3 exercises. The upstream repository is
[cambridge-FDS/ps3_claims](https://github.com/cambridge-FDS/ps3_claims).

- [Official problem set](https://languid-citipati-d0d.notion.site/PS-3-Feature-engineering-modelling-2ac32dea82c1814dad25e6d3c2af3b4e)
- [Team assignments, prerequisites and shared interfaces](TEAM_ASSIGNMENTS.md)
- [Instructor-specified glum tutorial](https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html)

Deadline: 20 November 2026 at 12 noon. One team member should email the public repository
link to Vahan (vg400@cam.ac.uk).

## Environment setup

Use the instructor's `environment.yml`, with Conda as the main environment manager.
There is no need to create a separate Pixi or uv environment. Install
[Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html),
or use an existing Mamba installation, then run:

```bash
git clone https://github.com/starries-creator/ps3_claims.git
cd ps3_claims
conda env create -f environment.yml
conda activate ps3
python -m pip install --no-build-isolation -e .
pre-commit install
```

With Mamba, replace the environment creation command with `mamba env create -f environment.yml`.
If an environment named `ps3` already exists on your machine, check what it is used for first.
You can create a differently named environment with
`conda env create -f environment.yml -n ps3-coursework`, then activate that name.
Do not delete or overwrite environments used for other courses.

`environment.yml` includes the dependencies for data handling, modelling, plotting, ipykernel
and pre-commit. It retains the upstream version ranges rather than locking every dependency;
Conda resolves compatible dependencies for each platform. In VS Code, select this environment's
Python/kernel for the `# %%` interactive cells in `.py` files.

## Basic checks after installation

```bash
python -c "from ps3.data import load_transform, create_sample_split; from ps3.preprocessing import Winsorizer; print('imports OK')"
python -m pytest tests/test_core.py
pre-commit run --all-files
```

Each team member must run `pre-commit install` in **their own clone**; the local hook is not
installed automatically when cloning. The current hooks use a pinned revision and check YAML,
TOML, Python syntax, conflict markers and newly added large files. They do not remove notebook
outputs. Passing these checks does not mean the coursework is complete.

## Data and coursework entry points

`ps3.data.load_transform()` reads two public OpenML data files directly; no account is required:

- [freMTPL2freq](https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff)
- [freMTPL2sev](https://www.openml.org/data/get_csv/20649149/freMTPL2sev.arff)

```python
from ps3.data import load_transform
df = load_transform()
```

The first read requires internet access. The existing loader reads from the remote source on
every call and has no local cache. `data/` is only for personal downloads/caching; the current
loader does not automatically read from that directory. Do not commit raw data, virtual
environments, model files or large per-policy predictions. Small metrics files and necessary
figures can go in `results/`.

## Exercise 3 modelling

Run commands from the repository root in the project environment:

```bash
python -m analyses.ps3_script check-data
python -m analyses.ps3_script benchmark
python -m analyses.ps3_script lightgbm
python -m analyses.ps3_script compare
```

`check-data` validates the data and shows the split counts. The model commands can
run independently. Use `python -m analyses.ps3_script all` to run both baselines and
compare them after loading the data once. Use `python -m analyses.ps3_script metrics`
to summarize saved metrics without accessing OpenML.

`load_model_data()` calls the instructor's `load_transform()` via the shared
preparation function and uses A's deterministic split with `IDpol` and a training
fraction of 0.8. Pure premium divides the clipped claim amount by Exposure so
policies with different periods of coverage share a per-unit-of-time scale.
Fitting and deviance use Exposure weights. Predicted total claims sum the
predicted pure premium times Exposure for each test policy.

The benchmark retains the categorical preprocessing, features and GLM settings
from the instructor's PS3 script. The initial LightGBM calls A's implementation
with the same prepared frame. Both pre-process and fit using only the training
sample. The comparison uses Exposure-weighted mean Tweedie deviance at power 1.5.

| Output | Contents |
| --- | --- |
| `results/tweedie_glm_metrics.json` | Benchmark settings, split and scores after a successful fit. The benchmark also prints coefficients. |
| `results/lightgbm_initial_metrics.json` | A's model rescored on the common split after a local fit. |
| `results/lgbm_baseline_metrics.json` | A's initial LightGBM run from PR #11. |
| `results/lgbm_tuning_metrics.json` | E's tuned LightGBM run from PR #12. Fold scores are in `results/lgbm_tuning_cv.json`. |
| `results/metrics.csv` | Available scores, test-minus-train deviance and test Gini when predictions are available. |
| `results/lorenz_comparison.png` | Test Lorenz curves for models with verified predictions, alongside the oracle and random references. |

A's published JSON supplies the initial LightGBM row. E's published JSON
supplies the tuned row when the tuning outputs are available.
The CSV labels each recorded run and checks the shared split settings, sample
counts, target, weights and Tweedie power. The published runs lack policy
fingerprints and claim totals. Those CSV fields stay blank until locally scored
metrics are available.
The `compare` command checks policy membership and recalculates test deviance
from each prediction file before plotting. The figure uses the instructor's
policy-count axis and reports unnormalized Gini as `1 - 2 * AUC`. Lower test
deviance indicates better out-of-sample fit, while higher Gini indicates stronger
risk ranking. The train–test deviance gap is a diagnostic for generalisation.

D and E can call `model_metrics()` from `analyses.part_c` with the common train and
test frames and policy-indexed predictions for `parametric_glm` and
`lightgbm_tuned`. Save their metrics via `write_model_metrics()` and keep each test
prediction Series in `results/<model>_predictions.csv` with columns `IDpol` and
`PurePremiumPredicted`. These prediction files are gitignored. For E, use
`run_lgbm_tuning()` from `analyses.lgbm_tuning` to obtain the selected model, then
`predict_pure_premium()` from `analyses.lgbm_baseline` on the common test frame.
Rerun `compare` as model outputs become available.

Run the project tests with `python -m pytest`.

## Directory structure and collaboration

| Location | Purpose |
| --- | --- |
| `ps3/data/` | Instructor-provided data loader and Ex1 stable split |
| `ps3/preprocessing/` | Ex2 Winsorizer and shared preprocessing |
| `analyses/ps3_script.py` | Exercise 3 shared input and runnable analysis entry point |
| `tests/` | Tests for the deterministic split, Winsorizer and shared modelling pipeline |
| `TEAM_ASSIGNMENTS.md` | A–E responsibilities, dependencies and deliverables for each exercise |

`ps3.data.create_sample_column` is the Ex1 split API. The earlier
`create_sample_split` entry point is still available. For D's improved GLM,
use the feature requirements in Exercise 3 and the assignments file.

Create your own branch from the latest `main` and submit a PR. Update from `main` after
prerequisite changes are merged. Keep model modules and outputs separate to reduce concurrent
edits to the main script. Use training data only for preprocessing and tuning, and keep the
data split, target, weights and evaluation conventions consistent across models. Update the
running instructions, results and necessary explanations when your work is complete.

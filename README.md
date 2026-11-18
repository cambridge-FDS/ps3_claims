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

The main analysis entry point is:

```bash
python analyses/ps3_script.py
```

**Run this command after completing the coursework implementation.** The current scaffold does
not yet create `sample`; the Winsorizer methods are empty, and the pipelines and search still
contain TODOs. Running the full script now will stop at unfinished sections. You can first run
the imports/data-loading cells individually in VS Code, then connect the subsequent stages
according to the assignments.

The Ex2 assignment test entry point is:

```bash
python -m pytest tests/preprocessing/test_winsorizer.py
```

The `assert False` is an instructor-provided test placeholder for B to complete alongside the
Transformer implementation. The full pytest suite should therefore not currently be used to
decide whether the environment was installed successfully.

## Directory structure and collaboration

| Location | Purpose |
| --- | --- |
| `ps3/data/` | Instructor-provided data loader and Ex1 stable split |
| `ps3/preprocessing/` | Ex2 Winsorizer and shared preprocessing |
| `analyses/ps3_script.py` | Instructor-provided analysis scaffold, to connect the models later |
| `tests/` | Basic tests and assignment tests still to be completed |
| `TEAM_ASSIGNMENTS.md` | A–E responsibilities, dependencies and deliverables for each exercise |

The official Ex1 instructions use `create_sample_column`, while the repository currently
exports `create_sample_split`. A will reconcile the names during implementation and retain a
compatible entry point. The Ex3 feature requirements also differ from the script comments;
see the assignments file.

Create your own branch from the latest `main` and submit a PR. Update from `main` after
prerequisite changes are merged. Keep model modules and outputs separate to reduce concurrent
edits to the main script. Use training data only for preprocessing and tuning, and keep the
data split, target, weights and evaluation conventions consistent across models. Update the
running instructions, results and necessary explanations when your work is complete.

# Repository Guidelines

## Project Structure & Module Organization

`sat_net/` contains the core event-driven satellite routing simulator, including topology, nodes, links, traffic, metrics, and `sat_net/solver/` routing algorithms such as PRIMAL-CVaR, PRIMAL-Avg, DQN, IQN, SAC, and SPF. `satnet_viewer/` provides the OpenGL/ImGui visualization app. Root-level scripts are entry points: `run_train.py`, `run_eval.py`, `run_spf.py`, `run_satnet_viewer.py`, and plotting utilities named `plot_*.py`. JSON environment and solver settings live in `configs/`. Static map data is in `assets/`; generated figures, runs, and model checkpoints are in `figs/`, `runs_*`, and `saved_models/`.

## Build, Test, and Development Commands

Create the Python 3.11 environment and install dependencies:

```bash
conda create -n risk_aware_routing python=3.11
conda activate risk_aware_routing
pip install -r requirements.txt
```

Run core workflows from the repository root:

```bash
python run_spf.py
python run_eval.py
python run_train.py --solver=configs/primal_cvar.json --num_epochs=10
python run_satnet_viewer.py
tensorboard --logdir=runs_train
```

Use `python -m compileall sat_net satnet_viewer *.py` as a fast syntax check before longer simulations.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and type hints where they clarify simulator contracts. Keep modules focused around existing boundaries: environment logic in `routing_env.py`, graph/topology logic in `network.py`, and algorithm-specific code under `sat_net/solver/`. Follow existing naming patterns: classes use `PascalCase`, functions and variables use `snake_case`, and config files use descriptive lowercase names such as `primal_cvar.json`. `.pylintrc` allows 120-character lines and disables docstring and strict naming warnings; run `pylint sat_net run_train.py run_eval.py` when changing shared code.

## Testing Guidelines

There is no dedicated automated test suite yet. For behavior changes, add focused tests under `tests/` using `test_*.py` names if introducing pytest. At minimum, run `compileall` plus a short deterministic smoke command such as `python run_spf.py`; for solver changes, run a reduced training command with `--num_epochs` and record the config and seed used.

## Commit & Pull Request Guidelines

The current history uses short messages such as `update` and `Update ReadMe.md`; prefer more specific imperative summaries, for example `solver: tune primal cvar checkpoint loading`. Pull requests should describe the simulator or solver behavior changed, list commands run, identify modified configs, and note whether outputs in `figs/`, `runs_*`, or `saved_models/` are intentionally included. Avoid committing transient run logs unless they are required reproducibility artifacts.

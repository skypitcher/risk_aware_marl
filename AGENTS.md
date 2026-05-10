# Repository Guidelines

## Project Structure & Module Organization

`sat_net/` contains the fixed-step, data-oriented satellite routing simulator. `sat_net/network.py` stores topology, neighbor/link matrices, and dense SPF next-hop rows in NumPy arrays, `sat_net/sim_kernel.py` owns mask-first flowlet/link transitions, `sat_net/routing_env.py` orchestrates topology, traffic, and solver calls, and `sat_net/solver/` defines the batched routing policy contract plus the SPF baseline with optional JAX/JIT next-hop execution. Root-level scripts are entry points: `run_train.py`, `run_eval.py`, `run_spf.py`, and plotting utilities named `plot_*.py`. JSON environment and solver settings live in `configs/`. Static map data is in `assets/`; generated figures and run outputs are in `figs/` and `runs_*`.

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
python run_train.py --solver=configs/spf.json --num_epochs=1
```

Use `python -m compileall sat_net *.py` as a fast syntax check before longer simulations.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and type hints where they clarify simulator contracts. Keep modules focused around existing boundaries: environment logic in `routing_env.py`, graph/topology logic in `network.py`, and batched policy code under `sat_net/solver/`. Follow existing naming patterns: classes use `PascalCase`, functions and variables use `snake_case`, and config files use descriptive lowercase names such as `spf.json`. `.pylintrc` allows 120-character lines and disables docstring and strict naming warnings; run `pylint sat_net run_train.py run_eval.py` when changing shared code.

## Testing Guidelines

There is no dedicated automated test suite yet. For behavior changes, add focused tests under `tests/` using `test_*.py` names if introducing pytest. At minimum, run `compileall` plus a short deterministic smoke command such as `python run_spf.py`; for solver changes, run a reduced training command with `--num_epochs` and record the config and seed used.

## Commit & Pull Request Guidelines

The current history uses short messages such as `update` and `Update ReadMe.md`; prefer more specific imperative summaries, for example `solver: add batched policy interface`. Pull requests should describe the simulator or solver behavior changed, list commands run, identify modified configs, and note whether outputs in `figs/` or `runs_*` are intentionally included. Avoid committing transient run logs unless they are required reproducibility artifacts.

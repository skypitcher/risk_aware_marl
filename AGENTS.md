# Repository Guidelines

## Project Structure & Module Organization

`sat_net/` contains the NumPy slot-array satellite routing simulator. Key modules are `array_vector_env.py` for fixed-slot vector environment orchestration, `flowlet_status.py` for shared state constants, `network.py` for topology and SPF tables, `traffic_region.py` for population-driven traffic, and `sat_net/agent/` for SPFAgent, MaDQN, PRIMAL-Avg, and PRIMAL-CVaR. Entry points are `run_train.py`, `run_eval.py`, and `run_spf.py`. Configs are under `configs/`, population assets under `assets/population/`, figures under `figs/`, and run outputs under `runs_*`.

## Build, Test, and Development Commands

Create the environment and install dependencies:

```bash
conda create -n risk_aware_routing python=3.11
conda activate risk_aware_routing
pip install -r requirements.txt
```

Run common workflows from the repository root:

```bash
python run_spf.py
python run_eval.py
python run_train.py --config configs/main.json --agent configs/agents/primal_cvar.json --max_sampling_steps 30000 --eval_interval_steps 0
```

Use `python -m compileall sat_net run_train.py run_eval.py run_spf.py` before longer simulations.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and useful type hints. Keep changes inside existing boundaries: environment logic and array kernels in `array_vector_env.py`, topology in `network.py`, traffic generation in `traffic_region.py`, and agents under `sat_net/agent/`. Classes use `PascalCase`; functions, variables, and config keys use `snake_case`. Config files should be descriptive lowercase names such as `configs/agents/spf.json`.

## Testing Guidelines

There is no dedicated automated test suite yet. For behavior changes, run `compileall` plus a short smoke command. For agent or reward changes, run a reduced training job and inspect `runs_train/<run_id>/metrics/` and replay stats. Add focused `tests/test_*.py` pytest files when introducing reusable checks.

## MARL Modeling Notes

Treat transitions as packet/flowlet trajectories, not node trajectories. Each row is decided by the current satellite agent, but the next decision for the same flowlet may belong to another satellite. Use CTDE by default: MaDQN and PRIMAL share replay and a global model for centralized training, while execution uses local observations and action masks.

Preserve `flowlet_id`, current `agent_id`, and next decision context when changing replay. Because flowlets aggregate packets, weight losses or metrics by `packet_count` or `flowlet_size` when approximating packet-level objectives. Preserve the legacy 94-dimensional observation and reward scale from tag `pre-sim-kernel-optimization-20260510` unless deliberately redesigning the learning problem: PRIMAL separates non-queue delay reward from queue-delay cost, while MaDQN penalizes full action delay in its baseline reward.

## Commit & Pull Request Guidelines

History currently has short messages such as `update`; prefer specific imperative summaries like `agent: restore legacy reward scale`. Pull requests should describe simulator or agent behavior changes, list commands run, identify modified configs, and note whether `figs/` or `runs_*` outputs are intentional artifacts.

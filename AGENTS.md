# Repository Guidelines

## Project Structure & Module Organization

`sat_net/` contains the fixed-step, data-oriented satellite routing simulator. `sat_net/network.py` stores topology, neighbor/link matrices, and dense SPF next-hop rows in NumPy arrays, `sat_net/sim_kernel.py` owns mask-first flowlet/link transitions, `sat_net/routing_env.py` orchestrates topology and traffic, and `sat_net/agent/` defines the batched MARL agent contract plus SPFAgent, MaDQN, PRIMAL-Avg, and PRIMAL-CVaR. The default research path is NumPy/SciPy simulation with PyTorch agents. Root-level scripts are entry points: `run_train.py`, `run_eval.py`, `run_spf.py`, `scripts/fetch_worldpop_population.py`, and plotting utilities named `plot_*.py`. JSON environment and agent settings live in `configs/`. Static map data is in `assets/`, including the WorldPop population grid in `assets/population/`; generated figures and run outputs are in `figs/` and `runs_*`.

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
python run_train.py --agent=configs/dqn.json --num_epochs=1 --eval_interval=0
```

Use `python -m compileall sat_net *.py` as a fast syntax check before longer simulations.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and type hints where they clarify simulator contracts. Keep modules focused around existing boundaries: environment logic in `routing_env.py`, graph/topology logic in `network.py`, and batched agent code under `sat_net/agent/`. Follow existing naming patterns: classes use `PascalCase`, functions and variables use `snake_case`, and config files use descriptive lowercase names such as `spf.json`. `.pylintrc` allows 120-character lines and disables docstring and strict naming warnings; run `pylint sat_net run_train.py run_eval.py` when changing shared code.

## Testing Guidelines

There is no dedicated automated test suite yet. For behavior changes, add focused tests under `tests/` using `test_*.py` names if introducing pytest. At minimum, run `compileall` plus a short deterministic smoke command such as `python run_spf.py`; for agent changes, run a reduced training command and inspect `runs_train/<run_id>/metrics/`.

## MARL Modeling Notes

Treat the underlying decision process as packet/flowlet trajectories, not node trajectories. A transition follows one flowlet as it moves from its current satellite to the next satellite, delivery, or drop; the next decision may belong to a different satellite agent. The decision owner for each row is still the current node, exposed as `RoutingBatch.agent_ids`.

Use CTDE as the default learning assumption. MaDQN and PRIMAL agents should use shared replay and a shared global model for centralized training, while execution remains decentralized through local per-node observations and action masks. This is parameter-sharing MARL, not fully independent per-node learning.

When changing replay or rewards, preserve `flowlet_id`, current `agent_id`, and the next decision context so packet trajectories remain reconstructable. If adding truly independent MARL later, each node needs its own memory/model, and bootstrap targets must use the next node's value model, not blindly reuse the previous node's model. Because flowlets aggregate packets, weight losses or metrics by `packet_count` or `flowlet_size` when the objective is intended to approximate packet-level behavior.

## Commit & Pull Request Guidelines

The current history uses short messages such as `update` and `Update ReadMe.md`; prefer more specific imperative summaries, for example `agent: add batched MARL interface`. Pull requests should describe the simulator or agent behavior changed, list commands run, identify modified configs, and note whether outputs in `figs/` or `runs_*` are intentionally included. Avoid committing transient run logs unless they are required reproducibility artifacts.

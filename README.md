# Asynchronous Risk-Aware Multi-Agent Packet Routing for Ultra-Dense LEO Satellite Networks

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2510.27506-b31b1b.svg)](https://[arxiv.org/abs/2510.27506](https://arxiv.org/abs/2510.27506))

> **Official implementation** of the preprint paper: "Asynchronous Risk-Aware Multi-Agent Packet Routing for Ultra-Dense LEO Satellite Networks"
>
> **Authors**: Ke He, Thang X. Vu, Le He, Lisheng Fan, Symeon Chatzinotas, and Björn Ottersten
>
> 📄 **[Read Paper (PDF)](https://arxiv.org/abs/2510.27506)**

![Ultra-Dense LEO Constellation](figs/topology.svg)

## 🌟 Overview

This repository presents **PRIMAL** (Principled Risk-aware Independent Multi-Agent Learning), a risk-aware multi-agent routing framework for ultra-dense LEO satellite networks. The current simulator targets the first shell of Starlink scale (1584 satellites), dynamic topology, region-based terrestrial traffic, and propagation/queueing delay modeling.

### Slot-Array Simulator

The simulator has been refactored from an event-queue packet model into a data-oriented slot-array kernel:

1. **Region traffic**: Flowlets are sampled from population/region weights and bind to the currently visible access satellite.
2. **Flowlet batches**: A flowlet represents a batch of packets with common source/target regions.
3. **Array network state**: Satellite positions, neighbor/link matrices, link delays, connectivity, queues, and flowlet state are stored in NumPy arrays.
4. **Mask-first slot kernels**: `sat_net/sim_kernel.py` owns flowlet/link state transitions and returns full-length masks before NumPy compression.
5. **MARL env API**: `RoutingEnv.reset()` returns a batched multi-agent `RoutingBatch`; every row is one satellite-agent decision for one flowlet.
6. **MARL training pipeline**: `sat_net/pipeline.py` runs `reset -> agent.act -> env.step -> observe outcomes -> train update` episodes.
7. **Reward and experiment logs**: `sat_net/reward.py` defines configurable transition rewards, and `sat_net/experiment.py` writes manifests, checkpoints, and train/eval metrics.
8. **SPF baseline**: Shortest-path next-hop rows are computed from sparse arrays, cached in a dense matrix, and refreshed with topology updates.
9. **Torch RL on NumPy kernels**: The default path is NumPy/SciPy simulation plus PyTorch MaDQN/PRIMAL training.

The default Starlink configs use a downsampled WorldPop 2020 total-population grid at `assets/population/worldpop_total_population_2020_1440x720.npy`; regenerate it with `python scripts/fetch_worldpop_population.py`.

### 📊 Key Results

- **70% reduction** in queuing delay (i.e. network congestion) compared to risk-oblivious baselines
- **12ms improvement** in end-to-end delay under loaded scenarios
- **5.8% CVaR violation rate** vs 75.5% for traditional approaches
- Simulates global region-to-region traffic over a dense network of **1584 satellites**

### Technical Development

Our PRIMAL framework resolves the fundamental conflict between shortest-path routing and congestion avoidance through:
- **Data-oriented simulation**: Routing, queues, topology updates, and flowlet state are represented as arrays
- **Primal-dual optimization**: Principled constraint handling without manual reward engineering to avoid reward-hacking
- **Implicit Quantile Networks**: Capture full distribution of routing outcomes
- **CVaR constraints**: Direct control over worst-case performance degradation

## 📋 Requirements

### System Requirements
- Python 3.11+
- 32GB RAM recommended for large Starlink-scale runs
- Ubuntu 20.04+ / Windows 10+ / macOS 12+

### Installation

```bash
# Clone the repository
git clone https://github.com/skypitcher/risk_aware_marl.git
cd risk_aware_marl

# Create conda environment
conda create -n risk_aware_routing python=3.11
conda activate risk_aware_routing

# Install dependencies
pip install -r requirements.txt
```

#### Troubleshooting

<details>
<summary>Cartopy installation issues</summary>

On some systems, Cartopy may require additional dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install libproj-dev proj-data proj-bin libgeos-dev

# macOS
brew install proj geos
```
</details>

## 🧠 Routing Agents

- **SPFAgent**: Dijkstra shortest-path first, backed by dense satellite next-hop rows and region-to-next-hop tables.
- **Batched MARL contract**: `sat_net/agent/base_agent.py` defines `RoutingBatch`, `RoutingDecision`, and `BaseAgent`.
- **RL agents**: MaDQN, PRIMAL-Avg, and PRIMAL-CVaR have been rebuilt for batched routing with PyTorch replay buffers. MaIQN/MaSAC remain retired until they are ported to the same interface.

## 📁 Project Structure

```
risk_aware_marl/
├── sat_net/                    # Core simulation framework
│   ├── routing_env.py          # Slot-array routing environment
│   ├── pipeline.py             # MARL train/eval episode pipeline
│   ├── reward.py               # Transition reward and cost shaping
│   ├── experiment.py           # Run manifests, metrics, and checkpoints
│   ├── sim_kernel.py           # Flowlet/link array transition kernels
│   ├── network.py              # Array-oriented satellite network topology
│   ├── traffic_region.py       # Region/population traffic model
│   └── agent/                  # Batched MARL routing-agent API
│       ├── base_agent.py       # RoutingBatch/RoutingDecision/BaseAgent contract
│       └── spf.py              # Shortest-path baseline
├── configs/                    # Configuration files
│   ├── main.json               # Active experiment entry point
│   ├── env/                    # Scenario and traffic configurations
│   ├── agents/                 # SPF, MaDQN, and PRIMAL configs
│   └── traffic/                # Fallback hand-written traffic regions
├── assets/population/          # Downsampled WorldPop population grid
├── figs/                       # Figures and plots
└── runs_*/                     # Experiment results
```

## 🚀 Quick Start

### SPF Baseline

```bash
# Generate SPF baseline results
python run_spf.py

# Evaluate SPF across seeds
python run_eval.py
```

### RL Training Status

`run_train.py` executes agent episodes through `sat_net/pipeline.py`. `configs/main.json` selects the scenario, while `configs/agents/*.json` selects the algorithm. Use `configs/agents/madqn.json`, `configs/agents/primal_avg.json`, or `configs/agents/primal_cvar.json` to train the rebuilt MaDQN/PRIMAL baselines.

```bash
python run_train.py --config configs/main.json --agent configs/agents/madqn.json --num_epochs 10 --eval_interval 5
```

Each run writes `manifest.json`, `summary.json`, checkpoint state, and CSV/JSONL metrics under `runs_train/<run_id>/`.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

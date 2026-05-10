# Asynchronous Risk-Aware Multi-Agent Packet Routing for Ultra-Dense LEO Satellite Networks

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2510.27506-b31b1b.svg)](https://[arxiv.org/abs/2510.27506](https://arxiv.org/abs/2510.27506))
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

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
3. **Array network state**: Satellite positions, link endpoints, link delays, connectivity, queues, and flowlet state are stored in NumPy arrays.
4. **SPF baseline**: Shortest-path next-hop rows are computed from sparse arrays and refreshed with topology updates.
5. **RL status**: Legacy RL solver classes remain in the repository, but the new kernel needs a batched transition API before RL training is re-enabled.

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
- CUDA 11.8+ (for GPU acceleration)
- 32GB RAM (recommended for training)
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
<summary>CUDA/PyTorch issues</summary>

If you encounter CUDA compatibility issues:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
</details>

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

## 🧠 Implemented Algorithms

### Our Contributions (PRIMAL Framework)
- **PRIMAL-CVaR** 🎯: Risk-aware routing with CVaR constraints at configurable risk levels (e.g., ε=0.25)
  - Learns full cost distribution via Implicit Quantile Networks
  - Directly constrains tail-end risks for robust performance
- **PRIMAL-Avg** 📊: Risk-neutral variant with expectation-based constraints
  - Optimizes average performance with primal-dual learning
  - Serves as ablation study for risk-awareness benefits

### Baseline Methods
- **SPF**: Dijkstra's Shortest Path First - Precomputed routing based on predictable orbital movements
- **MADQN**: Multi-agent asynchronous DQN with heuristic reward shaping [Lozano-Cuadra et al., 2025]
- **MaIQN**: Multi-agent Implicit Quantile Network (distributional but risk-oblivious)
- **MaSAC**: Multi-agent Soft Actor-Critic with maximum entropy

## 📁 Project Structure

```
risk_aware_marl/
├── sat_net/                    # Core simulation framework
│   ├── routing_env.py          # Slot-array routing environment
│   ├── network.py              # Array-oriented satellite network topology
│   ├── traffic_region.py       # Region/population traffic model
│   └── solver/                 # Routing algorithms
│       ├── primal_cvar.py      # Our risk-aware algorithm
│       ├── primal_avg.py       # Our risk-neutral algorithm
│       ├── dqn.py              # DQN baseline
│       └── spf.py              # Traditional routing
├── configs/                    # Configuration files
│   ├── starlink_dvbs2_*.json  # Network configurations
│   └── *.json                  # Algorithm hyperparameters
├── saved_models/               # Pre-trained models
├── figs/                       # Figures and plots
└── runs_*/                     # Experiment results
```

## 🚀 Quick Start

### SPF Baseline

```bash
# Generate SPF baseline results
python run_spf.py

# Evaluate configured solvers; unsupported legacy RL solvers are skipped
python run_eval.py
```

### RL Training Status

The legacy RL solver implementations are still present, but the slot-array simulator currently supports SPF only. RL training requires a batched transition API before `run_train.py` can be used for PRIMAL/MADQN/MaIQN/MaSAC again.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

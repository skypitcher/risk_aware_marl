"""
Script for evaluating trained models and baseline solvers (e.g., SPF) on the satellite network environment.
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import List

import pandas as pd

from sat_net.routing_env import RoutingEnvAsync
from sat_net.solver import SPF, BaseSolver, create_solver
from sat_net.util import NamedDict

TEST_RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"RUN_ID: {TEST_RUN_ID}")

PROJECT_ROOT = str(os.path.dirname(os.path.abspath(__file__)))
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)

RUNS_DIR = os.path.join(PROJECT_ROOT, "runs_eval")

LOG_DIR = os.path.join(RUNS_DIR, TEST_RUN_ID)
print(f"LOG_DIR: {LOG_DIR}")


def setup_logging(log_dir: str):
    """
    Sets up logging to file and console.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "console.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def evaluate_multi_seed(env: RoutingEnvAsync, solvers: List[BaseSolver], base_seed: int, num_seeds: int):
    """
    Evaluate multiple solvers with multiple seeds and collect comprehensive results.

    Args:
        env: The routing environment
        solvers: List of solvers to evaluate
        base_seed: Base seed for evaluation
        num_seeds: Number of different seeds to use

    Returns:
        Dictionary with comprehensive results for each solver
    """
    for solver in solvers:
        solver.set_eval()

        all_packets = []
        for seed_idx in range(num_seeds):
            eval_seed = base_seed + seed_idx * 1000  # Ensure seeds are well separated
            logging.info("Solver: %s  Seed %d/%d (seed=%d)", solver.name, seed_idx + 1, num_seeds, eval_seed)
            start_time = time.time()
            env.reset(seed=eval_seed, start_time=0)
            env.run(solver)
            eval_time = time.time() - start_time
            logging.info("Evaluation finished in %.2fs", eval_time)
            metrics = env.calc_metrics()
            logging.info("Test metrics: %s", metrics.to_json())
            all_packets.extend(env.generated_packets)

        logging.info("Solver: %s evaluated", solver.name)

        generated_df = pd.DataFrame([packet.to_dict() for packet in all_packets])
        generated_path = os.path.join(LOG_DIR, f"{solver.name}_packets.csv")
        generated_df.to_csv(generated_path, index=False)
        logging.info("%d packets saved to %s", len(all_packets), generated_path)
        print("")


def load_solver_from(env, saved_path: str):
    """
    Loads a trained solver from a specified path.
    """
    solver_config = NamedDict.load(f"{saved_path}/solver_config.json")
    solver = create_solver(obs_dim=env.obs_dim, action_dim=env.action_dim, solver_config=solver_config, tf_writer=None)
    solver.load_models(f"{saved_path}/models/best_model")
    solver.set_eval()
    return solver


def run_evaluation(model_path_list: list[str], eval_seed: int, num_eval_seeds: int):
    """
    Main function to set up and run the evaluation process.
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Setup logging
    setup_logging(LOG_DIR)

    # Load config and initialize environment and solver
    env_config = NamedDict.load("configs/starlink_dvbs2_test.json")
    logging.info("env_config: %s", env_config.to_string())
    env = RoutingEnvAsync(env_config, tf_writer=None)

    # Initialize solvers
    solvers = [load_solver_from(env, model_path) for model_path in model_path_list]
    solvers.append(SPF())

    if not solvers:
        logging.error("No solvers available for evaluation")
        return

    # Run multi-seed evaluation
    logging.info("Starting multi-seed evaluation with %d solvers and %d seeds", len(solvers), num_eval_seeds)
    evaluate_multi_seed(env, solvers, eval_seed, num_eval_seeds)


if __name__ == "__main__":
    run_evaluation(
        model_path_list=[
            "saved_models/madqn",
            "saved_models/primal_avg",
            "saved_models/primal_cvar",           
        ], 
        eval_seed=3333, 
        num_eval_seeds=5
    )

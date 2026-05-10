"""
Evaluate trained MARL routing agents and baseline agents.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from sat_net.agent import BaseAgent, create_agent
from sat_net.config import DEFAULT_MAIN_CONFIG, eval_agent_paths, load_config, load_env_config, merge_section
from sat_net.pipeline import run_marl_episode
from sat_net.routing_env import RoutingEnv
from sat_net.util import NamedDict

PROJECT_ROOT = str(os.path.dirname(os.path.abspath(__file__)))
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "console.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def evaluate_multi_seed(
    env: RoutingEnv,
    agents: list[BaseAgent],
    base_seed: int,
    num_seeds: int,
    log_dir: str,
):
    for agent in agents:
        all_flowlet_frames = []
        for seed_idx in range(num_seeds):
            eval_seed = base_seed + seed_idx * 1000
            logging.info("Agent: %s  Seed %d/%d (seed=%d)", agent.name, seed_idx + 1, num_seeds, eval_seed)
            try:
                result = run_marl_episode(env=env, agent=agent, seed=eval_seed, start_time=0, train=False)
            except NotImplementedError as exc:
                logging.warning("Skipping agent %s: %s", agent.name, exc)
                break

            logging.info("Evaluation finished in %.2fs", result.elapsed_seconds)
            logging.info("Test metrics: %s", result.metrics.to_json())
            flowlet_df = env.get_flowlet_dataframe()
            if not flowlet_df.empty:
                flowlet_df.insert(0, "seed", eval_seed)
                flowlet_df.insert(0, "agent", agent.name)
                all_flowlet_frames.append(flowlet_df)

        generated_df = pd.concat(all_flowlet_frames, ignore_index=True) if all_flowlet_frames else pd.DataFrame()
        generated_path = os.path.join(log_dir, f"{agent.name}_flowlets.csv")
        generated_df.to_csv(generated_path, index=False)
        logging.info("%d flowlets saved to %s", len(generated_df), generated_path)


def load_agent_from(env: RoutingEnv, saved_path: str):
    agent_config = NamedDict.load(f"{saved_path}/agent_config.json")
    agent = create_agent(
        agent_config,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
    )
    agent.load_models(f"{saved_path}/models/best_model")
    agent.set_eval()
    return agent


def main():
    eval_defaults = {
        "config": DEFAULT_MAIN_CONFIG,
        "env": None,
        "agent": None,
        "model": [],
        "eval_seed": 3333,
        "num_eval_seeds": 5,
        "runs_dir": "runs_eval",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_MAIN_CONFIG)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--agent", action="append", default=None)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument("--num_eval_seeds", type=int, default=None)
    parser.add_argument("--runs_dir", type=str, default=None)
    parsed_args = parser.parse_args()
    main_config = load_config(parsed_args.config)
    args = merge_section(eval_defaults, main_config, "eval", vars(parsed_args))

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runs_dir = os.path.join(PROJECT_ROOT, args.runs_dir)
    log_dir = os.path.join(runs_dir, run_id)
    print(f"RUN_ID: {run_id}")
    print(f"LOG_DIR: {log_dir}")
    setup_logging(log_dir)

    env_config = load_env_config(main_config, split="eval", override_path=args.env)
    main_config.save(os.path.join(log_dir, "main_config.json"))
    env_config.save(os.path.join(log_dir, "env_config.json"))
    logging.info("env_config: %s", env_config.to_string())
    env = RoutingEnv(env_config, tf_writer=None)

    agent_config_paths = eval_agent_paths(main_config, overrides=args.agent)
    agents = [
        create_agent(
            NamedDict.load(agent_config_path),
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
        )
        for agent_config_path in agent_config_paths
    ]
    agents.extend(load_agent_from(env, model_path) for model_path in args.model)

    logging.info("Starting multi-seed evaluation with %d agents and %d seeds", len(agents), args.num_eval_seeds)
    evaluate_multi_seed(
        env=env,
        agents=agents,
        base_seed=args.eval_seed,
        num_seeds=args.num_eval_seeds,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()

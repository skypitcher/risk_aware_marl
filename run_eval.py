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
from sat_net.array_vector_env import DAY_MS, ArrayVectorRoutingEnv, seeded_start_offsets_ms
from sat_net.pipeline import run_marl_rollout
from sat_net.util import NamedDict

PROJECT_ROOT = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "console.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def create_eval_env(env_config: NamedDict, num_envs: int, args: NamedDict):
    env_config = NamedDict(env_config.to_dict())
    env_config.verbose = False
    if args.get("concurrent_flowlets_per_env", None) is not None:
        env_config.traffic.concurrent_flowlets_per_env = int(args.concurrent_flowlets_per_env)
    if args.get("region_chunk_size", None) is not None:
        env_config.traffic.region_chunk_size = int(args.region_chunk_size)
    num_envs = max(int(num_envs), 1)
    return ArrayVectorRoutingEnv(
        env_config,
        num_envs=num_envs,
        utc_offset_span_ms=0.0,
        seed_stride=1,
        tf_writer=None,
    )


def evaluate_multi_seed(
    env_config: NamedDict,
    args: NamedDict,
    agents: list[BaseAgent],
    base_seed: int,
    num_seeds: int,
    duration_seconds: float,
    log_dir: str,
):
    metric_records: list[dict[str, object]] = []
    for agent in agents:
        eval_seeds = [base_seed + seed_idx * 1000 for seed_idx in range(num_seeds)]
        env = create_eval_env(env_config, num_envs=len(eval_seeds), args=args)
        actual_envs = max(int(getattr(env, "num_envs", 1)), 1)
        logging.info(
            "Agent %s eval_start seeds=%d duration_env=%.3fs vector_envs=%d",
            agent.name,
            num_seeds,
            duration_seconds,
            actual_envs,
        )
        all_flowlet_frames = []
        try:
            start_offsets_ms = seeded_start_offsets_ms(eval_seeds, span_ms=DAY_MS)
            reset_options = {"env_start_offsets_ms": start_offsets_ms}
            result = run_marl_rollout(
                env=env,
                agent=agent,
                seed=eval_seeds,
                start_time=0.0,
                train=False,
                duration_seconds=duration_seconds,
                reset_options=reset_options,
            )
        except NotImplementedError as exc:
            logging.warning("Skipping agent %s: %s", agent.name, exc)
            continue

        seed_metrics = _eval_seed_metrics(env, result.metrics)
        for seed, metrics in zip(eval_seeds, seed_metrics):
            metric_records.append(
                _metric_record(
                    agent_name=agent.name,
                    seed=seed,
                    phase="eval_seed",
                    metrics=metrics,
                    duration_seconds=duration_seconds,
                    result=result,
                )
            )
        metric_records.append(
            _metric_record(
                agent_name=agent.name,
                seed="aggregate",
                phase="eval_aggregate",
                metrics=result.metrics,
                duration_seconds=duration_seconds,
                result=result,
            )
            | {"num_eval_seeds": len(eval_seeds)}
        )

        logging.info(
            "Evaluation finished: seeds=%d env_steps=%d vector_steps=%d envs=%d simulated_env=%.3fs wall=%.2fs sim_speed=%.2fx",
            len(eval_seeds),
            result.step_stats.get("steps", 0),
            result.step_stats.get("vector_steps", 0),
            result.step_stats.get("num_envs", 1),
            float(result.step_stats.get("duration_ms", 0.0)) / 1000.0,
            result.elapsed_seconds,
            float(result.step_stats.get("sim_speed", 0.0)),
        )
        logging.info("Metrics: %s", result.metrics.get_summary())
        logging.info(
            "Rates: delivery=%.2f%% pending=%.2f%% drop=%.2f%%",
            result.metrics.delivery_rate * 100.0,
            result.metrics.pending_rate * 100.0,
            result.metrics.drop_rate * 100.0,
        )
        flowlet_df = env.get_flowlet_dataframe()
        if not flowlet_df.empty:
            if "env_id" in flowlet_df.columns:
                seed_by_env = {env_id: seed for env_id, seed in enumerate(eval_seeds)}
                flowlet_df.insert(0, "seed", flowlet_df["env_id"].map(seed_by_env))
            else:
                flowlet_df.insert(0, "seed", eval_seeds[0])
            flowlet_df.insert(0, "agent", agent.name)
            all_flowlet_frames.append(flowlet_df)

        generated_df = pd.concat(all_flowlet_frames, ignore_index=True) if all_flowlet_frames else pd.DataFrame()
        generated_path = os.path.join(log_dir, f"{agent.name}_flowlets.csv")
        generated_df.to_csv(generated_path, index=False)
        logging.info("%d flowlets saved to %s", len(generated_df), generated_path)

        metrics_df = pd.DataFrame(metric_records)
        metrics_csv_path = os.path.join(log_dir, "eval_metrics.csv")
        metrics_jsonl_path = os.path.join(log_dir, "eval_metrics.jsonl")
        metrics_df.to_csv(metrics_csv_path, index=False)
        metrics_df.to_json(metrics_jsonl_path, orient="records", lines=True)
        logging.info("Evaluation metrics saved to %s and %s", metrics_csv_path, metrics_jsonl_path)


def _eval_seed_metrics(env: ArrayVectorRoutingEnv, fallback):
    num_envs = max(int(getattr(env, "num_envs", 1)), 1)
    if num_envs <= 1:
        return [fallback]
    return [env.calc_metrics(env_id=env_id) for env_id in range(num_envs)]


def _metric_record(
    agent_name: str,
    seed: int | str,
    phase: str,
    metrics,
    duration_seconds: float,
    result,
) -> dict[str, object]:
    return {
        "agent": agent_name,
        "seed": seed,
        "phase": phase,
        "duration_seconds": float(duration_seconds),
        "env_steps": int(result.step_stats.get("steps", 0)),
        "vector_steps": int(result.step_stats.get("vector_steps", 0)),
        "num_envs": int(result.step_stats.get("num_envs", 1)),
        "wall_seconds": float(result.elapsed_seconds),
        **metrics.to_dict(),
    }


def load_agent_from(env: ArrayVectorRoutingEnv, saved_path: str):
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
        "agent": None,
        "model": [],
        "eval_seed": 3333,
        "duration_seconds": 60.0,
        "num_eval_seeds": 5,
        "vector_utc_span_seconds": 5400.0,
        "vector_seed_stride": 100000,
        "concurrent_flowlets_per_env": None,
        "region_chunk_size": 32,
        "runs_dir": "runs_eval",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_MAIN_CONFIG)
    parser.add_argument("--agent", action="append", default=None)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument("--duration_seconds", type=float, default=None)
    parser.add_argument("--num_eval_seeds", type=int, default=None)
    parser.add_argument("--vector_utc_span_seconds", type=float, default=None)
    parser.add_argument("--vector_seed_stride", type=int, default=None)
    parser.add_argument("--concurrent_flowlets_per_env", type=int, default=None)
    parser.add_argument("--region_chunk_size", type=int, default=None)
    parser.add_argument("--runs_dir", type=str, default=None)
    parsed_args = parser.parse_args()
    main_config = load_config(parsed_args.config)
    args = merge_section(eval_defaults, main_config, "eval", vars(parsed_args))

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runs_dir = os.path.join(PROJECT_ROOT, args.runs_dir)
    log_dir = os.path.join(runs_dir, run_id)
    setup_logging(log_dir)

    env_config = load_env_config(main_config)
    main_config.save(os.path.join(log_dir, "main_config.json"))
    env_config.save(os.path.join(log_dir, "env_config.json"))
    prototype_env = create_eval_env(env_config, num_envs=1, args=args)
    logging.info(
        (
            "RUN_EVAL env=%s seeds=%d vector_envs=%d duration_env=%.3fs "
            "models=%d log_dir=%s"
        ),
        main_config.get("name", "env"),
        int(args.num_eval_seeds),
        int(args.num_eval_seeds),
        float(args.duration_seconds),
        len(args.model),
        log_dir,
    )
    logging.info(
        "SIM sats=%d regions=%d slot=%.3fms",
        int(getattr(prototype_env.network, "num_satellites", 0)),
        len(prototype_env.traffic_model.regions),
        float(prototype_env.slot_ms),
    )

    agent_config_paths = eval_agent_paths(main_config, overrides=args.agent)
    agents = [
        create_agent(
            NamedDict.load(agent_config_path),
            obs_dim=prototype_env.obs_dim,
            action_dim=prototype_env.action_dim,
        )
        for agent_config_path in agent_config_paths
    ]
    agents.extend(load_agent_from(prototype_env, model_path) for model_path in args.model)

    logging.info(
        "Starting multi-seed evaluation with %d agents, %d vectorized seeds",
        len(agents),
        args.num_eval_seeds,
    )
    evaluate_multi_seed(
        env_config=env_config,
        args=args,
        agents=agents,
        base_seed=args.eval_seed,
        num_seeds=args.num_eval_seeds,
        duration_seconds=float(args.duration_seconds),
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()

"""
Main script for training a routing agent in the satellite network environment.
"""

import argparse
import logging
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from sat_net.routing_env import RoutingEnv
from sat_net.pipeline import evaluate_agent, run_marl_rollout
from sat_net.agent import BaseAgent, create_agent
from sat_net.config import DEFAULT_MAIN_CONFIG, load_agent_config, load_config, load_env_config, merge_section
from sat_net.experiment import ExperimentLogger, rollout_record
from sat_net.util import NamedDict, ms2str

PROJECT_ROOT = str(os.path.dirname(os.path.abspath(__file__)))
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)


class NullMetricWriter:
    def add_scalar(self, *_args, **_kwargs):
        pass

    def add_histogram(self, *_args, **_kwargs):
        pass

    def close(self):
        pass


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
        force=True,
    )


def set_seeds(seed):
    """
    Sets random seeds for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def eval_performance(
    env: RoutingEnv,
    agent: BaseAgent,
    sampling_step: int,
    rollout: int,
    experiment: ExperimentLogger,
    eval_seeds: list[int],
    eval_start_time_ms: float,
):
    """
    Evaluates the agent's performance over a fixed set of seeds.
    """
    result = evaluate_agent(env=env, agent=agent, seeds=eval_seeds, start_time=eval_start_time_ms)
    testing_time = sum(item.elapsed_seconds for item in result.rollouts)

    for i, item in enumerate(result.rollouts):
        metrics = item.metrics
        logging.info("Testing performance for seed: %d, progress=%d/%d", item.seed, i + 1, len(result.rollouts))
        logging.info("Tick: %s | %s", ms2str(env.start_time), metrics.get_summary())
        logging.info("Testing metrics: %s", metrics.to_json())
        record = rollout_record(
            sampling_step=sampling_step,
            rollout=rollout,
            phase="eval",
            result=item,
            simulated_time_ms=env.current_time,
        )
        experiment.append_jsonl("metrics/eval_rollouts.jsonl", record)
        experiment.append_csv("metrics/eval_rollouts.csv", record)

    logging.info(
        "Testing finished in %.2fs. Avg metrics: throughput=%.2f±%.2f, drop_rate=%.4f±%.4f, e2e_delay=%.2f±%.2f ms, cost=%.2f",
        testing_time,
        result.throughput_mean,
        result.throughput_std,
        result.drop_rate_mean,
        result.drop_rate_std,
        result.e2e_delay_mean,
        result.e2e_delay_std,
        result.cost_mean,
    )

    aggregate = {
        "sampling_step": sampling_step,
        "rollout": rollout,
        "phase": "eval_mean",
        "elapsed_seconds": testing_time,
        "num_rollouts": len(result.rollouts),
        "metrics": {
            "throughput": result.throughput_mean,
            "throughput_std": result.throughput_std,
            "drop_rate": result.drop_rate_mean,
            "drop_rate_std": result.drop_rate_std,
            "e2e_delay_mean": result.e2e_delay_mean,
            "e2e_delay_std": result.e2e_delay_std,
            "cost_mean": result.cost_mean,
        },
    }
    experiment.append_jsonl("metrics/eval_summary.jsonl", aggregate)
    experiment.append_csv("metrics/eval_summary.csv", aggregate)
    return result.throughput_mean, result.drop_rate_mean, result.e2e_delay_mean, result.cost_mean


def train(
    env: RoutingEnv,
    agent: BaseAgent,
    start_sampling_step: int,
    start_rollout: int,
    start_simulated_time_ms: float,
    start_best_score: float | None,
    log_dir: str,
    tf_writer,
    args: NamedDict,
    experiment: ExperimentLogger,
):
    """
    Continuing training loop. Rollout windows are only sampling chunks; the
    simulated clock advances monotonically across windows.
    """
    logging.info("Training started")

    eval_seeds = parse_seed_list(args.eval_seeds)
    selection_metric = str(args.selection_metric)
    best_score = start_best_score
    sampling_step = int(start_sampling_step)
    rollout = int(start_rollout)
    simulated_time_ms = float(start_simulated_time_ms)
    max_sampling_steps = int(args.max_sampling_steps)
    next_eval_step = _next_interval_step(sampling_step, int(args.eval_interval_steps))
    next_save_step = _next_interval_step(sampling_step, int(args.save_interval_steps))
    next_flowlet_dump_step = _next_interval_step(sampling_step, int(args.flowlet_dump_interval_steps))

    while sampling_step < max_sampling_steps:
        rollout += 1
        remaining_steps = max_sampling_steps - sampling_step
        train_seed = int(args.seed) + rollout if args.seed is not None else None
        logging.info(
            "Rollout %d start: sampling_step=%d/%d simulated_time=%s remaining_steps=%d",
            rollout,
            sampling_step,
            max_sampling_steps,
            ms2str(simulated_time_ms),
            remaining_steps,
        )

        train_result = run_marl_rollout(
            env=env,
            agent=agent,
            seed=train_seed,
            start_time=simulated_time_ms,
            train=True,
            max_steps=remaining_steps,
        )
        rollout_steps = int(train_result.step_stats.get("steps", 0))
        sampling_step += rollout_steps
        simulated_time_ms = float(env.current_time)
        logging.info("Training rollout finished in %.2fs", train_result.elapsed_seconds)

        metrics = train_result.metrics
        logging.info("Tick: %s | %s", ms2str(env.start_time), metrics.get_summary())
        logging.info("Training metrics: %s", metrics.to_json())
        agent_stats = agent.get_stats()
        if agent_stats is not None:
            logging.info("Agent stats: %s", agent_stats)
        logging.info("Reward/train stats: %s", train_result.agent_stats)

        train_record = rollout_record(
            sampling_step=sampling_step,
            rollout=rollout,
            phase="train",
            result=train_result,
            simulated_time_ms=simulated_time_ms,
        )
        experiment.append_jsonl("metrics/train_rollouts.jsonl", train_record)
        experiment.append_csv("metrics/train_rollouts.csv", train_record)

        if int(args.flowlet_dump_interval_steps) > 0 and (
            _interval_due(sampling_step, next_flowlet_dump_step) or sampling_step >= max_sampling_steps
        ):
            flowlet_csv_path = os.path.join(log_dir, f"flowlets/flowlets_step_{sampling_step}.csv")
            os.makedirs(os.path.dirname(flowlet_csv_path), exist_ok=True)
            env.save_flowlets_to_csv(flowlet_csv_path)
            logging.info("Flowlets saved to %s", flowlet_csv_path)
            next_flowlet_dump_step = _advance_interval_step(next_flowlet_dump_step, int(args.flowlet_dump_interval_steps))

        tf_writer.add_scalar("train/throughput", metrics.throughput, global_step=sampling_step)
        tf_writer.add_scalar("train/delivery_rate", metrics.delivery_rate, global_step=sampling_step)
        tf_writer.add_scalar("train/drop_rate", metrics.drop_rate, global_step=sampling_step)
        tf_writer.add_scalar("train/e2e_delay_mean", metrics.e2e_delay_mean, global_step=sampling_step)
        tf_writer.add_scalar("train/cost_mean", metrics.cost_mean, global_step=sampling_step)

        flowlet_df = env.get_flowlet_dataframe()
        delivered_flowlets = flowlet_df[flowlet_df["delivered"]] if not flowlet_df.empty else flowlet_df
        if not delivered_flowlets.empty:
            queue_costs = delivered_flowlets["total_queue_cost"].to_numpy()
            if len(queue_costs) > 0:
                tf_writer.add_histogram("train/queue_costs", queue_costs, global_step=sampling_step)
                tf_writer.add_scalar("train/cost", np.mean(queue_costs), global_step=sampling_step)
                tf_writer.add_scalar("train/cost_std", np.std(queue_costs), global_step=sampling_step)

            first_access_delays = delivered_flowlets["first_access_delay"].to_numpy()
            flowlet_delays = delivered_flowlets["total_delay"].to_numpy()
            small_packet_delays = delivered_flowlets.loc[
                ~delivered_flowlets["is_normal_packet"], "total_delay"
            ].to_numpy()
            normal_packet_delays = delivered_flowlets.loc[
                delivered_flowlets["is_normal_packet"], "total_delay"
            ].to_numpy()
            if len(flowlet_delays) > 0:
                tf_writer.add_histogram("train/all_delays", flowlet_delays, global_step=sampling_step)
                tf_writer.add_scalar("train/e2e_delay_mean", flowlet_delays.mean(), global_step=sampling_step)
                tf_writer.add_scalar("train/e2e_delay_std", flowlet_delays.std(), global_step=sampling_step)
                tf_writer.add_histogram("train/first_access_delays", first_access_delays, global_step=sampling_step)

            if len(small_packet_delays) > 0:
                tf_writer.add_histogram("train/small_packet_delays", small_packet_delays, global_step=sampling_step)
                tf_writer.add_scalar("train/small_packet_delay_mean", small_packet_delays.mean(), global_step=sampling_step)
                tf_writer.add_scalar("train/small_packet_delay_std", small_packet_delays.std(), global_step=sampling_step)

            if len(normal_packet_delays) > 0:
                tf_writer.add_histogram("train/normal_packet_delays", normal_packet_delays, global_step=sampling_step)
                tf_writer.add_scalar("train/normal_packet_delay_mean", normal_packet_delays.mean(), global_step=sampling_step)
                tf_writer.add_scalar("train/normal_packet_delay_std", normal_packet_delays.std(), global_step=sampling_step)

        model_dir_path = os.path.join(log_dir, "models")
        os.makedirs(model_dir_path, exist_ok=True)

        last_model_save_path = os.path.join(model_dir_path, "last_model")
        os.makedirs(last_model_save_path, exist_ok=True)
        agent.save_models(model_dir_path=last_model_save_path)
        experiment.write_json(
            "checkpoint_state.json",
            {
                "sampling_step": sampling_step,
                "rollout": rollout,
                "simulated_time_ms": simulated_time_ms,
                "last_model": str(Path(last_model_save_path).relative_to(log_dir)),
                "selection_metric": selection_metric,
                "best_score": best_score,
            },
        )

        if _interval_due(sampling_step, next_save_step) or sampling_step >= max_sampling_steps:
            step_model_save_path = os.path.join(model_dir_path, f"model_step_{sampling_step}")
            os.makedirs(step_model_save_path, exist_ok=True)
            agent.save_models(model_dir_path=step_model_save_path)
            next_save_step = _advance_interval_step(next_save_step, int(args.save_interval_steps))

        should_eval = int(args.eval_interval_steps) > 0 and sampling_step >= int(args.eval_after_steps)
        should_eval = should_eval and (_interval_due(sampling_step, next_eval_step) or sampling_step >= max_sampling_steps)
        if should_eval:
            avg_throughput, avg_drop_rate, avg_e2e_delay, avg_test_cost = eval_performance(
                env=env,
                agent=agent,
                sampling_step=sampling_step,
                rollout=rollout,
                experiment=experiment,
                eval_seeds=eval_seeds,
                eval_start_time_ms=float(args.eval_start_time_ms),
            )
            next_eval_step = _advance_interval_step(next_eval_step, int(args.eval_interval_steps))
            score = select_model_score(
                selection_metric=selection_metric,
                throughput=avg_throughput,
                drop_rate=avg_drop_rate,
                e2e_delay=avg_e2e_delay,
                cost=avg_test_cost,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_model_save_path = os.path.join(model_dir_path, "best_model")
                os.makedirs(best_model_save_path, exist_ok=True)
                agent.save_models(model_dir_path=best_model_save_path)
                logging.info(
                    "Best model saved to %s, %s score: %.6f",
                    best_model_save_path,
                    selection_metric,
                    best_score,
                )

        experiment.write_json(
            "summary.json",
            {
                "sampling_step": sampling_step,
                "max_sampling_steps": max_sampling_steps,
                "rollout": rollout,
                "simulated_time_ms": simulated_time_ms,
                "best_score": best_score,
                "selection_metric": selection_metric,
                "last_train_metrics": metrics.to_dict(),
                "last_train_agent_stats": train_result.agent_stats,
            },
        )


def _next_interval_step(current_step: int, interval: int) -> int | None:
    if interval <= 0:
        return None
    return ((current_step // interval) + 1) * interval


def _advance_interval_step(current_due: int | None, interval: int) -> int | None:
    if current_due is None or interval <= 0:
        return None
    return current_due + interval


def _interval_due(current_step: int, due_step: int | None) -> bool:
    return due_step is not None and current_step >= due_step


def select_model_score(
    selection_metric: str,
    throughput: float,
    drop_rate: float,
    e2e_delay: float,
    cost: float,
) -> float:
    if selection_metric == "throughput":
        return float(throughput)
    if selection_metric == "drop_rate":
        return -float(drop_rate)
    if selection_metric == "e2e_delay":
        return -float(e2e_delay)
    if selection_metric == "cost":
        return -float(cost)
    if selection_metric == "risk_adjusted":
        return float(throughput) - 1_000_000.0 * float(drop_rate) - 1_000.0 * float(cost)
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def archive_source_code(log_dir):
    """
    Archives the 'sat_net' source code directory into a zip file.
    """
    try:
        logging.info("Archiving source code...")
        archive_base_path = os.path.join(log_dir, "src")
        shutil.make_archive(base_name=archive_base_path, format="zip", root_dir=PROJECT_ROOT, base_dir="sat_net")
        logging.info("Source code archived to %s.zip", archive_base_path)
    except Exception as e:
        logging.error("Failed to archive source code: %s", e)


def main():
    """
    Parses arguments, sets up the environment and agent, and starts the training process.
    """
    train_defaults = {
        "config": DEFAULT_MAIN_CONFIG,
        "recover_runid": None,
        "env": None,
        "agent": None,
        "runs_dir": "runs_train",
        "max_sampling_steps": 600000,
        "start_time_ms": 0.0,
        "seed": 33333,
        "run_id": None,
        "eval_interval_steps": 600000,
        "eval_after_steps": 600000,
        "eval_start_time_ms": 0.0,
        "eval_seeds": "6666,7777,8888",
        "save_interval_steps": 60000,
        "flowlet_dump_interval_steps": 0,
        "selection_metric": "risk_adjusted",
        "notes": "",
        "archive_source": True,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_MAIN_CONFIG)
    parser.add_argument("--recover_runid", type=str, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--agent", type=str, default=None)
    parser.add_argument("--max_sampling_steps", type=int, default=None)
    parser.add_argument("--start_time_ms", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--runs_dir", type=str, default=None)
    parser.add_argument("--eval_interval_steps", type=int, default=None)
    parser.add_argument("--eval_after_steps", type=int, default=None)
    parser.add_argument("--eval_start_time_ms", type=float, default=None)
    parser.add_argument("--eval_seeds", type=str, default=None)
    parser.add_argument("--save_interval_steps", type=int, default=None)
    parser.add_argument("--flowlet_dump_interval_steps", type=int, default=None)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default=None,
        choices=["throughput", "drop_rate", "e2e_delay", "cost", "risk_adjusted"],
    )
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--archive_source", action=argparse.BooleanOptionalAction, default=None)

    parsed_args = parser.parse_args()
    main_config = load_config(parsed_args.config)
    args = merge_section(train_defaults, main_config, "train", vars(parsed_args))

    runs_dir = os.path.join(PROJECT_ROOT, args.runs_dir)
    os.makedirs(runs_dir, exist_ok=True)

    if args.recover_runid is not None:
        run_id = args.recover_runid
        log_dir = os.path.join(runs_dir, run_id)
        os.makedirs(log_dir, exist_ok=True)
        print(f"LOG_DIR: {log_dir}")
        setup_logging(log_dir)

        logging.info("Recovering from %s", run_id)

        saved_args = NamedDict.load(f"{log_dir}/args.json")
        for key, value in saved_args.items():
            if key not in {
                "config",
                "recover_runid",
                "max_sampling_steps",
                "eval_interval_steps",
                "eval_after_steps",
                "eval_seeds",
                "save_interval_steps",
                "flowlet_dump_interval_steps",
                "selection_metric",
                "notes",
                "archive_source",
            }:
                args[key] = value
        logging.info("Loaded args: %s", args.to_string())
        set_seeds(args.seed)
        logging.info("Using seed: %d", args.seed)

        tf_writer = NullMetricWriter()
        experiment = ExperimentLogger(log_dir)

        # Load from existing run
        env_config = NamedDict.load(f"{log_dir}/env_config.json")
        agent_config = NamedDict.load(f"{log_dir}/agent_config.json")

        logging.info("env_config: %s", env_config.to_string())
        logging.info("agent_config: %s", agent_config.to_string())

        env = RoutingEnv(env_config, tf_writer=tf_writer)
        agent = create_agent(
            agent_config,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            tf_writer=tf_writer,
        )
        agent.load_models(f"{log_dir}/models/last_model")
        checkpoint = NamedDict.load(f"{log_dir}/checkpoint_state.json")
        start_sampling_step = int(checkpoint.get("sampling_step", 0))
        start_rollout = int(checkpoint.get("rollout", 0))
        start_simulated_time_ms = float(checkpoint.get("simulated_time_ms", args.start_time_ms))
        start_best_score = checkpoint.get("best_score", None)
        logging.info(
            "Continuing from sampling_step=%d rollout=%d simulated_time=%s",
            start_sampling_step,
            start_rollout,
            ms2str(start_simulated_time_ms),
        )
    else:
        env_config = load_env_config(main_config, split="train", override_path=args.env)
        agent_config = load_agent_config(main_config, override_path=args.agent)

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = args.run_id or f"{agent_config.name}_{date_str}"
        print(f"RUN_ID: {run_id}")

        log_dir = os.path.join(runs_dir, run_id)
        os.makedirs(log_dir, exist_ok=True)
        print(f"LOG_DIR: {log_dir}")
        setup_logging(log_dir)
        experiment = ExperimentLogger(log_dir)

        tf_writer = NullMetricWriter()

        logging.info("args: %s", args.to_string())
        logging.info("main_config: %s", main_config.to_string())
        set_seeds(args.seed)
        logging.info("Using seed: %d", args.seed)

        # create env and agent
        env = RoutingEnv(env_config, tf_writer=tf_writer)
        agent = create_agent(
            agent_config,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            tf_writer=tf_writer,
        )

        logging.info("env_config: %s", env_config.to_string())
        logging.info("agent_config: %s", agent_config.to_string())

        env_config.save(os.path.join(log_dir, "env_config.json"))
        agent_config.save(os.path.join(log_dir, "agent_config.json"))
        main_config.save(os.path.join(log_dir, "main_config.json"))
        args.save(os.path.join(log_dir, "args.json"))
        experiment.save_manifest(
            args=args,
            env_config=env_config,
            agent_config=agent_config,
            notes=args.notes,
        )

        if args.archive_source:
            archive_source_code(log_dir)

        start_sampling_step = 0
        start_rollout = 0
        start_simulated_time_ms = float(args.start_time_ms)
        start_best_score = None

    train(
        env=env,
        agent=agent,
        start_sampling_step=start_sampling_step,
        start_rollout=start_rollout,
        start_simulated_time_ms=start_simulated_time_ms,
        start_best_score=start_best_score,
        log_dir=log_dir,
        tf_writer=tf_writer,
        args=args,
        experiment=experiment,
    )

    tf_writer.close()


if __name__ == "__main__":
    main()

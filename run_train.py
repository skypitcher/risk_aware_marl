"""
Main script for training a routing agent in the satellite network environment.
"""

import argparse
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from sat_net.routing_env import RoutingEnv
from sat_net.vector_env import DAY_MS, VectorRoutingEnv, seeded_start_offsets_ms
from sat_net.pipeline import EvaluationResult, run_marl_rollout, run_marl_steps
from sat_net.agent import BaseAgent, create_agent
from sat_net.config import DEFAULT_MAIN_CONFIG, load_agent_config, load_config, load_env_config, merge_section
from sat_net.experiment import ExperimentLogger, rollout_record
from sat_net.stats import ContinuingMetricsAccumulator, Metrics
from sat_net.util import NamedDict

PROJECT_ROOT = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


class NullMetricWriter:
    def add_scalar(self, *_args, **_kwargs):
        pass

    def add_histogram(self, *_args, **_kwargs):
        pass

    def close(self):
        pass


class TrainingProgressPrinter:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._last_len = 0
        self._wall_start = time.perf_counter()

    def update_train(self, global_step: int, max_steps: int, sim_seconds: float) -> None:
        if not self.enabled:
            return
        max_steps = max(int(max_steps), 1)
        global_step = min(max(int(global_step), 0), max_steps)
        percent = min(max(float(global_step) / float(max_steps), 0.0), 1.0) * 100.0
        wall_seconds = time.perf_counter() - self._wall_start
        eta_seconds = _eta_seconds(wall_seconds, percent)
        line = (
            f"TRAIN_PROGRESS step={global_step:,}/{max_steps:,} "
            f"({percent:6.2f}%) sim={float(sim_seconds):.3f}s "
            f"wall={_format_seconds(wall_seconds)} eta={_format_eta(eta_seconds)}"
        )
        self._write(line)

    def update_eval(self, sim_seconds: float, total_seconds: float, env_steps: int, vector_steps: int) -> None:
        if not self.enabled:
            return
        total_seconds = max(float(total_seconds), 1e-12)
        sim_seconds = min(max(float(sim_seconds), 0.0), total_seconds)
        percent = min(max(sim_seconds / total_seconds, 0.0), 1.0) * 100.0
        wall_seconds = time.perf_counter() - self._wall_start
        eta_seconds = _eta_seconds(wall_seconds, percent)
        line = (
            f"EVAL_PROGRESS sim={sim_seconds:.3f}s/{total_seconds:.3f}s "
            f"({percent:6.2f}%) env_steps={int(env_steps):,} "
            f"vector_steps={int(vector_steps):,} wall={_format_seconds(wall_seconds)} "
            f"eta={_format_eta(eta_seconds)}"
        )
        self._write(line)

    def _write(self, line: str) -> None:
        padding = " " * max(self._last_len - len(line), 0)
        sys.stdout.write("\r" + line + padding)
        sys.stdout.flush()
        self._last_len = len(line)

    def clear(self) -> None:
        if not (self.enabled and self._last_len):
            return
        sys.stdout.write("\r" + (" " * self._last_len) + "\r")
        sys.stdout.flush()
        self._last_len = 0


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


def create_training_env(env_config: NamedDict, args: NamedDict, tf_writer):
    num_envs = max(int(args.get("num_envs", 1)), 1)
    env_config = NamedDict(env_config.to_dict())
    env_config.verbose = bool(args.get("env_verbose", False))
    if num_envs <= 1:
        return RoutingEnv(env_config, tf_writer=tf_writer)
    return VectorRoutingEnv(
        env_config,
        num_envs=num_envs,
        utc_offset_span_ms=float(args.get("vector_utc_span_seconds", 5400.0)) * 1000.0,
        seed_stride=int(args.get("vector_seed_stride", 100000)),
        tf_writer=tf_writer,
    )


def create_evaluation_env(env_config: NamedDict, eval_seeds: list[int], tf_writer):
    env_config = NamedDict(env_config.to_dict())
    env_config.verbose = False
    num_envs = max(len(eval_seeds), 1)
    if num_envs <= 1:
        return RoutingEnv(env_config, tf_writer=tf_writer)
    return VectorRoutingEnv(
        env_config,
        num_envs=num_envs,
        utc_offset_span_ms=0.0,
        seed_stride=1,
        tf_writer=tf_writer,
    )


def assert_separate_train_eval_envs(train_env: RoutingEnv, eval_env: RoutingEnv) -> None:
    if train_env is eval_env:
        raise RuntimeError("Train and eval must use separate environment instances.")
    train_leaf_envs = _leaf_envs(train_env)
    eval_leaf_envs = _leaf_envs(eval_env)
    shared_env_ids = {id(env) for env in train_leaf_envs} & {id(env) for env in eval_leaf_envs}
    if shared_env_ids:
        raise RuntimeError("Train and eval vector environments share sub-environment instances.")
    train_network_ids = {id(env.network) for env in train_leaf_envs if getattr(env, "network", None) is not None}
    eval_network_ids = {id(env.network) for env in eval_leaf_envs if getattr(env, "network", None) is not None}
    if train_network_ids & eval_network_ids:
        raise RuntimeError("Train and eval environments share mutable network instances.")


def log_env_split(train_env: RoutingEnv, eval_env: RoutingEnv, eval_seeds: list[int]) -> None:
    assert_separate_train_eval_envs(train_env, eval_env)
    logging.info(
        "ENV_SPLIT train_envs=%d eval_envs=%d shared_state=false eval_seeds=%s",
        len(_leaf_envs(train_env)),
        len(_leaf_envs(eval_env)),
        ",".join(str(seed) for seed in eval_seeds),
    )


def _leaf_envs(env: RoutingEnv) -> list[RoutingEnv]:
    envs = getattr(env, "envs", None)
    return list(envs) if envs else [env]


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(item.strip()) for item in seed_text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("eval_seeds must contain at least one seed.")
    return seeds


def log_run_setup(
    run_id: str,
    log_dir: str,
    args: NamedDict,
    main_config: NamedDict,
    env_config: NamedDict,
    agent_config: NamedDict,
    env: RoutingEnv,
    recovered: bool = False,
) -> None:
    base_env = _base_env(env)
    traffic = env_config.traffic
    network = env_config.network
    agent_name = agent_config.get("name", "agent")
    env_name = main_config.get("name", "env")
    logging.info(
        (
            "RUN %srun_id=%s agent=%s env=%s seed=%s max_steps=%s "
            "num_envs=%d eval_every=%s eval_after=%s log_dir=%s"
        ),
        "resume " if recovered else "",
        run_id,
        agent_name,
        env_name,
        args.seed,
        _format_optional(args.max_sampling_steps),
        max(int(getattr(env, "num_envs", 1)), 1),
        _format_optional(args.eval_interval_steps),
        _format_optional(args.eval_after_steps),
        log_dir,
    )
    logging.info(
        (
            "SIM sats=%d regions=%d slot=%.3fms start_ms=%.3f traffic=%.1f pkt/ms "
            "mean_flowlet=%.1f pkts access=%.2fGbps isl=%.2fGbps buffer=%.1fMb"
        ),
        int(getattr(base_env.network, "num_satellites", 0)),
        len(base_env.traffic_model.regions),
        float(traffic.get("slot_ms", base_env.slot_ms)),
        float(args.start_time_ms),
        float(traffic.get("packet_rate_per_ms", base_env.packet_rate_per_ms)),
        float(traffic.get("mean_packets_per_flowlet", base_env.mean_packets_per_flowlet)),
        float(traffic.get("access_data_rate", base_env.access_data_rate)),
        float(network.get("isl_data_rate", 0.0)),
        float(network.get("link_buffer_size", 0.0)),
    )


def eval_performance(
    eval_env: RoutingEnv,
    agent: BaseAgent,
    sampling_step: int,
    rollout: int,
    experiment: ExperimentLogger,
    eval_seeds: list[int],
    eval_start_time_ms: float,
    duration_seconds: float,
    tf_writer,
    progress_enabled: bool = True,
):
    """
    Evaluates the agent's performance over a fixed set of seeds.
    """
    num_envs = max(int(getattr(eval_env, "num_envs", 1)), 1)
    was_eval = agent.is_eval()
    progress_printer = TrainingProgressPrinter(enabled=progress_enabled)
    start_offsets_ms = seeded_start_offsets_ms(eval_seeds, span_ms=DAY_MS)
    rollout_start_time_ms = float(eval_start_time_ms)
    reset_options = {}
    if isinstance(eval_env, VectorRoutingEnv):
        reset_options["env_start_offsets_ms"] = start_offsets_ms
    else:
        rollout_start_time_ms += float(start_offsets_ms[0])

    def progress_callback(vector_steps: int) -> None:
        elapsed_ms = max(float(eval_env.current_time - eval_env.start_time), 0.0)
        progress_printer.update_eval(
            sim_seconds=elapsed_ms / 1000.0,
            total_seconds=duration_seconds,
            env_steps=vector_steps * num_envs,
            vector_steps=vector_steps,
        )

    try:
        seed_arg = list(eval_seeds) if isinstance(eval_env, VectorRoutingEnv) else int(eval_seeds[0])
        item = run_marl_rollout(
            env=eval_env,
            agent=agent,
            seed=seed_arg,
            start_time=rollout_start_time_ms,
            train=False,
            duration_seconds=duration_seconds,
            reset_options=reset_options,
            progress_callback=progress_callback,
        )
    finally:
        progress_printer.clear()
        if was_eval:
            agent.set_eval()
        else:
            agent.set_train()

    seed_metrics = _eval_seed_metrics(eval_env, item.metrics)
    for seed, metrics in zip(eval_seeds, seed_metrics):
        tf_writer.add_scalar(f"eval_seed/{seed}/throughput", metrics.throughput, global_step=sampling_step)
        tf_writer.add_scalar(f"eval_seed/{seed}/delivery_rate", metrics.delivery_rate, global_step=sampling_step)
        tf_writer.add_scalar(f"eval_seed/{seed}/pending_rate", metrics.pending_rate, global_step=sampling_step)
        tf_writer.add_scalar(f"eval_seed/{seed}/drop_rate", metrics.drop_rate, global_step=sampling_step)
        tf_writer.add_scalar(f"eval_seed/{seed}/e2e_delay_mean", metrics.e2e_delay_mean, global_step=sampling_step)
        tf_writer.add_scalar(f"eval_seed/{seed}/queue_delay_mean", metrics.queue_delay_mean, global_step=sampling_step)
        tf_writer.add_scalar(
            f"eval_seed/{seed}/propagation_delay_mean",
            metrics.propagation_delay_mean,
            global_step=sampling_step,
        )
        tf_writer.add_scalar(
            f"eval_seed/{seed}/transmission_delay_mean",
            metrics.transmission_delay_mean,
            global_step=sampling_step,
        )
        tf_writer.add_scalar(f"eval_seed/{seed}/cost_mean", metrics.cost_mean, global_step=sampling_step)

    record = rollout_record(
        sampling_step=sampling_step,
        rollout=rollout,
        phase="eval",
        result=item,
        simulated_time_ms=item.step_stats.get("end_time_ms", eval_env.current_time),
    )
    record["optimizer_update_steps"] = _agent_optimizer_update_steps(agent, item.agent_stats)
    record["eval_seeds"] = list(eval_seeds)
    record["eval_start_offsets_ms"] = start_offsets_ms.tolist()
    experiment.append_jsonl("metrics/eval_rollouts.jsonl", record)
    experiment.append_csv("metrics/eval_rollouts.csv", record)

    result = _aggregate_eval_metrics(item, seed_metrics)
    testing_time = item.elapsed_seconds
    queue_delays = np.array([metrics.queue_delay_mean for metrics in seed_metrics], dtype=np.float64)
    propagation_delays = np.array(
        [metrics.propagation_delay_mean for metrics in seed_metrics],
        dtype=np.float64,
    )
    transmission_delays = np.array(
        [metrics.transmission_delay_mean for metrics in seed_metrics],
        dtype=np.float64,
    )
    ttl_drop_rates = np.array(
        [
            metrics.dropped_by_ttl / metrics.generated
            if metrics.generated
            else 0.0
            for metrics in seed_metrics
        ],
        dtype=np.float64,
    )

    logging.info(
        "EVAL %s",
        _format_eval_summary_log(
            step=sampling_step,
            rollout=rollout,
            num_rollouts=1,
            num_envs=num_envs,
            num_eval_seeds=len(eval_seeds),
            duration_seconds=duration_seconds,
            wall_seconds=testing_time,
            result=result,
            ttl_drop_rates=ttl_drop_rates,
            queue_delays=queue_delays,
            propagation_delays=propagation_delays,
            transmission_delays=transmission_delays,
        ),
    )
    tf_writer.add_scalar("eval/throughput_mean", result.throughput_mean, global_step=sampling_step)
    tf_writer.add_scalar("eval/throughput_std", result.throughput_std, global_step=sampling_step)
    tf_writer.add_scalar("eval/delivery_rate_mean", result.delivery_rate_mean, global_step=sampling_step)
    tf_writer.add_scalar("eval/delivery_rate_std", result.delivery_rate_std, global_step=sampling_step)
    tf_writer.add_scalar("eval/pending_rate_mean", result.pending_rate_mean, global_step=sampling_step)
    tf_writer.add_scalar("eval/pending_rate_std", result.pending_rate_std, global_step=sampling_step)
    tf_writer.add_scalar("eval/drop_rate_mean", result.drop_rate_mean, global_step=sampling_step)
    tf_writer.add_scalar("eval/drop_rate_std", result.drop_rate_std, global_step=sampling_step)
    tf_writer.add_scalar("eval/e2e_delay_mean", result.e2e_delay_mean, global_step=sampling_step)
    tf_writer.add_scalar("eval/e2e_delay_std", result.e2e_delay_std, global_step=sampling_step)
    tf_writer.add_scalar("eval/queue_delay_mean", _array_mean(queue_delays), global_step=sampling_step)
    tf_writer.add_scalar("eval/queue_delay_std", _array_std(queue_delays), global_step=sampling_step)
    tf_writer.add_scalar("eval/propagation_delay_mean", _array_mean(propagation_delays), global_step=sampling_step)
    tf_writer.add_scalar("eval/propagation_delay_std", _array_std(propagation_delays), global_step=sampling_step)
    tf_writer.add_scalar("eval/transmission_delay_mean", _array_mean(transmission_delays), global_step=sampling_step)
    tf_writer.add_scalar("eval/transmission_delay_std", _array_std(transmission_delays), global_step=sampling_step)
    tf_writer.add_scalar("eval/cost_mean", result.cost_mean, global_step=sampling_step)

    aggregate = {
        "step": sampling_step,
        "env_interaction_steps": sampling_step,
        "sampling_step": sampling_step,
        "optimizer_update_steps": _agent_optimizer_update_steps(agent, item.agent_stats),
        "rollout": rollout,
        "phase": "eval_mean",
        "elapsed_seconds": testing_time,
        "duration_seconds": duration_seconds,
        "num_envs": num_envs,
        "total_eval_envs": num_envs,
        "num_eval_seeds": len(eval_seeds),
        "eval_seeds": list(eval_seeds),
        "eval_start_offsets_ms": start_offsets_ms.tolist(),
        "num_rollouts": 1,
        "metrics": {
            "generated": item.metrics.generated,
            "delivered": item.metrics.delivered,
            "pending": item.metrics.pending,
            "dropped": item.metrics.dropped,
            "throughput": result.throughput_mean,
            "throughput_std": result.throughput_std,
            "delivery_rate": result.delivery_rate_mean,
            "delivery_rate_std": result.delivery_rate_std,
            "pending_rate": result.pending_rate_mean,
            "pending_rate_std": result.pending_rate_std,
            "drop_rate": result.drop_rate_mean,
            "drop_rate_std": result.drop_rate_std,
            "e2e_delay_mean": result.e2e_delay_mean,
            "e2e_delay_std": result.e2e_delay_std,
            "queue_delay_mean": _array_mean(queue_delays),
            "queue_delay_std": _array_std(queue_delays),
            "propagation_delay_mean": _array_mean(propagation_delays),
            "propagation_delay_std": _array_std(propagation_delays),
            "transmission_delay_mean": _array_mean(transmission_delays),
            "transmission_delay_std": _array_std(transmission_delays),
            "cost_mean": result.cost_mean,
        },
    }
    experiment.append_jsonl("metrics/eval_summary.jsonl", aggregate)
    experiment.append_csv("metrics/eval_summary.csv", aggregate)
    return result.throughput_mean, result.drop_rate_mean, result.e2e_delay_mean, result.cost_mean


def train(
    env: RoutingEnv,
    eval_env: RoutingEnv,
    agent: BaseAgent,
    start_sampling_step: int,
    start_rollout: int,
    start_simulated_time_ms: float,
    start_best_score: float | None,
    start_continuing_state: dict | None,
    log_dir: str,
    tf_writer,
    args: NamedDict,
    experiment: ExperimentLogger,
):
    """
    Continuing training loop. The environment is reset once and then advanced
    until the train-loop interaction-step budget is reached.
    """
    assert_separate_train_eval_envs(env, eval_env)
    logging.info("Training started")

    eval_seeds = parse_seed_list(args.eval_seeds)
    selection_metric = str(args.selection_metric)
    best_score = start_best_score
    sampling_step = int(start_sampling_step)
    iteration = int(start_rollout)
    simulated_time_ms = float(start_simulated_time_ms)
    max_sampling_steps = args.get("max_sampling_steps", None)
    max_sampling_steps = int(max_sampling_steps) if max_sampling_steps is not None else 1_000_000
    if max_sampling_steps <= 0:
        raise ValueError(f"max_sampling_steps must be positive, got {max_sampling_steps}.")
    previous_state = ContinuingMetricsAccumulator.from_state(start_continuing_state)
    previous_elapsed_ms = float(previous_state.elapsed_ms)
    num_envs = max(int(getattr(env, "num_envs", 1)), 1)
    env_start_time_ms = simulated_time_ms
    env.clear_duration_limit()
    agent.set_train()
    agent.on_rollout_start()
    env.reset(
        seed=args.seed,
        start_time=env_start_time_ms,
        options={
            "include_spf_table": agent.requires_shortest_path_table,
        },
    )
    simulated_time_ms = float(env.current_time)
    previous_metrics = Metrics()
    previous_metric_elapsed_ms = 0.0
    previous_agent_stats = agent.get_train_stats()

    next_log_step = _next_interval_step(sampling_step, int(args.log_interval_steps))
    next_eval_step = _next_interval_step(sampling_step, int(args.eval_interval_steps))
    next_save_step = _next_interval_step(sampling_step, int(args.save_interval_steps))
    next_flowlet_dump_step = _next_interval_step(sampling_step, int(args.flowlet_dump_interval_steps))
    progress_printer = TrainingProgressPrinter(enabled=bool(args.get("progress", True)))

    if sampling_step == 0 and int(args.eval_interval_steps) > 0:
        avg_throughput, avg_drop_rate, avg_e2e_delay, avg_test_cost = eval_performance(
            eval_env=eval_env,
            agent=agent,
            sampling_step=sampling_step,
            rollout=iteration,
            experiment=experiment,
            eval_seeds=eval_seeds,
            eval_start_time_ms=float(args.eval_start_time_ms),
            duration_seconds=float(args.eval_duration_seconds),
            tf_writer=tf_writer,
            progress_enabled=bool(args.get("progress", True)),
        )
        score = select_model_score(
            selection_metric=selection_metric,
            throughput=avg_throughput,
            drop_rate=avg_drop_rate,
            e2e_delay=avg_e2e_delay,
            cost=avg_test_cost,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_model_save_path = os.path.join(log_dir, "models", "best_model")
            os.makedirs(best_model_save_path, exist_ok=True)
            agent.save_models(model_dir_path=best_model_save_path)
            logging.info(
                "CHECKPOINT best_model=best_model selection=%s score=%.6f",
                selection_metric,
                best_score,
            )

    while sampling_step < max_sampling_steps:
        iteration += 1
        chunk_start_step = sampling_step
        chunk_max_steps = _next_due_step_delta(
            sampling_step,
            max_sampling_steps,
            next_log_step,
            next_eval_step,
            next_save_step,
            next_flowlet_dump_step,
        )
        remaining_steps = max_sampling_steps - sampling_step
        if chunk_max_steps is not None:
            remaining_steps = chunk_max_steps if remaining_steps is None else min(remaining_steps, chunk_max_steps)

        def progress_callback(vector_steps: int) -> None:
            current_ms = previous_elapsed_ms + max(float(env.current_time - env.start_time), 0.0)
            progress_printer.update_train(
                global_step=chunk_start_step + vector_steps * num_envs,
                max_steps=max_sampling_steps,
                sim_seconds=current_ms / 1000.0,
            )

        train_result = run_marl_steps(
            env=env,
            agent=agent,
            train=True,
            max_steps=remaining_steps,
            progress_callback=progress_callback,
        )
        rollout_steps = int(train_result.step_stats.get("steps", 0))
        if rollout_steps <= 0:
            progress_printer.clear()
            break
        sampling_step += rollout_steps
        simulated_time_ms = float(env.current_time)
        window_duration_ms = float(train_result.step_stats.get("duration_ms", 0.0))
        cumulative_elapsed_ms = previous_elapsed_ms + max(float(env.current_time - env.start_time), 0.0)

        metrics = train_result.metrics
        metric_elapsed_ms = _metric_elapsed_ms(env)
        window_metrics = _diff_metrics(
            current=metrics,
            previous=previous_metrics,
            previous_elapsed_ms=previous_metric_elapsed_ms,
            current_elapsed_ms=metric_elapsed_ms,
        )
        window_agent_stats = _diff_train_stats(train_result.agent_stats, previous_agent_stats)
        cumulative_metrics = metrics
        cumulative_record = {
            "windows": iteration,
            "elapsed_ms": cumulative_elapsed_ms,
            "elapsed_seconds": cumulative_elapsed_ms / 1000.0,
            "metrics": cumulative_metrics.to_dict(),
        }
        step_stats = train_result.step_stats
        vector_steps = int(step_stats.get("vector_steps", rollout_steps))
        env_steps_per_second = rollout_steps / max(float(train_result.elapsed_seconds), 1e-12)
        decisions_per_step = float(step_stats.get("decisions", 0)) / max(rollout_steps, 1)
        window_duration_s = window_duration_ms / 1000.0
        progress_printer.clear()
        logging.info(
            "TRAIN %s",
            _format_train_log(
                step=sampling_step,
                iteration=iteration,
                env_steps=rollout_steps,
                vector_steps=vector_steps,
                num_envs=num_envs,
                sim_elapsed_s=cumulative_elapsed_ms / 1000.0,
                max_sampling_steps=max_sampling_steps,
                window_duration_s=window_duration_s,
                wall_seconds=train_result.elapsed_seconds,
                sim_speed=float(step_stats.get("sim_speed", 0.0)),
                env_steps_per_second=env_steps_per_second,
                decisions_per_step=decisions_per_step,
                recent=window_metrics,
                total=cumulative_metrics,
                agent=agent,
                train_stats=train_result.agent_stats,
                recent_train_stats=window_agent_stats,
            ),
        )
        previous_metrics = metrics
        previous_metric_elapsed_ms = metric_elapsed_ms
        previous_agent_stats = train_result.agent_stats

        train_result.metrics = window_metrics
        train_record = rollout_record(
            sampling_step=sampling_step,
            rollout=iteration,
            phase="train",
            result=train_result,
            simulated_time_ms=simulated_time_ms,
            cumulative=cumulative_record,
        )
        train_record["optimizer_update_steps"] = _agent_optimizer_update_steps(agent, train_result.agent_stats)
        train_record["window_optimizer_update_steps"] = int(window_agent_stats.get("optimizer_update_steps", 0))
        train_record["updates_per_env_step"] = train_record["window_optimizer_update_steps"] / max(rollout_steps, 1)
        experiment.append_jsonl("metrics/train_rollouts.jsonl", train_record)
        experiment.append_csv("metrics/train_rollouts.csv", train_record)

        if int(args.flowlet_dump_interval_steps) > 0 and (
            _interval_due(sampling_step, next_flowlet_dump_step)
            or sampling_step >= max_sampling_steps
        ):
            flowlet_csv_path = os.path.join(log_dir, f"flowlets/flowlets_step_{sampling_step}.csv")
            os.makedirs(os.path.dirname(flowlet_csv_path), exist_ok=True)
            env.save_flowlets_to_csv(flowlet_csv_path)
            logging.info("Flowlets saved to %s", flowlet_csv_path)
            while _interval_due(sampling_step, next_flowlet_dump_step):
                next_flowlet_dump_step = _advance_interval_step(
                    next_flowlet_dump_step,
                    int(args.flowlet_dump_interval_steps),
                )

        tf_writer.add_scalar("train/throughput", window_metrics.throughput, global_step=sampling_step)
        tf_writer.add_scalar("train/delivery_rate", window_metrics.delivery_rate, global_step=sampling_step)
        tf_writer.add_scalar("train/pending_rate", window_metrics.pending_rate, global_step=sampling_step)
        tf_writer.add_scalar("train/drop_rate", window_metrics.drop_rate, global_step=sampling_step)
        tf_writer.add_scalar("train/e2e_delay_mean", window_metrics.e2e_delay_mean, global_step=sampling_step)
        tf_writer.add_scalar("train/queue_delay_mean", window_metrics.queue_delay_mean, global_step=sampling_step)
        tf_writer.add_scalar(
            "train/propagation_delay_mean",
            window_metrics.propagation_delay_mean,
            global_step=sampling_step,
        )
        tf_writer.add_scalar(
            "train/transmission_delay_mean",
            window_metrics.transmission_delay_mean,
            global_step=sampling_step,
        )
        tf_writer.add_scalar("train/cost_mean", window_metrics.cost_mean, global_step=sampling_step)
        tf_writer.add_scalar("train/sim_speed", train_result.step_stats.get("sim_speed", 0.0), global_step=sampling_step)
        tf_writer.add_scalar("train/env_steps_per_wall_second", env_steps_per_second, global_step=sampling_step)
        tf_writer.add_scalar("train/decisions_per_env_step", decisions_per_step, global_step=sampling_step)
        tf_writer.add_scalar(
            "train/optimizer_update_steps",
            train_record["optimizer_update_steps"],
            global_step=sampling_step,
        )
        tf_writer.add_scalar(
            "train/window_optimizer_update_steps",
            train_record["window_optimizer_update_steps"],
            global_step=sampling_step,
        )
        tf_writer.add_scalar("train/updates_per_env_step", train_record["updates_per_env_step"], global_step=sampling_step)
        for stat_key in (
            "utd",
            "effective_utd",
            "update_credit",
            "transitions_added_total",
            "transitions_since_update_credit",
        ):
            if stat_key in train_result.agent_stats and isinstance(train_result.agent_stats[stat_key], (int, float)):
                tf_writer.add_scalar(f"train/{stat_key}", train_result.agent_stats[stat_key], global_step=sampling_step)
        tf_writer.add_scalar("train/generated_packets_per_env_step", window_metrics.generated / max(rollout_steps, 1), global_step=sampling_step)
        tf_writer.add_scalar("train/delivered_packets_per_env_step", window_metrics.delivered / max(rollout_steps, 1), global_step=sampling_step)
        tf_writer.add_scalar("train/cumulative_simulated_seconds", cumulative_elapsed_ms / 1000.0, global_step=sampling_step)
        tf_writer.add_scalar("train_cumulative/throughput", cumulative_metrics.throughput, global_step=sampling_step)
        tf_writer.add_scalar("train_cumulative/delivery_rate", cumulative_metrics.delivery_rate, global_step=sampling_step)
        tf_writer.add_scalar("train_cumulative/pending_rate", cumulative_metrics.pending_rate, global_step=sampling_step)
        tf_writer.add_scalar("train_cumulative/drop_rate", cumulative_metrics.drop_rate, global_step=sampling_step)
        tf_writer.add_scalar("train_cumulative/e2e_delay_mean", cumulative_metrics.e2e_delay_mean, global_step=sampling_step)
        tf_writer.add_scalar(
            "train_cumulative/queue_delay_mean",
            cumulative_metrics.queue_delay_mean,
            global_step=sampling_step,
        )
        tf_writer.add_scalar(
            "train_cumulative/propagation_delay_mean",
            cumulative_metrics.propagation_delay_mean,
            global_step=sampling_step,
        )
        tf_writer.add_scalar(
            "train_cumulative/transmission_delay_mean",
            cumulative_metrics.transmission_delay_mean,
            global_step=sampling_step,
        )
        tf_writer.add_scalar("train_cumulative/cost_mean", cumulative_metrics.cost_mean, global_step=sampling_step)

        if not isinstance(tf_writer, NullMetricWriter):
            flowlet_df = env.get_flowlet_dataframe()
            delivered_flowlets = flowlet_df[flowlet_df["delivered"]] if not flowlet_df.empty else flowlet_df
        else:
            delivered_flowlets = None
        if delivered_flowlets is not None and not delivered_flowlets.empty:
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
                "rollout": iteration,
                "simulated_time_ms": simulated_time_ms,
                "last_model": str(Path(last_model_save_path).relative_to(log_dir)),
                "selection_metric": selection_metric,
                "best_score": best_score,
                "max_sampling_steps": max_sampling_steps,
                "optimizer_update_steps": _agent_optimizer_update_steps(agent, train_result.agent_stats),
                "eval_duration_seconds": float(args.eval_duration_seconds),
                "horizon": "continuing_env",
                "num_envs": num_envs,
                "continuing_metrics_state": {
                    "windows": iteration,
                    "elapsed_ms": cumulative_elapsed_ms,
                },
                "continuing_metrics": cumulative_record,
            },
        )

        if (
            _interval_due(sampling_step, next_save_step)
            or sampling_step >= max_sampling_steps
        ):
            step_model_save_path = os.path.join(model_dir_path, f"model_step_{sampling_step}")
            os.makedirs(step_model_save_path, exist_ok=True)
            agent.save_models(model_dir_path=step_model_save_path)
            while _interval_due(sampling_step, next_save_step):
                next_save_step = _advance_interval_step(next_save_step, int(args.save_interval_steps))

        should_eval = int(args.eval_interval_steps) > 0 and sampling_step >= int(args.eval_after_steps)
        should_eval = should_eval and (
            _interval_due(sampling_step, next_eval_step)
            or sampling_step >= max_sampling_steps
        )
        if should_eval:
            avg_throughput, avg_drop_rate, avg_e2e_delay, avg_test_cost = eval_performance(
                eval_env=eval_env,
                agent=agent,
                sampling_step=sampling_step,
                rollout=iteration,
                experiment=experiment,
                eval_seeds=eval_seeds,
                eval_start_time_ms=float(args.eval_start_time_ms),
                duration_seconds=float(args.eval_duration_seconds),
                tf_writer=tf_writer,
                progress_enabled=bool(args.get("progress", True)),
            )
            while _interval_due(sampling_step, next_eval_step):
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
                    "CHECKPOINT best_model=best_model selection=%s score=%.6f",
                    selection_metric,
                    best_score,
                )

        experiment.write_json(
            "summary.json",
            {
                "sampling_step": sampling_step,
                "step": sampling_step,
                "env_interaction_steps": sampling_step,
                "optimizer_update_steps": _agent_optimizer_update_steps(agent, train_result.agent_stats),
                "eval_duration_seconds": float(args.eval_duration_seconds),
                "max_sampling_steps": max_sampling_steps,
                "rollout": iteration,
                "simulated_time_ms": simulated_time_ms,
                "cumulative_simulated_seconds": cumulative_elapsed_ms / 1000.0,
                "horizon": "continuing_env",
                "num_envs": num_envs,
                "best_score": best_score,
                "selection_metric": selection_metric,
                "last_train_metrics": window_metrics.to_dict(),
                "cumulative_train": cumulative_record,
                "last_train_agent_stats": train_result.agent_stats,
            },
        )

        while _interval_due(sampling_step, next_log_step):
            next_log_step = _advance_interval_step(next_log_step, int(args.log_interval_steps))


def _aggregate_eval_metrics(rollout, metrics: list[Metrics]) -> EvaluationResult:
    throughputs = np.array([item.throughput for item in metrics], dtype=np.float64)
    delivery_rates = np.array([item.delivery_rate for item in metrics], dtype=np.float64)
    pending_rates = np.array([item.pending_rate for item in metrics], dtype=np.float64)
    drop_rates = np.array([item.drop_rate for item in metrics], dtype=np.float64)
    e2e_delays = np.array([item.e2e_delay_mean for item in metrics], dtype=np.float64)
    costs = np.array([item.cost_mean for item in metrics], dtype=np.float64)
    return EvaluationResult(
        rollouts=[rollout],
        throughput_mean=float(throughputs.mean()) if len(throughputs) else 0.0,
        throughput_std=float(throughputs.std()) if len(throughputs) else 0.0,
        drop_rate_mean=float(drop_rates.mean()) if len(drop_rates) else 0.0,
        drop_rate_std=float(drop_rates.std()) if len(drop_rates) else 0.0,
        e2e_delay_mean=float(e2e_delays.mean()) if len(e2e_delays) else 0.0,
        e2e_delay_std=float(e2e_delays.std()) if len(e2e_delays) else 0.0,
        cost_mean=float(costs.mean()) if len(costs) else 0.0,
        delivery_rate_mean=float(delivery_rates.mean()) if len(delivery_rates) else 0.0,
        delivery_rate_std=float(delivery_rates.std()) if len(delivery_rates) else 0.0,
        pending_rate_mean=float(pending_rates.mean()) if len(pending_rates) else 0.0,
        pending_rate_std=float(pending_rates.std()) if len(pending_rates) else 0.0,
    )


def _eval_seed_metrics(eval_env: RoutingEnv, fallback: Metrics) -> list[Metrics]:
    envs = getattr(eval_env, "envs", None)
    if not envs:
        return [fallback]
    return [env.calc_metrics() for env in envs]


def _base_env(env: RoutingEnv):
    envs = getattr(env, "envs", None)
    return envs[0] if envs else env


def _metric_elapsed_ms(env: RoutingEnv) -> float:
    envs = getattr(env, "envs", None)
    if envs:
        return sum(max(float(item.current_time - item.start_time), 0.0) for item in envs)
    return max(float(env.current_time - env.start_time), 0.0)


def _diff_metrics(
    current: Metrics,
    previous: Metrics,
    previous_elapsed_ms: float,
    current_elapsed_ms: float,
) -> Metrics:
    duration_s = max((float(current_elapsed_ms) - float(previous_elapsed_ms)) / 1000.0, 1e-12)
    current_elapsed_s = max(float(current_elapsed_ms) / 1000.0, 0.0)
    previous_elapsed_s = max(float(previous_elapsed_ms) / 1000.0, 0.0)

    generated = _count_delta(current.generated, previous.generated)
    delivered = _count_delta(current.delivered, previous.delivered)
    dropped = _count_delta(current.dropped, previous.dropped)
    generated_normal = _count_delta(current.generated_normal_packet, previous.generated_normal_packet)
    generated_small = _count_delta(current.generated_small_packet, previous.generated_small_packet)
    delivered_normal = _count_delta(current.delivered_normal_packet, previous.delivered_normal_packet)
    delivered_small = _count_delta(current.delivered_small_packet, previous.delivered_small_packet)
    dropped_normal = _count_delta(current.dropped_normal_packet, previous.dropped_normal_packet)
    dropped_small = _count_delta(current.dropped_small_packet, previous.dropped_small_packet)
    dropped_by_ttl = _count_delta(current.dropped_by_ttl, previous.dropped_by_ttl)
    pending = max(generated - delivered - dropped, 0)
    pending_normal = max(generated_normal - delivered_normal - dropped_normal, 0)
    pending_small = max(generated_small - delivered_small - dropped_small, 0)

    delivered_mbit = max(
        current.throughput * current_elapsed_s - previous.throughput * previous_elapsed_s,
        0.0,
    )
    return Metrics(
        generated=generated,
        generated_normal_packet=generated_normal,
        generated_small_packet=generated_small,
        delivered=delivered,
        delivered_normal_packet=delivered_normal,
        delivered_small_packet=delivered_small,
        dropped=dropped,
        dropped_by_ttl=dropped_by_ttl,
        dropped_normal_packet=dropped_normal,
        dropped_small_packet=dropped_small,
        pending=pending,
        pending_normal_packet=pending_normal,
        pending_small_packet=pending_small,
        throughput=delivered_mbit / duration_s,
        service_rate=delivered / duration_s,
        delivery_rate=delivered / generated if generated else 0.0,
        drop_rate=dropped / generated if generated else 0.0,
        pending_rate=pending / generated if generated else 0.0,
        normal_packet_delivery_rate=delivered_normal / generated_normal if generated_normal else 0.0,
        normal_packet_drop_rate=dropped_normal / generated_normal if generated_normal else 0.0,
        normal_packet_pending_rate=pending_normal / generated_normal if generated_normal else 0.0,
        small_packet_delivery_rate=delivered_small / generated_small if generated_small else 0.0,
        small_packet_drop_rate=dropped_small / generated_small if generated_small else 0.0,
        small_packet_pending_rate=pending_small / generated_small if generated_small else 0.0,
        e2e_delay_mean=_mean_delta(
            current.e2e_delay_mean,
            current.delivered,
            previous.e2e_delay_mean,
            previous.delivered,
            delivered,
        ),
        queue_delay_mean=_mean_delta(
            current.queue_delay_mean,
            current.delivered,
            previous.queue_delay_mean,
            previous.delivered,
            delivered,
        ),
        transmission_delay_mean=_mean_delta(
            current.transmission_delay_mean,
            current.delivered,
            previous.transmission_delay_mean,
            previous.delivered,
            delivered,
        ),
        propagation_delay_mean=_mean_delta(
            current.propagation_delay_mean,
            current.delivered,
            previous.propagation_delay_mean,
            previous.delivered,
            delivered,
        ),
        normal_packet_e2e_delay_mean=_mean_delta(
            current.normal_packet_e2e_delay_mean,
            current.delivered_normal_packet,
            previous.normal_packet_e2e_delay_mean,
            previous.delivered_normal_packet,
            delivered_normal,
        ),
        normal_packet_queue_delay_mean=_mean_delta(
            current.normal_packet_queue_delay_mean,
            current.delivered_normal_packet,
            previous.normal_packet_queue_delay_mean,
            previous.delivered_normal_packet,
            delivered_normal,
        ),
        normal_packet_transmission_delay_mean=_mean_delta(
            current.normal_packet_transmission_delay_mean,
            current.delivered_normal_packet,
            previous.normal_packet_transmission_delay_mean,
            previous.delivered_normal_packet,
            delivered_normal,
        ),
        normal_packet_propagation_delay_mean=_mean_delta(
            current.normal_packet_propagation_delay_mean,
            current.delivered_normal_packet,
            previous.normal_packet_propagation_delay_mean,
            previous.delivered_normal_packet,
            delivered_normal,
        ),
        small_packet_e2e_delay_mean=_mean_delta(
            current.small_packet_e2e_delay_mean,
            current.delivered_small_packet,
            previous.small_packet_e2e_delay_mean,
            previous.delivered_small_packet,
            delivered_small,
        ),
        small_packet_queue_delay_mean=_mean_delta(
            current.small_packet_queue_delay_mean,
            current.delivered_small_packet,
            previous.small_packet_queue_delay_mean,
            previous.delivered_small_packet,
            delivered_small,
        ),
        small_packet_transmission_delay_mean=_mean_delta(
            current.small_packet_transmission_delay_mean,
            current.delivered_small_packet,
            previous.small_packet_transmission_delay_mean,
            previous.delivered_small_packet,
            delivered_small,
        ),
        small_packet_propagation_delay_mean=_mean_delta(
            current.small_packet_propagation_delay_mean,
            current.delivered_small_packet,
            previous.small_packet_propagation_delay_mean,
            previous.delivered_small_packet,
            delivered_small,
        ),
        cost_mean=_mean_delta(current.cost_mean, current.delivered, previous.cost_mean, previous.delivered, delivered),
        cost_small_packet_mean=_mean_delta(
            current.cost_small_packet_mean,
            current.delivered_small_packet,
            previous.cost_small_packet_mean,
            previous.delivered_small_packet,
            delivered_small,
        ),
        cost_normal_packet_mean=_mean_delta(
            current.cost_normal_packet_mean,
            current.delivered_normal_packet,
            previous.cost_normal_packet_mean,
            previous.delivered_normal_packet,
            delivered_normal,
        ),
    )


def _diff_train_stats(current: dict, previous: dict | None) -> dict:
    previous = previous or {}
    transitions = _count_delta(current.get("transitions", 0), previous.get("transitions", 0))
    optimizer_update_steps = _count_delta(
        current.get("optimizer_update_steps", 0),
        previous.get("optimizer_update_steps", 0),
    )
    stats = {
        "transitions": transitions,
        "optimizer_update_steps": optimizer_update_steps,
        "terminal_transitions": _count_delta(
            current.get("terminal_transitions", 0),
            previous.get("terminal_transitions", 0),
        ),
        "delivered_transitions": _count_delta(
            current.get("delivered_transitions", 0),
            previous.get("delivered_transitions", 0),
        ),
        "dropped_transitions": _count_delta(
            current.get("dropped_transitions", 0),
            previous.get("dropped_transitions", 0),
        ),
        "truncated_transitions": _count_delta(
            current.get("truncated_transitions", 0),
            previous.get("truncated_transitions", 0),
        ),
        "reward_sum": float(current.get("reward_sum", 0.0)) - float(previous.get("reward_sum", 0.0)),
        "cost_sum": float(current.get("cost_sum", 0.0)) - float(previous.get("cost_sum", 0.0)),
    }
    stats["reward_mean"] = stats["reward_sum"] / max(transitions, 1)
    stats["cost_mean"] = stats["cost_sum"] / max(transitions, 1)
    return stats


def _format_train_log(
    step: int,
    iteration: int,
    env_steps: int,
    vector_steps: int,
    num_envs: int,
    sim_elapsed_s: float,
    max_sampling_steps: int,
    window_duration_s: float,
    wall_seconds: float,
    sim_speed: float,
    env_steps_per_second: float,
    decisions_per_step: float,
    recent: Metrics,
    total: Metrics,
    agent: BaseAgent,
    train_stats: dict,
    recent_train_stats: dict,
) -> str:
    ttl_recent = recent.dropped_by_ttl / recent.generated if recent.generated else 0.0
    ttl_total = total.dropped_by_ttl / total.generated if total.generated else 0.0
    payload = {
        "global_step": step,
        "iter": iteration,
        "time/envs": num_envs,
        "time/env_steps": env_steps,
        "time/vector_steps": vector_steps,
        "time/max_steps": max_sampling_steps,
        "time/sim": _format_seconds(sim_elapsed_s),
        "time/window": _format_seconds(window_duration_s),
        "time/wall": _format_seconds(wall_seconds),
        "time/env_steps/s": env_steps_per_second,
        "time/sim_speed": sim_speed,
        "rollout/decisions_per_step": decisions_per_step,
        "rollout/recent_generated": recent.generated,
        "rollout/total_generated": total.generated,
        "rollout/recent_pending": recent.pending,
        "rollout/total_pending": total.pending,
        "rollout/recent_pending_rate": recent.pending_rate,
        "rollout/total_pending_rate": total.pending_rate,
        "rollout/recent_drop_rate": recent.drop_rate,
        "rollout/total_drop_rate": total.drop_rate,
        "rollout/recent_ttl_drop_rate": ttl_recent,
        "rollout/total_ttl_drop_rate": ttl_total,
        "rollout/recent_delivery_rate": recent.delivery_rate,
        "rollout/total_delivery_rate": total.delivery_rate,
        "rollout/recent_delay_ms": recent.e2e_delay_mean,
        "rollout/total_delay_ms": total.e2e_delay_mean,
        "rollout/recent_queue_ms": recent.queue_delay_mean,
        "rollout/total_queue_ms": total.queue_delay_mean,
        "rollout/recent_prop_ms": recent.propagation_delay_mean,
        "rollout/total_prop_ms": total.propagation_delay_mean,
        "rollout/recent_tx_ms": recent.transmission_delay_mean,
        "rollout/total_tx_ms": total.transmission_delay_mean,
        "rollout/recent_cost": recent.cost_mean,
        "rollout/total_cost": total.cost_mean,
        "rollout/recent_throughput": _format_throughput(recent.throughput),
        "rollout/total_throughput": _format_throughput(total.throughput),
    }
    payload.update(_rl_log_fields(agent, train_stats, recent_train_stats))
    return _format_kv(payload)


def _format_eval_summary_log(
    step: int,
    rollout: int,
    num_rollouts: int,
    num_envs: int,
    num_eval_seeds: int,
    duration_seconds: float,
    wall_seconds: float,
    result: EvaluationResult,
    ttl_drop_rates: np.ndarray,
    queue_delays: np.ndarray,
    propagation_delays: np.ndarray,
    transmission_delays: np.ndarray,
) -> str:
    env_steps = sum(int(item.step_stats.get("steps", 0)) for item in result.rollouts)
    vector_steps = sum(int(item.step_stats.get("vector_steps", 0)) for item in result.rollouts)
    decisions = sum(int(item.step_stats.get("decisions", 0)) for item in result.rollouts)
    aggregate_duration_ms = sum(float(item.step_stats.get("aggregate_duration_ms", 0.0)) for item in result.rollouts)
    generated = sum(int(item.metrics.generated) for item in result.rollouts)
    pending = sum(int(item.metrics.pending) for item in result.rollouts)
    sim_speed = (aggregate_duration_ms / 1000.0) / max(float(wall_seconds), 1e-12)
    env_steps_per_second = env_steps / max(float(wall_seconds), 1e-12)
    decisions_per_step = decisions / max(env_steps, 1)
    return _format_kv(
        {
            "global_step": step,
            "iter": rollout,
            "eval/rollouts": num_rollouts,
            "eval/seeds": num_eval_seeds,
            "eval/vector_envs": num_envs,
            "eval/total_envs": num_envs,
            "time/envs": num_envs,
            "time/env_steps": env_steps,
            "time/vector_steps": vector_steps,
            "time/sim": _format_seconds(duration_seconds),
            "time/wall": _format_seconds(wall_seconds),
            "time/env_steps/s": env_steps_per_second,
            "time/sim_speed": sim_speed,
            "eval/decisions_per_step": decisions_per_step,
            "eval/generated": generated,
            "eval/pending": pending,
            "eval/delivery_rate": result.delivery_rate_mean,
            "eval/delivery_rate_std": result.delivery_rate_std,
            "eval/pending_rate": result.pending_rate_mean,
            "eval/pending_rate_std": result.pending_rate_std,
            "eval/drop_rate": result.drop_rate_mean,
            "eval/drop_rate_std": result.drop_rate_std,
            "eval/ttl_drop_rate": _array_mean(ttl_drop_rates),
            "eval/ttl_drop_rate_std": _array_std(ttl_drop_rates),
            "eval/delay_ms": result.e2e_delay_mean,
            "eval/delay_ms_std": result.e2e_delay_std,
            "eval/queue_ms": _array_mean(queue_delays),
            "eval/queue_ms_std": _array_std(queue_delays),
            "eval/prop_ms": _array_mean(propagation_delays),
            "eval/prop_ms_std": _array_std(propagation_delays),
            "eval/tx_ms": _array_mean(transmission_delays),
            "eval/tx_ms_std": _array_std(transmission_delays),
            "eval/cost": result.cost_mean,
            "eval/throughput": _format_throughput(result.throughput_mean),
            "eval/throughput_std": _format_throughput(result.throughput_std),
        }
    )


def _rl_log_fields(agent: BaseAgent, current_stats: dict, recent_stats: dict) -> dict[str, object]:
    global_agent = getattr(agent, "global_agent", None)
    fields: dict[str, object] = {
        "train/reward_mean": float(recent_stats.get("reward_mean", 0.0)),
        "train/cost_mean": float(recent_stats.get("cost_mean", 0.0)),
        "train/transitions": int(recent_stats.get("transitions", 0)),
        "train/window_updates": int(recent_stats.get("optimizer_update_steps", 0)),
    }
    epsilon = getattr(global_agent, "epsilon_train", None)
    if epsilon is not None:
        fields["train/epsilon"] = float(epsilon)
    alpha = _call_scalar(global_agent, "alpha")
    if alpha is not None:
        fields["train/alpha"] = alpha
    lambdar = _call_scalar(global_agent, "lambdar")
    if lambdar is not None:
        fields["train/lambda"] = lambdar

    replay_size = current_stats.get("replay", {}).get("replay_size")
    replay_buffer = getattr(global_agent, "replay_buffer", None)
    if replay_size is None and replay_buffer is not None:
        replay_size = len(replay_buffer)
    if replay_size is not None:
        fields["train/replay_size"] = int(replay_size)

    optimizer_update_steps = _agent_optimizer_update_steps(agent, current_stats)
    fields["train/updates"] = optimizer_update_steps
    for key in (
        "utd",
        "effective_utd",
        "update_credit",
        "transitions_added_total",
        "transitions_since_update_credit",
    ):
        if key in current_stats:
            fields[f"train/{key}"] = current_stats[key]
    if "pending_transitions" in current_stats:
        fields["train/pending"] = int(current_stats.get("pending_transitions", 0))
    return fields


def _agent_optimizer_update_steps(agent: BaseAgent, stats: dict | None = None) -> int:
    stats = stats or {}
    if "optimizer_update_steps" in stats:
        return int(stats.get("optimizer_update_steps") or 0)
    global_agent = getattr(agent, "global_agent", None)
    training_steps = getattr(global_agent, "training_steps", None)
    return int(training_steps or 0)


def _format_kv(payload: dict[str, object]) -> str:
    return " ".join(f"{key}={_format_log_value(key, value)}" for key, value in payload.items())


def _format_log_value(key: str, value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return _compact_count(value) if abs(int(value)) >= 10_000 else str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if "rate" in key:
            return _pct(value)
        if key.endswith("/s"):
            return f"{value:.1f}"
        if "speed" in key:
            return f"{value:.2f}x"
        if "reward" in key or key.endswith("/cost") or "lambda" in key or "alpha" in key or "epsilon" in key:
            return f"{value:.4f}"
        if (
            "delay_ms" in key
            or "queue_ms" in key
            or "prop_ms" in key
            or "tx_ms" in key
            or "decisions_per_step" in key
        ):
            return f"{value:.2f}"
        return f"{value:.3f}"
    return str(value)


def _format_metrics(metrics: Metrics) -> str:
    ttl_rate = metrics.dropped_by_ttl / metrics.generated if metrics.generated else 0.0
    return (
        f"pkts={_compact_count(metrics.generated)} delivered={_pct(metrics.delivery_rate)} "
        f"pending={_pct(metrics.pending_rate)} "
        f"drop={_pct(metrics.drop_rate)}(ttl={_pct(ttl_rate)}) "
        f"delay={metrics.e2e_delay_mean:.1f}ms[q={metrics.queue_delay_mean:.1f},"
        f"prop={metrics.propagation_delay_mean:.1f},tx={metrics.transmission_delay_mean:.1f}] "
        f"cost={metrics.cost_mean:.2f} throughput={_format_throughput(metrics.throughput)} "
        f"service={_compact_count(metrics.service_rate)}/s"
    )


def _format_rl_state(agent: BaseAgent, current_stats: dict, window_stats: dict) -> str:
    global_agent = getattr(agent, "global_agent", None)
    parts = []
    epsilon = getattr(global_agent, "epsilon_train", None)
    if epsilon is not None:
        parts.append(f"eps={float(epsilon):.4f}")
    alpha = _call_scalar(global_agent, "alpha")
    if alpha is not None:
        parts.append(f"alpha={alpha:.4f}")
    lambdar = _call_scalar(global_agent, "lambdar")
    if lambdar is not None:
        parts.append(f"lambda={lambdar:.4f}")

    replay_size = current_stats.get("replay", {}).get("replay_size")
    replay_buffer = getattr(global_agent, "replay_buffer", None)
    if replay_size is None and replay_buffer is not None:
        replay_size = len(replay_buffer)
    if replay_size is not None:
        parts.append(f"replay={_compact_count(replay_size)}")

    training_steps = getattr(global_agent, "training_steps", None)
    if training_steps is not None:
        parts.append(f"updates={_compact_count(training_steps)}")
    parts.append(f"trans={_compact_count(window_stats.get('transitions', 0))}")
    parts.append(f"r={float(window_stats.get('reward_mean', 0.0)):.3f}")
    if "cost_mean" in window_stats:
        parts.append(f"c={float(window_stats.get('cost_mean', 0.0)):.4f}")
    if "pending_transitions" in current_stats:
        parts.append(f"pending={_compact_count(current_stats.get('pending_transitions', 0))}")
    return " ".join(parts) if parts else "n/a"


def _call_scalar(obj, method_name: str) -> float | None:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    value = method()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _array_mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _array_std(values: np.ndarray) -> float:
    return float(values.std()) if len(values) else 0.0


def _count_delta(current, previous) -> int:
    return max(int(current) - int(previous), 0)


def _mean_delta(current_mean: float, current_count: int, previous_mean: float, previous_count: int, delta_count: int) -> float:
    if delta_count <= 0:
        return 0.0
    total = float(current_mean) * int(current_count) - float(previous_mean) * int(previous_count)
    return float(total) / float(delta_count)


def _compact_count(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    for suffix, scale in (("B", 1_000_000_000.0), ("M", 1_000_000.0), ("K", 1_000.0)):
        if value >= scale:
            return f"{sign}{value / scale:.1f}{suffix}"
    return f"{sign}{value:.0f}"


def _format_throughput(mbps: float) -> str:
    mbps = float(mbps)
    if abs(mbps) >= 1000.0:
        return f"{mbps / 1000.0:.2f}Gbps"
    return f"{mbps:.1f}Mbps"


def _pct(rate: float) -> str:
    return f"{float(rate) * 100.0:.2f}%"


def _format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if abs(seconds) < 1.0:
        return f"{seconds:.3f}s"
    if abs(seconds) < 100.0:
        return f"{seconds:.1f}s"
    return f"{seconds:.0f}s"


def _eta_seconds(wall_seconds: float, percent: float) -> float | None:
    if percent <= 1e-9:
        return None
    progress = percent / 100.0
    return max(float(wall_seconds) * (1.0 - progress) / progress, 0.0)


def _format_eta(seconds: float | None) -> str:
    return "n/a" if seconds is None else _format_seconds(seconds)


def _format_optional(value) -> str:
    if value is None:
        return "none"
    return str(value)


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


def _next_due_step_delta(current_step: int, *due_steps: int | None) -> int | None:
    candidates = [int(step) for step in due_steps if step is not None and int(step) > current_step]
    if not candidates:
        return None
    return max(1, min(candidates) - current_step)


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
        "max_sampling_steps": 1_000_000,
        "num_envs": 1,
        "vector_utc_span_seconds": 5400.0,
        "vector_seed_stride": 100000,
        "start_time_ms": 0.0,
        "seed": 33333,
        "run_id": None,
        "eval_duration_seconds": 60.0,
        "log_interval_steps": 10000,
        "eval_interval_steps": 60000,
        "eval_after_steps": 60000,
        "eval_start_time_ms": 0.0,
        "eval_seeds": "6666,7777,8888",
        "save_interval_steps": 60000,
        "flowlet_dump_interval_steps": 0,
        "selection_metric": "risk_adjusted",
        "env_verbose": False,
        "progress": True,
        "notes": "",
        "archive_source": True,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_MAIN_CONFIG)
    parser.add_argument("--recover_runid", type=str, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--agent", type=str, default=None)
    parser.add_argument("--max_sampling_steps", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--vector_utc_span_seconds", type=float, default=None)
    parser.add_argument("--vector_seed_stride", type=int, default=None)
    parser.add_argument("--start_time_ms", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--runs_dir", type=str, default=None)
    parser.add_argument("--eval_duration_seconds", type=float, default=None)
    parser.add_argument("--log_interval_steps", type=int, default=None)
    parser.add_argument("--eval_interval_steps", type=int, default=None)
    parser.add_argument("--eval_after_steps", type=int, default=None)
    parser.add_argument("--eval_start_time_ms", type=float, default=None)
    parser.add_argument("--eval_seeds", type=str, default=None)
    parser.add_argument("--save_interval_steps", type=int, default=None)
    parser.add_argument("--flowlet_dump_interval_steps", type=int, default=None)
    parser.add_argument("--env_verbose", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
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
        setup_logging(log_dir)

        logging.info("Recovering from %s", run_id)

        saved_args = NamedDict.load(f"{log_dir}/args.json")
        for key, value in saved_args.items():
            if key not in {
                "config",
                "recover_runid",
                "duration_seconds",
                "max_sampling_steps",
                "eval_duration_seconds",
                "log_interval_steps",
                "eval_interval_steps",
                "eval_after_steps",
                "eval_seeds",
                "save_interval_steps",
                "flowlet_dump_interval_steps",
                "selection_metric",
                "progress",
                "notes",
                "archive_source",
            }:
                args[key] = value
        set_seeds(args.seed)

        tf_writer = NullMetricWriter()
        experiment = ExperimentLogger(log_dir)

        env_config = NamedDict.load(f"{log_dir}/env_config.json")
        agent_config = NamedDict.load(f"{log_dir}/agent_config.json")

        eval_seeds = parse_seed_list(args.eval_seeds)
        env = create_training_env(env_config, args, tf_writer=tf_writer)
        eval_env = create_evaluation_env(env_config, eval_seeds, tf_writer=tf_writer)
        log_env_split(env, eval_env, eval_seeds)
        agent = create_agent(
            agent_config,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            tf_writer=tf_writer,
        )
        log_run_setup(
            run_id=run_id,
            log_dir=log_dir,
            args=args,
            main_config=main_config,
            env_config=env_config,
            agent_config=agent_config,
            env=env,
            recovered=True,
        )
        logging.info("CONFIG loaded args.json env_config.json agent_config.json")
        agent.load_models(f"{log_dir}/models/last_model")
        checkpoint = NamedDict.load(f"{log_dir}/checkpoint_state.json")
        start_sampling_step = int(checkpoint.get("sampling_step", 0))
        start_rollout = int(checkpoint.get("rollout", 0))
        start_simulated_time_ms = float(checkpoint.get("simulated_time_ms", args.start_time_ms))
        start_best_score = checkpoint.get("best_score", None)
        start_continuing_state = checkpoint.get("continuing_metrics_state", None)
        logging.info(
            "RESUME step=%d iteration=%d sim_time_ms=%.3f",
            start_sampling_step,
            start_rollout,
            start_simulated_time_ms,
        )
    else:
        env_config = load_env_config(main_config, override_path=args.env)
        agent_config = load_agent_config(main_config, override_path=args.agent)

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = args.run_id or f"{agent_config.name}_{date_str}"

        log_dir = os.path.join(runs_dir, run_id)
        os.makedirs(log_dir, exist_ok=True)
        setup_logging(log_dir)
        experiment = ExperimentLogger(log_dir)

        tf_writer = NullMetricWriter()

        set_seeds(args.seed)

        eval_seeds = parse_seed_list(args.eval_seeds)
        env = create_training_env(env_config, args, tf_writer=tf_writer)
        eval_env = create_evaluation_env(env_config, eval_seeds, tf_writer=tf_writer)
        log_env_split(env, eval_env, eval_seeds)
        agent = create_agent(
            agent_config,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            tf_writer=tf_writer,
        )
        log_run_setup(
            run_id=run_id,
            log_dir=log_dir,
            args=args,
            main_config=main_config,
            env_config=env_config,
            agent_config=agent_config,
            env=env,
            recovered=False,
        )

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
        logging.info("CONFIG saved args.json main_config.json env_config.json agent_config.json manifest.json")

        if args.archive_source:
            archive_source_code(log_dir)

        start_sampling_step = 0
        start_rollout = 0
        start_simulated_time_ms = float(args.start_time_ms)
        start_best_score = None
        start_continuing_state = None

    train(
        env=env,
        eval_env=eval_env,
        agent=agent,
        start_sampling_step=start_sampling_step,
        start_rollout=start_rollout,
        start_simulated_time_ms=start_simulated_time_ms,
        start_best_score=start_best_score,
        start_continuing_state=start_continuing_state,
        log_dir=log_dir,
        tf_writer=tf_writer,
        args=args,
        experiment=experiment,
    )

    tf_writer.close()


if __name__ == "__main__":
    main()

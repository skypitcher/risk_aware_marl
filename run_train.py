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

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from sat_net.routing_env import RoutingEnvAsync
from sat_net.solver import BaseSolver, create_solver
from sat_net.util import NamedDict, ms2str

PROJECT_ROOT = str(os.path.dirname(os.path.abspath(__file__)))
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)


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


def set_seeds(seed):
    """
    Sets random seeds for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def eval_performace(env: RoutingEnvAsync, solver: BaseSolver):
    """
    Evaluates the solver's performance over a fixed set of seeds.
    """
    eval_seed_list = [6666, 7777, 8888]
    solver.set_eval()

    throughput_list = []
    drop_rate_list = []
    e2e_delay_list = []
    cost_list = []
    start_time = time.time()
    for i, eval_seed in enumerate(eval_seed_list):
        logging.info("Testing performance for seed: %d, progress=%d/%d", eval_seed, i + 1, len(eval_seed_list))
        env.reset(seed=eval_seed)
        env.run(solver)
        metrics = env.calc_metrics()
        logging.info("Tick: %s | %s", ms2str(env.start_time), metrics.get_summary())
        logging.info("Testing metrics: %s", metrics.to_json())
        throughput_list.append(metrics.throughput)
        drop_rate_list.append(metrics.drop_rate)
        e2e_delay_list.append(metrics.e2e_delay_mean)
        cost_list.append(metrics.queue_delay_mean)

    testing_time = time.time() - start_time

    avg_throughput = np.mean(throughput_list)
    avg_drop_rate = np.mean(drop_rate_list)
    avg_e2e_delay = np.mean(e2e_delay_list)
    std_throughput = np.std(throughput_list)
    std_drop_rate = np.std(drop_rate_list)
    std_e2e_delay = np.std(e2e_delay_list)
    avg_test_cost = np.mean(cost_list)
    logging.info(
        "Testing finished in %.2fs. Avg metrics: throughput=%.2f±%.2f, drop_rate=%.4f±%.4f, e2e_delay=%.2f±%.2f ms, cost=%.2f",
        testing_time,
        avg_throughput,
        std_throughput,
        avg_drop_rate,
        std_drop_rate,
        avg_e2e_delay,
        std_e2e_delay,
        avg_test_cost,
    )

    return avg_throughput, avg_drop_rate, avg_e2e_delay, avg_test_cost


def train(env: RoutingEnvAsync, solver: BaseSolver, start_epoch: int, max_epoch: int, log_dir, tf_writer):
    """
    Main training loop.
    """
    logging.info("Training started")
    logging.info("You can run ``tensorboard --logdir=runs`` to see the training progress.")

    best_throughput = None
    for epoch in range(start_epoch, max_epoch + 1):
        logging.info("Epoch %d/%d", epoch, max_epoch)

        # Training
        start_time = time.time()
        solver.set_train()
        env.reset()
        env.run(solver)
        training_time = time.time() - start_time
        logging.info("Training finished in %.2fs", training_time)

        # calculating metrics
        metrics = env.calc_metrics()
        logging.info("Tick: %s | %s", ms2str(env.start_time), metrics.get_summary())
        logging.info("Training metrics: %s", metrics.to_json())
        solver_stats = solver.get_stats()
        if solver_stats is not None:
            logging.info("Solver stats: %s", solver_stats)

        packet_csv_path = os.path.join(log_dir, f"packets/packets_epoch_{epoch}.csv")
        os.makedirs(os.path.dirname(packet_csv_path), exist_ok=True)
        env.save_packets_to_csv(packet_csv_path)  # Assuming this method exists
        logging.info("Packets saved to %s", packet_csv_path)

        tf_writer.add_scalar("epoch/throughput", metrics.throughput, global_step=epoch)
        tf_writer.add_scalar("epoch/delivery_rate", metrics.delivery_rate, global_step=epoch)
        tf_writer.add_scalar("epoch/drop_rate", metrics.drop_rate, global_step=epoch)

        if len(env.delivered_packets) > 0:
            queue_costs = np.array([p.total_queue_cost for p in env.delivered_packets])
            if len(queue_costs) > 0:
                tf_writer.add_histogram("epoch/queue_costs", queue_costs, global_step=epoch)
                tf_writer.add_scalar("epoch/cost", np.mean(queue_costs), global_step=epoch)
                tf_writer.add_scalar("epoch/cost_std", np.std(queue_costs), global_step=epoch)

            first_gsl_delays = np.array([p.first_gsl_delay for p in env.delivered_packets])
            packet_delays = np.array([p.e2e_delay for p in env.delivered_packets])
            small_packet_delays = np.array([p.e2e_delay for p in env.delivered_packets if not p.is_normal_packet])
            normal_packet_delays = np.array([p.e2e_delay for p in env.delivered_packets if p.is_normal_packet])
            if len(packet_delays) > 0:
                tf_writer.add_histogram("epoch/all_delays", packet_delays, global_step=epoch)
                tf_writer.add_scalar("epoch/e2e_delay_mean", packet_delays.mean(), global_step=epoch)
                tf_writer.add_scalar("epoch/e2e_delay_std", packet_delays.std(), global_step=epoch)
                tf_writer.add_histogram("epoch/first_gsl_delays", first_gsl_delays, global_step=epoch)

            if len(small_packet_delays) > 0:
                tf_writer.add_histogram("epoch/small_packet_delays", small_packet_delays, global_step=epoch)
                tf_writer.add_scalar("epoch/small_packet_delay_mean", small_packet_delays.mean(), global_step=epoch)
                tf_writer.add_scalar("epoch/small_packet_delay_std", small_packet_delays.std(), global_step=epoch)

            if len(normal_packet_delays) > 0:
                tf_writer.add_histogram("epoch/normal_packet_delays", normal_packet_delays, global_step=epoch)
                tf_writer.add_scalar("epoch/normal_packet_delay_mean", normal_packet_delays.mean(), global_step=epoch)
                tf_writer.add_scalar("epoch/normal_packet_delay_std", normal_packet_delays.std(), global_step=epoch)

        # save model for last epoch
        model_dir_path = os.path.join(log_dir, "models")
        os.makedirs(model_dir_path, exist_ok=True)

        last_model_save_path = os.path.join(model_dir_path, "last_model")
        os.makedirs(last_model_save_path, exist_ok=True)
        solver.save_models(model_dir_path=last_model_save_path)

        # save model for eval
        epoch_model_save_path = os.path.join(model_dir_path, f"model_epoch_{epoch}")
        os.makedirs(epoch_model_save_path, exist_ok=True)
        solver.save_models(model_dir_path=epoch_model_save_path)

        if epoch >= 20 and epoch % 10 == 0:
            avg_throughput, avg_drop_rate, avg_e2e_delay, avg_test_cost = eval_performace(env, solver)
            # save the best model
            if best_throughput is None or best_throughput < avg_throughput:
                best_throughput = avg_throughput
                best_model_save_path = os.path.join(model_dir_path, "best_model")
                os.makedirs(best_model_save_path, exist_ok=True)
                solver.save_models(model_dir_path=best_model_save_path)
                logging.info("Best model saved to %s, best_throughput: %.2f", best_model_save_path, best_throughput)


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
    Parses arguments, sets up the environment and solver, and starts the training process.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover_runid", type=str, default=None)
    parser.add_argument("--recover_epoch", type=int, default=1)
    parser.add_argument("--env", type=str, default="configs/starlink_dvbs2_train.json")
    parser.add_argument("--solver", type=str, default="configs/dqn.json")
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=33333)

    args = parser.parse_args()
    args = NamedDict(args)

    runs_dir = os.path.join(PROJECT_ROOT, "runs_train")
    os.makedirs(runs_dir, exist_ok=True)

    if args.recover_runid is not None:
        run_id = args.recover_runid
        log_dir = os.path.join(runs_dir, run_id)
        os.makedirs(log_dir, exist_ok=True)
        print(f"LOG_DIR: {log_dir}")
        setup_logging(log_dir)

        logging.info("Recovering from %s", run_id)
        start_epoch = args.recover_epoch
        logging.info("Starting from epoch %d", start_epoch)

        args.update(NamedDict.load(f"{log_dir}/args.json"))
        logging.info("Loaded args: %s", args.to_string())
        set_seeds(args.seed)
        logging.info("Using seed: %d", args.seed)

        tf_writer = SummaryWriter(log_dir=log_dir)

        # Load from existing run
        env_config = NamedDict.load(f"{log_dir}/env_config.json")
        solver_config = NamedDict.load(f"{log_dir}/solver_config.json")

        logging.info("env_config: %s", env_config.to_string())
        logging.info("solver_config: %s", solver_config.to_string())

        env = RoutingEnvAsync(env_config, tf_writer=tf_writer)
        solver = create_solver(env.obs_dim, env.action_dim, solver_config, tf_writer)
        solver.load_models(f"{log_dir}/models/last_model")
    else:
        env_config = NamedDict.load(args.env)
        solver_config = NamedDict.load(args.solver)

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = f"{solver_config.name}_{date_str}"
        print(f"RUN_ID: {run_id}")

        log_dir = os.path.join(runs_dir, run_id)
        os.makedirs(log_dir, exist_ok=True)
        print(f"LOG_DIR: {log_dir}")
        setup_logging(log_dir)

        tf_writer = SummaryWriter(log_dir=log_dir)

        logging.info("args: %s", args.to_string())
        set_seeds(args.seed)
        logging.info("Using seed: %d", args.seed)

        # create env and solver
        env = RoutingEnvAsync(env_config, tf_writer=tf_writer)
        solver = create_solver(env.obs_dim, env.action_dim, solver_config, tf_writer)

        logging.info("env_config: %s", env_config.to_string())
        logging.info("solver_config: %s", solver_config.to_string())

        env_config.save(os.path.join(log_dir, "env_config.json"))
        solver_config.save(os.path.join(log_dir, "solver_config.json"))
        args.save(os.path.join(log_dir, "args.json"))

        archive_source_code(log_dir)

        start_epoch = 1

    train(env=env, solver=solver, start_epoch=start_epoch, max_epoch=args.num_epochs, log_dir=log_dir, tf_writer=tf_writer)

    tf_writer.close()


if __name__ == "__main__":
    main()

import argparse
import time

from sat_net import ArrayVectorRoutingEnv
from sat_net.pipeline import run_marl_rollout
from sat_net.agent import create_agent
from sat_net.config import DEFAULT_MAIN_CONFIG, DEFAULT_SPF_AGENT_CONFIG, load_config, load_env_config, merge_section

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_MAIN_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--duration_seconds", type=float, default=None)
    parser.add_argument("--concurrent_flowlets_per_env", type=int, default=None)
    parser.add_argument("--region_chunk_size", type=int, default=None)
    args_raw = parser.parse_args()

    main_config = load_config(args_raw.config)
    args = merge_section(
        {
            "config": DEFAULT_MAIN_CONFIG,
            "seed": 3333,
            "duration_seconds": 60.0,
            "concurrent_flowlets_per_env": None,
            "region_chunk_size": 32,
        },
        main_config,
        "spf",
        vars(args_raw),
    )

    start_time = time.time()
    env_config = load_env_config(main_config)
    if args.get("concurrent_flowlets_per_env", None) is not None:
        env_config.traffic.concurrent_flowlets_per_env = int(args.concurrent_flowlets_per_env)
    if args.get("region_chunk_size", None) is not None:
        env_config.traffic.region_chunk_size = int(args.region_chunk_size)
    agent_config = load_config(DEFAULT_SPF_AGENT_CONFIG)
    env = ArrayVectorRoutingEnv(env_config, num_envs=1)
    agent = create_agent(agent_config)
    result = run_marl_rollout(
        env=env,
        agent=agent,
        seed=args.seed,
        train=False,
        duration_seconds=float(args.duration_seconds),
    )
    metrics = result.metrics
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"time_elapsed: {time_elapsed}s")
    print(metrics.get_summary())
    # print(f"metrics: {metrics.to_json(pretty=True)}")

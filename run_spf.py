import argparse
import time

from sat_net import RoutingEnv
from sat_net.pipeline import run_marl_rollout
from sat_net.agent import create_agent
from sat_net.config import DEFAULT_MAIN_CONFIG, DEFAULT_SPF_AGENT_CONFIG, load_config, load_env_config, merge_section

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_MAIN_CONFIG)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args_raw = parser.parse_args()

    main_config = load_config(args_raw.config)
    args = merge_section(
        {"config": DEFAULT_MAIN_CONFIG, "env": None, "seed": 3333},
        main_config,
        "spf",
        vars(args_raw),
    )

    start_time = time.time()
    env_config = load_env_config(main_config, split="spf", override_path=args.env)
    agent_config = load_config(DEFAULT_SPF_AGENT_CONFIG)
    env = RoutingEnv(env_config)
    agent = create_agent(agent_config)
    result = run_marl_rollout(env=env, agent=agent, seed=args.seed, train=False)
    metrics = result.metrics
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"time_elapsed: {time_elapsed}s")
    print(metrics.get_summary())
    # print(f"metrics: {metrics.to_json(pretty=True)}")

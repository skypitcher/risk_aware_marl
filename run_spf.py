import time

from sat_net import RoutingEnv
from sat_net.pipeline import run_marl_episode
from sat_net.agent import create_agent
from sat_net.util import NamedDict

if __name__ == "__main__":
    start_time = time.time()
    env_config = NamedDict.load("configs/starlink_dvbs2_train.json")
    agent_config = NamedDict.load("configs/spf.json")
    env = RoutingEnv(env_config)
    agent = create_agent(agent_config)
    result = run_marl_episode(env=env, agent=agent, seed=3333, train=False)
    metrics = result.metrics
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"time_elapsed: {time_elapsed}s")
    print(metrics.get_summary())
    # print(f"metrics: {metrics.to_json(pretty=True)}")

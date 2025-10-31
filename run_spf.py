import time

from sat_net import RoutingEnvAsync, SPF
from sat_net.util import NamedDict

if __name__ == "__main__":
    start_time = time.time()
    env_config = NamedDict.load("configs/starlink_dvbs2_train.json")
    env = RoutingEnvAsync(env_config)
    solver = SPF()
    env.reset(seed=3333)
    env.run(solver)
    metrics = env.calc_metrics()
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"time_elapsed: {time_elapsed}s")
    print(metrics.get_summary())
    # print(f"metrics: {metrics.to_json(pretty=True)}")

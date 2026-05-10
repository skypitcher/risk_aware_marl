import time

from sat_net import RoutingEnv
from sat_net.solver import create_solver
from sat_net.util import NamedDict

if __name__ == "__main__":
    start_time = time.time()
    env_config = NamedDict.load("configs/starlink_dvbs2_train.json")
    solver_config = NamedDict.load("configs/spf.json")
    env = RoutingEnv(env_config)
    solver = create_solver(solver_config)
    env.reset(seed=3333)
    env.run(solver)
    metrics = env.calc_metrics()
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"time_elapsed: {time_elapsed}s")
    print(metrics.get_summary())
    # print(f"metrics: {metrics.to_json(pretty=True)}")

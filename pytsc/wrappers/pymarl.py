from pytsc import TrafficSignalNetwork
from smac.env import MultiAgentEnv


class PyMARLTrafficSignalNetwork(MultiAgentEnv):
    step_stats = None

    def __init__(self, map_name="monaco", simulator_backend="sumo", **kwargs):
        kwargs.pop("scenario", None)
        self.tsc_env = TrafficSignalNetwork(
            map_name, simulator_backend=simulator_backend, **kwargs
        )
        self.episode_limit = self.tsc_env.episode_limit

    def apply_actions(self, actions):
        self.tsc_env.action_space.apply(actions)

    def close(self):
        pass

    def get_avail_actions(self):
        return self.tsc_env.get_action_mask()

    def get_env_info(self):
        env_info = {
            "agents": list(self.tsc_env.traffic_signals.keys()),
            "episode_limit": self.episode_limit,
            "n_actions": self.get_total_actions(),
            "adjacency_matrix": self.tsc_env.parsed_network.adjacency_matrix,
            "n_agents": self.tsc_env.n_agents,
            "obs_shape": self.get_obs_size(),
            "obs_info": self.tsc_env.observation_space.get_observation_info(),
            "state_shape": self.get_state_size(),
            "distance_matrix": self.tsc_env.parsed_network.distance_matrix,
            "edge_features": self.tsc_env.parsed_network.edge_features,
            "in_degrees": self.tsc_env.parsed_network.in_degrees,
            "out_degrees": self.tsc_env.parsed_network.out_degrees,
            "agent_positions": {
                ts_id: ts.position
                for ts_id, ts in self.tsc_env.traffic_signals.items()
            },
        }
        return env_info

    def get_obs(self):
        return self.tsc_env.get_observations()

    def get_obs_size(self):
        return self.tsc_env.get_observation_size()

    def get_pressures(self):
        return self.tsc_env.metrics.pressures

    def get_state(self):
        return self.tsc_env.get_state()

    def get_local_rewards(self):
        return self.tsc_env.get_rewards()

    def get_state_size(self):
        return self.tsc_env.get_state_size()

    def get_stats(self):
        return self.tsc_env.get_env_stats()

    def get_total_actions(self):
        return self.tsc_env.get_action_size()

    def reset(self):
        self.tsc_env.episode_count += 1
        if self.tsc_env.episode_over:
            self.tsc_env.restart()
        return self.get_obs(), self.get_state()

    def step(self, actions):
        return self.tsc_env.step(actions)

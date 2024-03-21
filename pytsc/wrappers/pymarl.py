from pytsc import TrafficSignalNetwork
from smac.env import MultiAgentEnv


class PyMARLTrafficSignalNetwork(MultiAgentEnv):
    step_stats = None

    def __init__(self, map_name="monaco", simulator_backend="sumo", **kwargs):
        self.tsc_env = TrafficSignalNetwork(
            map_name,
            simulator_backend=simulator_backend,
            additional_config=kwargs,
        )
        self.episode_limit = self.tsc_env.episode_limit
        self.n_agents = len(self.tsc_env.traffic_signals)

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
            "n_agents": self.n_agents,
            "obs_shape": self.get_obs_size(),
            "state_shape": self.get_state_size(),
        }
        return env_info

    def get_obs(self):
        return self.tsc_env.get_observations()

    def get_obs_size(self):
        return self.tsc_env.get_observation_size()

    def get_state(self):
        observations = self.tsc_env.get_observations()
        state = []
        for obs in observations:
            state.extend(obs)
        return state

    def get_state_size(self):
        return int(
            self.tsc_env.get_observation_size()
            * len(self.tsc_env.traffic_signals)
        )

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

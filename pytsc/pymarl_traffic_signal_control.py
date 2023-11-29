from pytsc.traffic_signal_network import TrafficSignalNetwork
from smac.env import MultiAgentEnv


class PyMARLTrafficSignalNetwork(MultiAgentEnv):
    step_stats = None

    def __init__(self, map_name="monaco", simulator_type="sumo", **kwargs):
        self.tsc_env = TrafficSignalNetwork(
            map_name, simulator_type=simulator_type, add_config=kwargs
        )
        self.episode_limit = self.tsc_env.episode_limit
        self.n_agents = len(self.tsc_env.traffic_signals)

    def apply_actions(self, actions):
        self.tsc_env.apply_actions(actions)

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
            "local_rewards_shape": self.get_local_rewards_size(),
            "outgoing_density_map_shape": self.get_local_rewards_size(),
            "norm_mean_incoming_queues": self.n_agents,
            "norm_mean_incoming_speeds": self.n_agents,
            # "kuramotos_shape": self.get_local_rewards_size() ** 2,
            # "orders_shape": self.get_local_rewards_size(),
            # "node_feats_shape": self.tsc_env.metrics.max_n_controlled_phases,
            "edge_feats": self.n_agents,
        }
        return env_info

    def get_norm_mean_queued_for_each_ts(self):
        return self.tsc_env.metrics.norm_mean_queued_for_each_ts

    def get_norm_mean_speed_for_each_ts(self):
        return self.tsc_env.metrics.norm_mean_speed_for_each_ts

    def get_local_rewards(self):
        return self.tsc_env.get_local_rewards()

    def get_kuramotos(self):
        return self.tsc_env.get_kuramotos()

    def get_orders(self):
        return self.tsc_env.get_orders()

    def get_local_rewards_size(self):
        return self.tsc_env.get_local_rewards_size()

    def get_edge_features(self):
        return self.tsc_env.get_speed_matrix()

    def get_node_features(self):
        return self.tsc_env.get_phase_splits()

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

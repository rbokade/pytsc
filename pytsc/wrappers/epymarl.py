from pytsc import TrafficSignalNetwork
from smac.env import MultiAgentEnv


class EPyMARLTrafficSignalNetwork(MultiAgentEnv):
    step_stats = None

    def __init__(self, map_name="pasubio", simulator_backend="sumo", **kwargs):
        kwargs.pop("scenario", None)
        self.tsc_env = TrafficSignalNetwork(
            map_name, simulator_backend=simulator_backend, **kwargs
        )
        self.episode_limit = self.tsc_env.episode_limit
        self.common_reward = kwargs.get("common_reward", True)
        self.reward_scalarlization = kwargs.get("reward_scalarlization", "mean")

    def apply_actions(self, actions):
        self.tsc_env.action_space.apply(actions)

    def close(self):
        pass

    def get_avail_actions(self):
        return self.tsc_env.get_action_mask()

    def get_env_info(self):
        domain_classes = getattr(self.tsc_env.config, "domain_classes", ["baseline"])
        n_domains = len(domain_classes) if domain_classes is not None else 1
        env_info = {
            "agents": list(self.tsc_env.traffic_signals.keys()),
            "episode_limit": self.episode_limit,
            "n_actions": self.get_total_actions(),
            "adjacency_matrix": self.tsc_env.parsed_network.adjacency_matrix,
            "n_agents": self.tsc_env.n_agents,
            "obs_shape": self.get_obs_size(),
            "obs_info": self.tsc_env.observation_space.get_observation_info(),
            "state_shape": self.get_state_size(),
            "n_domains": n_domains,
            "domain_classes": domain_classes,
        }
        return env_info

    def get_obs(self):
        return self.tsc_env.get_observations()

    def get_obs_size(self):
        return self.tsc_env.get_observation_size()

    def get_network_flow(self):
        return self.tsc_env.metrics.network_flow

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

    def get_domain_class(self):
        return self.tsc_env.config.domain_class

    def set_domain_class(self, domain_class):
        self.tsc_env.config.set_domain_class(domain_class)

    def is_terminated(self):
        return self.tsc_env.simulator.is_terminated

    def sim_step(self):
        return self.tsc_env.simulator.sim_step

    def reset(self):
        self.tsc_env.episode_count += 1
        if self.tsc_env.episode_over:
            self.tsc_env.restart()
        return self.get_obs(), self.get_state()

    def step(self, actions):
        reward, eipsode_over, env_info = self.tsc_env.step(actions)
        if self.common_reward:
            if self.reward_scalarlization == "mean":
                reward = reward / self.tsc_env.n_agents
        else:
            reward = self.get_local_rewards()
        return self.get_obs(), reward, eipsode_over, False, env_info

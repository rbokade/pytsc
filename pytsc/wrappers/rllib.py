from pytsc import TrafficSignalNetwork

import numpy as np

from gym.spaces import Dict, Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv


policy_mapping_dict = {
    "all_scenario": {
        "description": "tsc all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class RLlibTrafficSignalNetwork(MultiAgentEnv):
    step_stats = None

    def __init__(self, simulator_config):
        scenario = simulator_config.get("scenario")
        simulator_backend = simulator_config.get("simulator_backend")
        additional_config = simulator_config.get(simulator_backend)
        self.tsc_env = TrafficSignalNetwork(
            scenario,
            simulator_backend=simulator_backend,
            additional_config=additional_config,
        )
        self.episode_limit = self.tsc_env.episode_limit
        self.n_agents = len(self.tsc_env.traffic_signals)
        self.observation_space = Dict(
            {
                "action_mask": Box(
                    low=0,
                    high=1,
                    shape=(self.tsc_env.get_action_size(),),
                    dtype=np.float32,
                ),
                "obs": Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.tsc_env.get_observation_size(),),
                    dtype=np.float32,
                ),
                "state": Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.tsc_env.get_observation_size(),),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Discrete(self.tsc_env.get_action_size())
        # self._agent_ids = list(self.tsc_env.traffic_signals.keys())
        agents = list(self.tsc_env.traffic_signals.keys())
        self.agents = [f"agent_{a}" for a in agents]

    def close(self):
        pass

    def _get_obs_dict(self):
        obs_dict = {}
        action_mask = self.tsc_env.get_action_mask()
        obs = self.tsc_env.get_observations()
        state = self.tsc_env.get_observations()
        for idx, agent in enumerate(self.agents):
            obs_dict[agent] = {
                "action_mask": np.array(action_mask[idx], dtype=np.float32),
                "obs": np.array(obs[idx], dtype=np.float32),
                "state": np.array(state[idx], dtype=np.float32),
            }
        return obs_dict

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "adjacency_matrix": self.tsc_env.parsed_network.adjacency_matrix,
            "policy_mapping_info": policy_mapping_dict,
        }
        return env_info

    def reset(self):
        self.tsc_env.episode_count += 1
        if self.tsc_env.episode_over:
            self.tsc_env.restart()
        return self._get_obs_dict()

    def step(self, actions):
        actions_list = [a for a in actions.values()]
        reward, done, info = self.tsc_env.step(actions_list)
        # rewards = self.tsc_env.get_local_rewards()
        # info = {agent: {} for agent in self.agents}
        # info.update()
        return (
            self._get_obs_dict(),
            {i: reward / len(self.agents) for i in self.agents},
            {"__all__": done},
            {agent: info for agent in self.agents},
        )

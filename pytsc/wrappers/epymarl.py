import gc
import random

import numpy as np
from smac.env import MultiAgentEnv

from pytsc import TrafficSignalNetwork


class EPyMARLTrafficSignalNetwork(MultiAgentEnv):
    """
    A  wrapper for the TrafficSignalNetwork environment that allows for
    multi-agent reinforcement learning. This environment is designed to
    be used with the EPyMARL framework, which is a multi-agent reinforcement
    learning library.
    """

    step_stats = None

    def __init__(self, map_name="pasubio", simulator_backend="sumo", **kwargs):
        kwargs.pop("scenario", None)
        self.tsc_env = TrafficSignalNetwork(
            map_name, simulator_backend=simulator_backend, **kwargs
        )
        self.episode_limit = self.tsc_env.episode_limit
        self.common_reward = kwargs.get("common_reward", True)
        self.reward_scalarization = kwargs.get("reward_scalarization", "mean")

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
            "lane_features": self.tsc_env.observation_space.get_per_agent_lane_features(),
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
        obs, state = self.get_obs(), self.get_state()
        if self.tsc_env.episode_over:
            self.tsc_env.restart()
        return obs, state

    def step(self, actions):
        reward, episode_over, env_info = self.tsc_env.step(actions)
        if self.common_reward:
            if self.reward_scalarization == "mean":
                reward = reward / self.tsc_env.n_agents
        else:
            reward = self.get_local_rewards()
        return self.get_obs(), reward, episode_over, False, env_info


class DomainRandomizedEPyMARLTrafficSignalNetwork(MultiAgentEnv):
    """
    A wrapper for the TrafficSignalNetwork environment that supports
    domain randomization. It randomly selects a map from a provided list at
    each reset, and pads outputs so that the observation and action interfaces
    remain fixed (with a maximum number of agents).
    Args:
        map_names (list): A list of map names to choose from.
        max_n_agents (int): The maximum number of agents.
        simulator_backend (str): The simulator backend to use (default "sumo").
        **kwargs: Additional keyword arguments for TrafficSignalNetwork.
    """

    step_stats = None

    def __init__(self, map_names, simulator_backend="sumo", **kwargs):
        kwargs.pop("scenario", None)
        kwargs.pop("map_name", None)
        self.kwargs = kwargs
        self.map_names = map_names
        self.simulator_backend = simulator_backend
        self.seed = kwargs.get("seed", 0)
        random.seed(self.seed)
        self.current_env = None
        # Initialize first environment instance.
        self.max_n_agents = self._get_max_n_agents()
        self._reset_traffic_signal_network()
        self.current_n_agents = len(self.current_env.tsc_env.traffic_signals)
        self.episode_limit = self.current_env.episode_limit

    def _get_max_n_agents(self):
        """
        Get the maximum number of agents from the list of map names.
        This assumes that all maps have the same number of agents.
        """
        max_n_agents = 0
        for map_name in self.map_names:
            tsc_env = TrafficSignalNetwork(
                map_name, simulator_backend=self.simulator_backend, **self.kwargs
            )
            max_n_agents = max(max_n_agents, len(tsc_env.traffic_signals))
            tsc_env.simulator.close_simulator()
            del tsc_env
        return max_n_agents

    def _get_map_name(self):
        return random.choice(self.map_names)

    def _reset_traffic_signal_network(self):
        if self.current_env is not None:
            self.current_env.simulator.close_simulator()
            del self.current_env
            gc.collect()
        self.current_env = EPyMARLTrafficSignalNetwork(
            map_name=self._get_map_name(),
            simulator_backend=self.simulator_backend,
            **self.kwargs,
        )

    def _pad_adjacency_matrix(self, adjacency_matrix):
        if isinstance(adjacency_matrix, np.ndarray):
            pad_rows = self.max_n_agents - adjacency_matrix.shape[0]
            pad_cols = self.max_n_agents - adjacency_matrix.shape[1]
            padded_matrix = np.pad(
                adjacency_matrix,
                ((0, pad_rows), (0, pad_cols)),
                mode="constant",
                constant_values=0,
            )
            return padded_matrix.tolist()
        else:
            padded_matrix = []
            for row in adjacency_matrix:
                padded_row = row + [0] * (self.max_n_agents - len(row))
                padded_matrix.append(padded_row)
            for _ in range(self.max_n_agents - len(adjacency_matrix)):
                padded_matrix.append([0] * self.max_n_agents)
            return padded_matrix

    def _pad_list(self, lst, pad_value=0):
        """
        Pad a list to length max_n_agents.
        """
        return lst + [pad_value] * (self.max_n_agents - len(lst))

    def apply_actions(self, actions):
        """
        Apply actions and remove padding if necessary.
        """
        valid_actions = actions[: self.current_n_agents]
        self.current_env.apply_actions(valid_actions)

    def get_avail_actions(self):
        """
        Get available actions from the current environment and pad the list.
        """
        action_mask = self.current_env.get_avail_actions()
        if len(action_mask) < self.max_n_agents:
            pad_action_mask = [
                [1] * self.current_env.get_total_actions()
                for _ in range(self.max_n_agents - len(action_mask))
            ]
            action_mask = action_mask + pad_action_mask
        return action_mask

    def get_env_info(self):
        """
        Get environment info from the current environment and pad the
        adjacency matrix. Also, set n_agents to max_n_agents.
        """
        env_info = self.current_env.get_env_info()
        env_info["adjacency_matrix"] = self._pad_adjacency_matrix(
            env_info["adjacency_matrix"]
        )
        env_info["n_agents"] = self.max_n_agents
        return env_info

    def get_obs(self):
        """
        Get observations from the current environment and pad the list of
        observations.
        """
        obs = self.current_env.get_obs()
        if len(obs) < self.max_n_agents:
            pad_obs = [
                [-1] * self.current_env.get_obs_size()
                for _ in range(self.max_n_agents - len(obs))
            ]
            obs = obs + pad_obs
        return obs

    def get_obs_size(self):
        return self.current_env.get_obs_size()

    def get_state(self):
        """
        Get the state from the current environment. The state is padded
        to match the max_n_agents.
        """
        state = self.current_env.get_state()
        if len(state) < self.max_n_agents:
            pad_state = [
                [-1] * self.current_env.get_state_size()
                for _ in range(self.max_n_agents - len(state))
            ]
            state = state + pad_state
        return state

    def get_local_rewards(self):
        """
        Get local rewards from the current environment and pad the list.
        """
        local_rewards = self.current_env.get_local_rewards()
        if len(local_rewards) < self.max_n_agents:
            pad_rewards = [0] * (self.max_n_agents - len(local_rewards))
            local_rewards = local_rewards + pad_rewards
        return local_rewards

    def get_state_size(self):
        return self.current_env.get_state_size()

    def get_total_actions(self):
        return self.current_env.get_total_actions()

    def get_stats(self):
        return self.current_env.get_stats()

    def reset(self):
        """
        Reset the environment. Reinitialize the underlying TrafficSignalNetwork
        using a (potentially) different map to achieve domain randomization.
        """
        self.current_env.tsc_env.episode_count += 1
        obs, state = self.get_obs(), self.get_state()
        if self.current_env.tsc_env.episode_over:
            is_terminated = self.current_env.tsc_env.simulator.is_terminated
            self.current_env.tsc_env.restart(reset=False)
            if is_terminated:
                self._reset_traffic_signal_network()
        return obs, state

    def step(self, actions):
        """
        Step through the environment using valid (unpadded) actions,
        then pad the observations before returning.
        """
        valid_actions = actions[: self.current_n_agents]
        _, reward, episode_over, _, env_info = self.current_env.step(valid_actions)
        if self.current_env.common_reward:
            if self.current_env.reward_scalarization == "mean":
                reward = reward / self.current_env.n_agents
        else:
            reward = self.get_local_rewards()
        padded_obs = self.get_obs()
        return padded_obs, reward, episode_over, False, env_info

    def close(self):
        pass

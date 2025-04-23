from abc import ABC, abstractmethod

import numpy as np


class BaseRewardFunction(ABC):
    """
    Base class for reward functions in traffic signal control.
    This class defines the interface for different reward functions
    and provides common functionality for reward management.
    Args:
        metrics (Metrics): Metrics object containing simulation parameters.
        traffic_signals (dict): Dictionary of traffic signals in the network.
    """
    def __init__(self, metrics, traffic_signals):
        self.metrics = metrics
        self.traffic_signals = traffic_signals
        self.parsed_network = metrics.parsed_network
        self.config = metrics.config

    @abstractmethod
    def get_global_reward(self):
        """
        Calculate the global reward based on pressure and flickering.
        Returns:
            float: Global reward for the network.
        """
        raise NotImplementedError

    @abstractmethod
    def get_local_reward(self):
        """
        Get the local reward for each traffic signal.
        Returns:
            list: List of local rewards for each traffic signal.
        """
        raise NotImplementedError


class QueueLength(BaseRewardFunction):
    """
    Queue Length reward function.
    Args:
        metrics (Metrics): Metrics object containing simulation parameters.
        traffic_signals (dict): Dictionary of traffic signals in the network.
    """
    def __init__(self, metrics, traffic_signals):
        super(QueueLength, self).__init__(metrics, traffic_signals)

    def get_global_reward(self):
        """
        Calculate the global reward based on pressure and flickering.
        Returns:
            float: Global reward for the network.
        """
        fc = self.config.misc["flickering_coef"]
        reward = 1e-6
        reward += fc * self.metrics.flickering_signal
        reward += self.metrics.n_queued
        return -1 * reward

    def get_local_reward(self):
        """
        Calculate the local reward for each traffic signal based on queue length and flickering.
        Returns:
            list: List of local rewards for each traffic signal.
        """
        fc = self.config.misc["flickering_coef"]
        gamma = self.config.misc["reward_gamma"]
        k_hop_neighbors = self.parsed_network.k_hop_neighbors
        local_rewards = {
            ts_id: -fc * ts.controller.program.phase_changed - ts.n_queued - 1e-6
            for ts_id, ts in self.traffic_signals.items()
        }
        rewards = {}
        for ts_id in self.traffic_signals.keys():
            rewards[ts_id] = local_rewards[ts_id]
            for k in range(1, len(self.traffic_signals.keys())):
                neighbors_k = k_hop_neighbors[ts_id].get(k, [])
                for neighbor_ts_id in neighbors_k:
                    rewards[ts_id] += gamma**k * local_rewards[neighbor_ts_id]
        return list(rewards.values())


class MaxPressure(BaseRewardFunction):
    """
    Max Pressure reward function.
    Args:
        metrics (Metrics): Metrics object containing simulation parameters.
        traffic_signals (dict): Dictionary of traffic signals in the network.    
    """
    def __init__(self, metrics, traffic_signals):
        super(MaxPressure, self).__init__(metrics, traffic_signals)

    def get_global_reward(self):
        """
        Calculate the global reward based on pressure and flickering.
        Returns:
            float: Global reward for the network.
        """
        fc = self.config.misc["flickering_coef"]
        reward = 1e-6
        reward -= fc * self.metrics.flickering_signal
        reward -= self.metrics.pressure
        return reward

    def get_local_reward(self):
        """
        Calculate the local reward for each traffic signal based on pressure and flickering.
        Returns:
            list: List of local rewards for each traffic signal.
        """
        fc = self.config.misc["flickering_coef"]
        gamma = self.config.misc["reward_gamma"]
        k_hop_neighbors = self.parsed_network.k_hop_neighbors
        local_rewards = {
            ts_id: -fc * ts.controller.program.phase_changed - ts.pressure - 1e-6
            for ts_id, ts in self.traffic_signals.items()
        }
        rewards = {}
        for ts_id in self.traffic_signals.keys():
            rewards[ts_id] = local_rewards[ts_id]
            for k in range(1, len(self.traffic_signals.keys())):
                neighbors_k = k_hop_neighbors[ts_id].get(k, [])
                for neighbor_ts_id in neighbors_k:
                    rewards[ts_id] += gamma**k * local_rewards[neighbor_ts_id]
        return list(rewards.values())

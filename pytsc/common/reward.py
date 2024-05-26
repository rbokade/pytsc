from abc import ABC, abstractmethod

import numpy as np


class BaseRewardFunction(ABC):
    def __init__(self, metrics, traffic_signals):
        self.metrics = metrics
        self.traffic_signals = traffic_signals
        self.parsed_network = metrics.parsed_network
        self.config = metrics.config

    @abstractmethod
    def get_global_reward(self):
        raise NotImplementedError

    @abstractmethod
    def get_local_reward(self):
        raise NotImplementedError


class QueueLength(BaseRewardFunction):
    def __init__(self, metrics, traffic_signals):
        super(QueueLength, self).__init__(metrics, traffic_signals)

    def get_global_reward(self):
        fc = self.config.misc["flickering_coef"]
        reward = 0
        reward -= fc * self.metrics.flickering_signal
        reward -= self.metrics.n_queued
        return reward

    def get_local_reward(self):
        fc = self.config.misc["flickering_coef"]
        gamma = self.config.misc["reward_gamma"]
        k_hop_neighbors = self.parsed_network.k_hop_neighbors
        local_rewards = {
            ts_id: -fc * ts.controller.program.phase_changed
            - np.mean(ts.queue_lengths)
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
    def __init__(self, metrics, traffic_signals):
        super(MaxPressure, self).__init__(metrics, traffic_signals)

    def get_global_reward(self):
        fc = self.config.misc["flickering_coef"]
        reward = 0
        reward -= fc * self.metrics.flickering_signal
        reward -= self.metrics.pressure
        return reward

    def get_local_reward(self):
        fc = self.config.misc["flickering_coef"]
        gamma = self.config.misc["reward_gamma"]
        k_hop_neighbors = self.parsed_network.k_hop_neighbors
        local_rewards = {
            ts_id: -fc * ts.controller.program.phase_changed
            - np.mean(ts.pressure)
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

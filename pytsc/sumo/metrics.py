import numpy as np

from pytsc.common.metrics import BaseMetricsParser


class MetricsParser(BaseMetricsParser):
    """
    Retrieve performance metrics from subscriptions.
    """

    def __init__(self, parsed_network, simulator, traffic_signals):
        self.config = parsed_network.config
        self.simulator = simulator
        self.parsed_network = parsed_network
        self.traffic_signals = traffic_signals

    @property
    def flickering_signal(self):
        return np.mean(
            [
                ts.controller.phase_changed
                for ts in self.traffic_signals.values()
            ]
        )

    @property
    def time_step(self):
        return self.simulator.step_measurements["sim"]["time_step"]

    @property
    def n_emergency_brakes(self):
        return self.simulator.step_measurements["sim"]["n_emergency_brakes"]

    @property
    def n_teleported(self):
        return self.simulator.step_measurements["sim"]["n_teleported"]

    @property
    def n_vehicles_exited(self):
        return self.simulator.step_measurements["sim"]["n_arrived"]

    @property
    def n_vehicles_inserted(self):
        return self.simulator.step_measurements["sim"]["n_departed"]

    @property
    def norm_mean_queued(self):
        """
        mean queued per lane per ts (normalized by lane length)
        """
        return np.mean(
            [
                np.mean(ts.norm_queue_lengths)
                for ts in self.traffic_signals.values()
            ]
        )

    @property
    def mean_queued(self):
        """
        mean queued per lane per ts
        """
        return np.mean(
            [np.mean(ts.queue_lengths) for ts in self.traffic_signals.values()]
        )

    @property
    def n_queued(self):
        return np.sum(
            [np.sum(ts.queue_lengths) for ts in self.traffic_signals.values()]
        )

    @property
    def mean_wait_time(self):
        return np.mean(
            [
                np.mean(ts.mean_wait_times)
                for ts in self.traffic_signals.values()
            ]
        )

    @property
    def average_travel_time(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(
            data["average_travel_time"] for data in lane_measurements.values()
        ) / len(lane_measurements)

    @property
    def mean_speed(self):
        return np.mean(
            [np.mean(ts.mean_speeds) for ts in self.traffic_signals.values()]
        )

    @property
    def density(self):
        return np.mean(
            [np.mean(ts.densities) for ts in self.traffic_signals.values()]
        )

    @property
    def reward(self):
        fc = self.config.misc_config["flickering_coef"]
        reward = 0
        reward -= fc * self.flickering_signal
        reward -= self.norm_mean_queued
        return reward

    @property
    def rewards(self):
        fc = self.config.misc_config["flickering_coef"]
        gamma = self.config.misc_config["reward_gamma"]
        k_hop_neighbors = self.parsed_network.k_hop_neighbors
        local_rewards = {
            ts_id: -fc * ts.controller.phase_changed
            - np.mean(ts.norm_queue_lengths)
            for ts_id, ts in self.traffic_signals.items()
        }
        rewards = {}
        for ts_id, ts in self.traffic_signals.items():
            rewards[ts_id] = local_rewards[ts_id]
            for k in range(1, len(self.traffic_signals.keys())):
                neighbors_k = k_hop_neighbors[ts_id].get(k, [])
                for neighbor_ts_id in neighbors_k:
                    rewards[ts_id] += (
                        gamma**k * local_rewards[neighbor_ts_id]
                    )
        return list(rewards.values())

    def get_step_stats(self):
        return {
            "time_step": self.time_step,
            "n_emergency_brakes": self.n_emergency_brakes,
            "n_teleported": self.n_teleported,
            "n_inserted": self.n_vehicles_inserted,
            "n_exited": self.n_vehicles_exited,
            "n_queued": self.n_queued,
            "mean_speed": self.mean_speed,
            "mean_wait_time": self.mean_wait_time,
            "average_travel_time": self.average_travel_time,
            "density": self.density,
        }

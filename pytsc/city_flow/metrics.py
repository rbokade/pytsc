import numpy as np

from pytsc.common.metrics import BaseMetricsParser


class MetricsParser(BaseMetricsParser):
    """
    Network wide metrics
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
    def n_queued(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        total_queued = 0
        for lane_id, data in lane_measurements.items():
            total_queued += data["n_queued"]
        return total_queued

    @property
    def norm_mean_queued_per_ts(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        total_norm_queued = sum(
            data["norm_queue_length"] for data in lane_measurements.values()
        )
        norm_mean_queued = total_norm_queued / len(self.traffic_signals)
        return norm_mean_queued

    @property
    def norm_mean_queued_for_each_ts(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        norm_mean_queued_for_each_tsc = []
        for ts_id in self.traffic_signals.keys():
            incoming_lanes = self.parsed_network.incoming_lanes_map[ts_id]
            norm_queued = 0
            for incoming_lane in incoming_lanes:
                norm_queued += lane_measurements[incoming_lane][
                    "norm_queue_length"
                ]
            norm_queued /= len(incoming_lanes)
            norm_mean_queued_for_each_tsc.append(norm_queued)
        return np.asarray(norm_mean_queued_for_each_tsc)

    @property
    def norm_mean_speed_for_each_ts(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        norm_mean_speed_for_each_tsc = []
        for ts_id in self.traffic_signals.keys():
            incoming_lanes = self.parsed_network.incoming_lanes_map[ts_id]
            norm_speed = 0
            for incoming_lane in incoming_lanes:
                norm_speed += lane_measurements[incoming_lane][
                    "norm_mean_speed"
                ]
            norm_speed /= len(incoming_lanes)
            norm_mean_speed_for_each_tsc.append(norm_speed)
        return norm_mean_speed_for_each_tsc

    @property
    def mean_speed(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        total_vehicles = sum(
            data["n_vehicles"] for data in lane_measurements.values()
        )
        if total_vehicles == 0:
            return 0.0
        else:
            total_vehicle_speed = sum(
                data["mean_speed"] * data["n_vehicles"]
                for data in lane_measurements.values()
            )
            return total_vehicle_speed / total_vehicles

    @property
    def density(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        total_occupancy = sum(
            data["occupancy"] for data in lane_measurements.values()
        )
        return total_occupancy / len(lane_measurements)

    @property
    def average_travel_time(self):
        return self.simulator.step_measurements["sim"]["average_travel_time"]

    @property
    def mean_delay(self):
        return 1 - np.mean(
            [
                np.mean(ts.norm_mean_speeds)
                for ts in self.traffic_signals.values()
            ]
        )

    @property
    def reward(self):
        fc = self.config.misc_config["flickering_coef"]
        reward = 0
        reward -= fc * self.flickering_signal
        reward -= self.norm_mean_queued_per_ts
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
                n_neighbors = len(neighbors_k)
                mean_neighbors_reward = 0
                if n_neighbors > 0:
                    mean_neighbors_reward = (
                        sum(
                            local_rewards[neighbor_ts_id]
                            for neighbor_ts_id in neighbors_k
                        )
                        / n_neighbors
                    )
                rewards[ts_id] += (gamma**k) * mean_neighbors_reward
        return np.asarray(list(rewards.values()))

    def get_step_stats(self):
        step_stats = {
            "time": self.simulator.step_measurements["sim"]["time"],
            "average_travel_time": self.average_travel_time,
            "n_queued": self.n_queued,
            "mean_speed": self.mean_speed,
            "mean_delay": self.mean_delay,
            "density": self.density,
            "norm_mean_queued_per_ts": self.norm_mean_queued_per_ts,
        }
        return step_stats

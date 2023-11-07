import numpy as np

from pytsc.common.metrics import BaseMetricsParser
from pytsc.common.utils import compute_local_order_for_agent


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
            [ts.controller.phase_changed for ts in self.traffic_signals.values()]
        )

    @property
    def n_queued(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        total_queued = 0
        for data in lane_measurements.values():
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
                norm_queued += lane_measurements[incoming_lane]["norm_queue_length"]
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
                norm_speed += lane_measurements[incoming_lane]["norm_mean_speed"]
            norm_speed /= len(incoming_lanes)
            norm_mean_speed_for_each_tsc.append(norm_speed)
        return norm_mean_speed_for_each_tsc

    @property
    def mean_speed(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        total_vehicles = sum(data["n_vehicles"] for data in lane_measurements.values())
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
        total_occupancy = sum(data["occupancy"] for data in lane_measurements.values())
        return total_occupancy / len(lane_measurements)

    @property
    def average_travel_time(self):
        return self.simulator.step_measurements["sim"]["average_travel_time"]

    @property
    def mean_delay(self):
        return 1 - np.mean(
            [np.mean(ts.norm_mean_speeds) for ts in self.traffic_signals.values()]
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
        local_rewards = {ts_id: 0 for ts_id in self.traffic_signals.keys()}
        for ts_id, ts in self.traffic_signals.items():
            local_rewards[ts_id] -= fc * ts.controller.phase_changed
            local_rewards[ts_id] -= np.mean(ts.norm_queue_lengths)
            for k in range(1, len(self.traffic_signals.keys())):
                neighbors_k = k_hop_neighbors[ts_id].get(k, [])
                n_neighbors = len(neighbors_k)
                mean_neighbors_reward = 0
                if n_neighbors > 0:
                    total_neigh_reward = sum(
                        local_rewards[neighbor_ts_id] for neighbor_ts_id in neighbors_k
                    )
                    mean_neighbors_reward = total_neigh_reward / n_neighbors
                local_rewards[ts_id] += (gamma**k) * mean_neighbors_reward
        return np.asarray(list(local_rewards.values()))

    @property
    def phase_angles(self):
        phase_angles = {}
        for ts_id, ts in self.traffic_signals.items():
            phase_angles[ts_id] = ts.controller.phase / ts.controller.n_phases
            phase_angles[ts_id] *= 2 * np.pi
        return phase_angles

    @property
    def orders(self):
        ts_ids = list(self.traffic_signals.keys())
        neighbors_lanes = self.parsed_network.neighbors_lanes
        n_ts = len(ts_ids)
        phase_angles_matrix = np.zeros((n_ts, n_ts))
        for i, (ts_id, _) in enumerate(self.traffic_signals.items()):
            neighbors = neighbors_lanes[ts_id]
            if not neighbors:
                continue
            for j, (neigh_ts_id, neigh_ts) in enumerate(self.traffic_signals.items()):
                if neigh_ts_id in neighbors:
                    lanes = neighbors_lanes[ts_id][neigh_ts_id]
                    travel_time = sum(
                        (self.parsed_network.lane_lengths[lane])
                        / self.parsed_network.lane_max_speeds[lane]
                        for lane in lanes
                    )  # intersection_width = 20m
                    offset_t = travel_time / len(lanes)
                    offset_t /= self.config.cityflow_config["delta_time"]
                    offset_t = int(offset_t)
                    t = len(neigh_ts.controller.phase_history) - 1
                    offset_idx = max(t - offset_t, 0)
                    offset_phase_index_t = neigh_ts.controller.phase_history[offset_idx]
                    offset_phase_index_t /= neigh_ts.controller.n_phases
                    offset_phase_angle = offset_phase_index_t * 2 * np.pi
                    phase_angles_matrix[i, j] = offset_phase_angle
        # Compute local orders
        local_orders = np.zeros(n_ts)
        for i in range(n_ts):
            local_orders[i] = compute_local_order_for_agent(
                phase_angles_matrix[i, :],
                self.parsed_network.adjacency_matrix[i, :],
            )
        return local_orders

    @property
    def kuramotos(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        directional_lanes = self.parsed_network.directional_lanes
        neighbors_lanes = self.parsed_network.neighbors_lanes
        phase_angles = self.phase_angles
        n_ts = len(self.traffic_signals.items())
        kuramotos = np.zeros((n_ts, n_ts))
        for i, (ts_id, _) in enumerate(self.traffic_signals.items()):
            neighbors = neighbors_lanes[ts_id]
            if not neighbors:
                continue
            for j, (neigh_ts_id, neigh_ts) in enumerate(self.traffic_signals.items()):
                if neigh_ts_id in neighbors:
                    lanes = neighbors_lanes[ts_id][neigh_ts_id]
                    travel_time = sum(
                        (self.parsed_network.lane_lengths[lane])
                        / self.parsed_network.lane_max_speeds[lane]
                        for lane in lanes
                    )  # intersection_width = 20m
                    offset_t = travel_time / len(lanes)
                    offset_t /= self.config.cityflow_config["delta_time"]
                    offset_t = int(offset_t)
                    t = len(neigh_ts.controller.phase_history) - 1
                    offset_idx = max(t - offset_t, 0)
                    offset_phase_index = neigh_ts.controller.phase_history[offset_idx]
                    offset_phase_index /= neigh_ts.controller.n_phases
                    offset_phase_angle = offset_phase_index * 2 * np.pi
                    neigh_lanes = directional_lanes[ts_id][neigh_ts_id]
                    coupling_strength = sum(
                        [lane_measurements[lane]["occupancy"] for lane in neigh_lanes]
                    ) / len(neigh_lanes)
                    kuramotos[i, j] = coupling_strength * np.sin(
                        offset_phase_angle - phase_angles[ts_id]
                    )
        return kuramotos.flatten().tolist()

    def get_step_stats(self):
        step_stats = {
            "time": self.simulator.step_measurements["sim"]["time"],
            "average_travel_time": self.average_travel_time,
            "n_queued": self.n_queued,
            "mean_speed": self.mean_speed,
            "mean_delay": self.mean_delay,
            "density": self.density,
            "norm_mean_queued_per_ts": self.norm_mean_queued_per_ts,
            "kuramoto": np.sum(np.abs(self.kuramotos)) / np.sum(self.parsed_network.adjacency_matrix),
            "order": np.mean(self.orders),
        }
        phases = {
            f"{ts_id}_phase": ts.controller.program.current_phase
            for ts_id, ts in self.traffic_signals.items()
        }
        step_stats.update(phases)
        return step_stats

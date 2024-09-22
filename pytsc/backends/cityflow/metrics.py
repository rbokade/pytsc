import numpy as np

from pytsc.common.metrics import BaseMetricsParser
from pytsc.common.utils import compute_max_spanning_tree


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
                ts.controller.program.phase_changed
                for ts in self.traffic_signals.values()
            ]
        )

    @property
    def n_queued(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        total_queued = 0
        for data in lane_measurements.values():
            total_queued += data["n_queued"]
        return total_queued

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
        total_occupancy = sum(
            data["occupancy"] for data in lane_measurements.values()
        ).item()
        return total_occupancy / len(lane_measurements)

    @property
    def average_travel_time(self):
        return self.simulator.step_measurements["sim"]["average_travel_time"]

    @property
    def mean_delay(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        mean_speed = sum(
            data["norm_mean_speed"] for data in lane_measurements.values()
        ) / len(lane_measurements)
        return 1 - mean_speed

    @property
    def time_step(self):
        return self.simulator.step_measurements["sim"]["time_step"]

    @property
    def pressure(self):
        return np.sum([ts.pressure for ts in self.traffic_signals.values()]).item()

    @property
    def pressures(self):
        return [ts.pressure for ts in self.traffic_signals.values()]

    @property
    def density_map(self):
        neighbors_lanes = self.parsed_network.neighbors_lanes
        ts_ids = list(self.traffic_signals.keys())
        density_map = np.zeros((len(ts_ids), len(ts_ids)))
        lane_measurements = self.simulator.step_measurements["lane"]
        for i, ts in enumerate(ts_ids):
            neighbors = neighbors_lanes[ts]
            if not neighbors:
                continue
            for j, n_ts_id in enumerate(ts_ids):
                if n_ts_id in neighbors:
                    lanes = neighbors_lanes[ts][n_ts_id]
                    total_occupancy = 0
                    for lane in lanes:
                        current_density = lane_measurements[lane]["occupancy"]
                        total_occupancy += current_density
                    mean_occupancy = total_occupancy / len(lanes)
                    density_map[i, j] = np.clip(mean_occupancy, 0, 1).item()
        undirected_density_map = (
            density_map + density_map.T
        ) / 2  # sum up inc and out lanes and convert to symmetric matrix
        return undirected_density_map + 1e-6 * self.parsed_network.adjacency_matrix

    @property
    def mst(self):
        return compute_max_spanning_tree(self.density_map)

    def get_step_stats(self):
        step_stats = {
            "time_step": self.time_step,
            "average_travel_time": self.average_travel_time,
            "n_queued": self.n_queued,
            "mean_speed": self.mean_speed,
            "mean_delay": self.mean_delay,
            "density": self.density,
            "pressure": self.pressure,
        }
        if self.config.misc["return_agent_stats"]:
            agent_stats = {}
            agent_stats.update(
                {
                    f"{ts.id}__phase": ts.controller.current_phase
                    for ts in self.traffic_signals.values()
                }
            )
            agent_stats.update(
                {
                    f"{ts.id}__n_queued": np.sum(ts.queue_lengths).item()
                    for ts in self.traffic_signals.values()
                }
            )
            agent_stats.update(
                {
                    f"{ts.id}__mean_speed": np.mean(ts.mean_speeds).item()
                    for ts in self.traffic_signals.values()
                }
            )
            agent_stats.update(
                {
                    f"{ts.id}__mean_delay": 1 - np.mean(ts.norm_mean_speeds).item()
                    for ts in self.traffic_signals.values()
                }
            )
            agent_stats.update(
                {
                    f"{ts.id}__mean_density": np.mean(ts.densities).item()
                    for ts in self.traffic_signals.values()
                }
            )
            agent_stats.update(
                {
                    f"{ts.id}__pressure": np.mean(ts.pressure).item()
                    for ts in self.traffic_signals.values()
                }
            )
            step_stats.update(agent_stats)
        if self.config.misc["return_lane_stats"]:
            lane_measurements = self.simulator.step_measurements["lane"]
            stat_keys = ("n_vehicles", "n_queued", "mean_speed", "occupancy")
            lane_stats = {}
            for lane, stat_dict in lane_measurements.items():
                for k in stat_keys:
                    try:
                        lane_stats[f"{lane}__{k}"] = stat_dict[k].item()
                    except Exception:
                        lane_stats[f"{lane}__{k}"] = stat_dict[k]
            step_stats.update(lane_stats)
        return step_stats

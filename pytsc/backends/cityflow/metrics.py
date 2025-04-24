import numpy as np

from pytsc.common.metrics import BaseMetricsParser
from pytsc.common.utils import compute_max_spanning_tree


class MetricsParser(BaseMetricsParser):
    """
    Network wide metrics

    Args:
        parsed_network (ParsedNetwork): Parsed network object.
        simulator (CityFlowSimulator): CityFlow simulator object.
        traffic_signals (dict): Dictionary of traffic signals.
    """

    def __init__(self, parsed_network, simulator, traffic_signals):
        self.config = parsed_network.config
        self.simulator = simulator
        self.parsed_network = parsed_network
        self.traffic_signals = traffic_signals

    @property
    def flickering_signal(self):
        """
        The flickering signal is the average of the phase changed signals
        of all traffic signals in the network. It indicates how often
        the traffic signals are changing phases.

        Returns:
            float: The flickering signal.
        """
        return np.mean(
            [
                ts.controller.program.phase_changed
                for ts in self.traffic_signals.values()
            ]
        )

    @property
    def n_queued(self):
        """
        The total number of queued vehicles in the network.

        Returns:
            int: The total number of queued vehicles.
        """
        lane_measurements = self.simulator.step_measurements["lane"]
        total_queued = 0
        for data in lane_measurements.values():
            total_queued += data["n_queued"]
        return total_queued

    @property
    def n_queued_norm(self):
        """
        The normalized number of queued vehicles in the network.

        Returns:
            float: The normalized number of queued vehicles.
        """
        lane_measurements = self.simulator.step_measurements["lane"]
        n_queued_norm = 0
        for lane_id, data in lane_measurements.items():
            lane_length = self.parsed_network.lane_lengths[lane_id]
            n_queued_norm += data["n_queued"] / lane_length
        return n_queued_norm / len(lane_measurements)

    @property
    def mean_speed(self):
        """
        The mean speed of vehicles in the network.

        Returns:
            float: The mean speed of vehicles.
        """
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
        """
        The density of vehicles in the network.

        Returns:
            float: The density of vehicles.
        """
        lane_measurements = self.simulator.step_measurements["lane"]
        total_occupancy = sum(
            data["occupancy"] for data in lane_measurements.values()
        ).item()
        return total_occupancy / len(lane_measurements)

    @property
    def average_travel_time(self):
        """
        The average travel time of vehicles in the network.

        Returns:
            float: The average travel time of vehicles.
        """
        return self.simulator.step_measurements["sim"]["average_travel_time"]

    @property
    def norm_mean_speed(self):
        """
        The normalized mean speed of vehicles in the network.

        Returns:    
            float: The normalized mean speed of vehicles.
        """
        lane_measurements = self.simulator.step_measurements["lane"]
        lane_max_speeds = self.parsed_network.lane_max_speeds
        return sum(
            data["mean_speed"] / lane_max_speeds[lane_id]
            for lane_id, data in lane_measurements.items()
        ) / len(lane_measurements)

    @property
    def mean_delay(self):
        """
        The mean delay of vehicles in the network.

        Returns:
            float: The mean delay of vehicles.
        """
        return 1 - self.norm_mean_speed

    @property
    def time_step(self):
        """
        The time step of the simulation.

        Returns:
            float: The time step of the simulation.
        """
        return self.simulator.step_measurements["sim"]["time_step"]

    @property
    def pressure(self):
        """
        The pressure of the network is the sum of the pressure of all
        traffic signals in the network. It indicates the congestion level
        of the network.

        Returns:    
            float: The pressure of the network.
        """
        return np.sum([ts.pressure for ts in self.traffic_signals.values()]).item()

    @property
    def pressures(self):
        """
        The pressure of each traffic signal in the network.

        Returns:
            list: A list of pressure values for each traffic signal.
        """
        return [ts.pressure for ts in self.traffic_signals.values()]

    @property
    def density_map(self):
        """
        The density map is a matrix that represents the density of vehicles
        between traffic signals in the network. It is calculated based on
        the occupancy of lanes connecting the traffic signals.

        Returns:
            np.ndarray: The density map of the network.
        """
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
        """
        The maximum spanning tree (MST) of the density map.

        Returns:
            np.ndarray: The maximum spanning tree of the density map.
        """
        return compute_max_spanning_tree(self.density_map)

    @property
    def network_flow(self):
        """
        The network flow is the product of the density and the normalized mean speed.

        Returns:
            float: The network flow.
        """
        return self.density * self.norm_mean_speed

    def get_step_stats(self):
        """
        Get the step statistics for the simulation.

        Returns:
            dict: A dictionary containing various statistics for the current step.
        """
        step_stats = {
            "time_step": self.time_step,
            "average_travel_time": self.average_travel_time,
            "n_queued": self.n_queued,
            "mean_speed": self.mean_speed,
            "mean_delay": self.mean_delay,
            "density": self.density,
            "pressure": self.pressure,
            "network_flow": self.network_flow,
        }
        if self.config.misc["return_agent_stats"]:
            for ts in self.traffic_signals.values():
                ts_stats = {
                    f"{ts.id}__phase": ts.controller.current_phase,
                    f"{ts.id}__n_queued": ts.n_queued,
                    f"{ts.id}__mean_speed": ts.mean_speed,
                    f"{ts.id}__mean_delay": ts.mean_delay,
                    f"{ts.id}__density": ts.occupancy,
                    f"{ts.id}__pressure": ts.pressure,
                }
                step_stats.update(ts_stats)
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

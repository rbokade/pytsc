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
        return np.sum([ts.pressure for ts in self.traffic_signals.values()])

    def get_step_stats(self):
        agent_stats = {}
        agent_stats.update(
            {
                f"n_queued_{ts.id}": np.sum(ts.queue_lengths)
                for ts in self.traffic_signals.values()
            }
        )
        agent_stats.update(
            {
                f"mean_speed_{ts.id}": np.mean(ts.mean_speeds)
                for ts in self.traffic_signals.values()
            }
        )
        agent_stats.update(
            {
                f"mean_delay_{ts.id}": 1 - np.mean(ts.norm_mean_speeds)
                for ts in self.traffic_signals.values()
            }
        )
        agent_stats.update(
            {
                f"mean_density_{ts.id}": np.mean(ts.densities)
                for ts in self.traffic_signals.values()
            }
        )
        agent_stats.update(
            {
                f"pressure_{ts.id}": np.mean(ts.pressure)
                for ts in self.traffic_signals.values()
            }
        )
        step_stats = {
            "time_step": self.time_step,
            "average_travel_time": self.average_travel_time,
            "n_queued": self.n_queued,
            "mean_speed": self.mean_speed,
            "mean_delay": self.mean_delay,
            "density": self.density,
            "pressure": self.pressure,
        }
        step_stats.update(agent_stats)
        return step_stats

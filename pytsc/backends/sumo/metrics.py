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
    def n_queued(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(data["n_queued"] for data in lane_measurements.values())

    @property
    def n_queued_norm(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(data["n_queued"] for data in lane_measurements.values()) / len(
            lane_measurements
        )

    @property
    def mean_wait_time(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(data["mean_wait_time"] for data in lane_measurements.values()) / len(
            lane_measurements
        )

    @property
    def average_travel_time(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(
            data["average_travel_time"] for data in lane_measurements.values()
        ) / len(lane_measurements)

    @property
    def mean_speed(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(data["mean_speed"] for data in lane_measurements.values()) / len(
            lane_measurements
        )

    @property
    def density(self):
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(data["occupancy"] for data in lane_measurements.values()) / len(
            lane_measurements
        )

    @property
    def mean_delay(self):
        return 1 - self.mean_speed

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
        step_stats = {
            "time_step": self.time_step,
            "n_emergency_brakes": self.n_emergency_brakes,
            "n_teleported": self.n_teleported,
            "n_inserted": self.n_vehicles_inserted,
            "n_exited": self.n_vehicles_exited,
            "n_queued": self.n_queued,
            "mean_speed": self.mean_speed,
            "mean_wait_time": self.mean_wait_time,
            "mean_delay": self.mean_delay,
            "average_travel_time": self.average_travel_time,
            "density": self.density,
        }
        step_stats.update(agent_stats)
        return step_stats

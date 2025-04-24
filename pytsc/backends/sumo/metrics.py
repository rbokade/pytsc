import numpy as np

from pytsc.common.metrics import BaseMetricsParser


class MetricsParser(BaseMetricsParser):
    """
    Traffic signal network metrics parser for SUMO simulator.

    Args:
        parsed_network (ParsedNetwork): Parsed network object.
        simulator (SUMOSimulator): SUMO simulator object.
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
    def time_step(self):
        """
        The time step of the simulation.

        Returns:
            float: The time step of the simulation.
        """
        return self.simulator.step_measurements["sim"]["time_step"]

    @property
    def n_emergency_brakes(self):
        """
        The number of emergency brakes in the simulation.

        Returns:
            int: The number of emergency brakes.
        """
        return self.simulator.step_measurements["sim"]["n_emergency_brakes"]

    @property
    def n_teleported(self):
        """
        The number of teleported vehicles in the simulation.

        Returns:
            int: The number of teleported vehicles.
        """
        return self.simulator.step_measurements["sim"]["n_teleported"]

    @property
    def n_vehicles_exited(self):
        """
        The number of vehicles that exited the simulation.

        Returns:    
            int: The number of vehicles that exited the simulation.
        """
        return self.simulator.step_measurements["sim"]["n_arrived"]

    @property
    def n_vehicles_inserted(self):
        """
        The number of vehicles that were inserted into the simulation.

        Returns:
            int: The number of vehicles that were inserted into the simulation.
        """
        return self.simulator.step_measurements["sim"]["n_departed"]

    @property
    def n_queued(self):
        """
        The total number of queued vehicles in the network.

        Returns:
            int: The total number of queued vehicles.
        """
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(data["n_queued"] for data in lane_measurements.values())

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
    def average_wait_time(self):
        """
        The average wait time of vehicles in the network.

        Returns:
            float: The average wait time of vehicles.
        """
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(
            data["average_wait_time"] for data in lane_measurements.values()
        ) / len(lane_measurements)

    @property
    def average_travel_time(self):
        """ 
        The average travel time of vehicles in the network.

        Returns:
            float: The average travel time of vehicles.
        """
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(
            data["average_travel_time"] for data in lane_measurements.values()
        ) / len(lane_measurements)

    @property
    def mean_speed(self):
        """
        The mean speed of vehicles in the network.

        Returns:
            float: The mean speed of vehicles.
        """
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(data["mean_speed"] for data in lane_measurements.values()) / len(
            lane_measurements
        )

    @property
    def density(self):
        """
        The density of vehicles in the network.

        Returns:
            float: The density of vehicles.
        """
        lane_measurements = self.simulator.step_measurements["lane"]
        return sum(data["occupancy"] for data in lane_measurements.values()) / len(
            lane_measurements
        )

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
    def pressure(self):
        """
        The pressure of the traffic signals in the network.

        Returns:
            float: The pressure of the traffic signals.
        """
        return np.sum([ts.pressure for ts in self.traffic_signals.values()])

    def get_step_stats(self):
        """
        Get the statistics of the current simulation step.

        Returns:
            dict: Dictionary containing the statistics of the current step.
        """
        step_stats = {
            "time_step": self.time_step,
            "n_emergency_brakes": self.n_emergency_brakes,
            "n_teleported": self.n_teleported,
            "n_inserted": self.n_vehicles_inserted,
            "n_exited": self.n_vehicles_exited,
            "density": self.density,
            "n_queued": self.n_queued,
            "mean_speed": self.mean_speed,
            "mean_delay": self.mean_delay,
            "average_travel_time": self.average_travel_time,
            "average_wait_time": self.average_wait_time,
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

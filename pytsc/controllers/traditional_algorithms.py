from abc import ABC

import logging

import numpy as np

from pytsc.common.utils import EnvLogger

# EnvLogger.set_log_level(logging.WARNING)

class BasePhaseSelector(ABC):
    def __init__(self, traffic_signal, round_robin=True, **kwargs):
        self.traffic_signal = traffic_signal
        self.round_robin = round_robin

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.traffic_signal.id})"


class FixedTimePhaseSelector(BasePhaseSelector):
    def __init__(self, traffic_signal, green_time=25):
        super(FixedTimePhaseSelector, self).__init__(traffic_signal)
        self.green_time = green_time
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        if (
            self.controller.current_phase_index
            in self.controller.green_phase_indices
            and self.controller.time_on_phase < self.green_time
        ):
            return self.controller.current_phase_index
        else:
            return self.controller.next_phase_index


class GreedyPhaseSelector(BasePhaseSelector):
    def __init__(self, traffic_signal):
        super(GreedyPhaseSelector, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        action_mask = self.controller.get_allowable_phase_switches()
        queues = []
        if (
            self.controller.current_phase_index
            in self.controller.green_phase_indices
        ):
            for act, available in enumerate(action_mask):
                if available:
                    queue = self._compute_queue_for_phase(inp, act)
                else:
                    queue = float("-inf")
                queues.append((queue, act))
            _, max_queue_phase_index = max(queues)
        else:
            max_queue_phase_index = self.controller.next_phase_index
        return max_queue_phase_index

    def _compute_queue_for_phase(self, inp, phase_index):
        inc_vehicles = 0
        phase = self.traffic_signal.config["phases"][phase_index]
        phase_inc_out_lanes = self.traffic_signal.config[
            "phase_to_inc_out_lanes"
        ][phase]
        for inc_lane in phase_inc_out_lanes.keys():
            inc_vehicles += sum(
                inp["lane"][inc_lane]["position_speed_matrices"][0]
            ) / (
                sum(inp["lane"][inc_lane]["position_speed_matrices"][1]) + 1e-6
            )
        return inc_vehicles


class MaxPressurePhaseSelector(BasePhaseSelector):
    def __init__(self, traffic_signal):
        super(MaxPressurePhaseSelector, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        action_mask = self.controller.get_allowable_phase_switches()
        pressures = []
        if (
            self.controller.current_phase_index
            in self.controller.green_phase_indices
        ):
            for act, available in enumerate(action_mask):
                if available:
                    pressure = self._compute_pressure_for_phase(inp, act)
                else:
                    pressure = float("-inf")
                pressures.append((pressure, act))
            _, max_pressure_phase_index = max(pressures)
        else:
            max_pressure_phase_index = self.controller.next_phase_index
        return max_pressure_phase_index

    def _compute_pressure_for_phase(self, inp, phase_index):
        """
        inp (dict): network.simulator.step_measurements["lane"]
        """
        pressure = 0
        phase = self.traffic_signal.config["phases"][phase_index]
        phase_inc_out_lanes = self.traffic_signal.config[
            "phase_to_inc_out_lanes"
        ][phase]
        for inc_lane, out_lanes in phase_inc_out_lanes.items():
            inc_lane_vehicles = sum(
                inp["lane"][inc_lane]["position_speed_matrices"][0]
            )
            out_lane_vehicles = 0
            for out_lane in out_lanes:
                out_lane_vehicles += sum(
                    inp["lane"][out_lane]["position_speed_matrices"][0]
                )
            pressure += np.abs(inc_lane_vehicles - out_lane_vehicles)
        return pressure


class SOTLPhaseSelector(BasePhaseSelector):
    """
    mu: threshold for vehicles on green phase lanes
    theta: threshold for vehicles on red phase lanes
    phi_min: minimum green time
    """

    def __init__(self, traffic_signal, theta=3, mu=4, phi_min=5):
        super(SOTLPhaseSelector, self).__init__(traffic_signal)
        self.mu = mu
        self.theta = theta
        self.phi_min = phi_min
        self.last_vehicle_time = {}
        self.controller = traffic_signal.controller
        EnvLogger.log_info(
            "SOTL parameters:\n"
            + f"\nmu: {self.mu} | phi = {self.phi_min} | theta: {self.theta}"
        )

    def get_action(self, inp):
        action_mask = self.controller.get_allowable_phase_switches()
        if action_mask[self.controller.current_phase_index]:
            red_flow = self._compute_flow_for_phase(
                inp, self.controller.next_green_phase_index
            )
            green_flow = self._compute_flow_for_phase(
                inp, self.controller.current_phase_index
            )
            if (
                self.controller.time_on_phase >= self.phi_min
                and not 0 < green_flow < self.mu
                and red_flow >= self.theta
            ):
                return self.controller.next_phase_index
            else:
                return self.controller.current_phase_index
        else:
            return self.controller.next_phase_index

    def _compute_flow_for_phase(self, inp, phase_index):
        total_vehicles = 0
        phase = self.traffic_signal.config["phases"][phase_index]
        phase_inc_out_lanes = self.traffic_signal.config[
            "phase_to_inc_out_lanes"
        ][phase]
        for inc_lane in phase_inc_out_lanes.keys():
            total_vehicles += len(
                inp["lane"][inc_lane]["position_speed_matrices"][0]
            )
        return total_vehicles

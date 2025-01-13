from abc import ABC

import logging

import numpy as np

from pytsc.common.utils import EnvLogger

# EnvLogger.set_log_level(logging.WARNING)


class BasePhaseSelector(ABC):
    def __init__(self, traffic_signal, round_robin=True, **kwargs):
        self.traffic_signal = traffic_signal
        self.round_robin = round_robin
        self.visibility = traffic_signal.config["visibility"]

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.traffic_signal.id})"


class FixedTimePhaseSelector(BasePhaseSelector):
    def __init__(self, traffic_signal, green_time=25):
        super(FixedTimePhaseSelector, self).__init__(traffic_signal)
        self.green_time = green_time
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        if (
            self.controller.current_phase_index in self.controller.green_phase_indices
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
        if self.controller.current_phase_index in self.controller.green_phase_indices:
            for act, available in enumerate(action_mask):
                if available:
                    queue = self._compute_queue_for_phase(inp, act)
                else:
                    queue = float("-inf")
                queues.append((queue, act))
            max_queue_value = max(queues, key=lambda x: x[0])[0]
            tied_actions = [act for queue, act in queues if queue == max_queue_value]
            max_queue_phase_index = np.random.choice(tied_actions)
        else:
            max_queue_phase_index = self.controller.next_phase_index
        return max_queue_phase_index

    def _compute_queue_for_phase(self, inp, phase_index):
        inc_vehicles = 0
        phase = self.traffic_signal.config["phases"][phase_index]
        phase_inc_out_lanes = self.traffic_signal.config["phase_to_inc_out_lanes"][
            phase
        ]
        for inc_lane in phase_inc_out_lanes.keys():
            _, speed_mat = inp["lane"][inc_lane]["position_speed_matrices"]
            mean_speed = sum(speed_mat[: self.visibility]) / self.visibility
            n_queued = 1 - mean_speed
            inc_vehicles += n_queued
        return inc_vehicles


class MaxPressurePhaseSelector(BasePhaseSelector):
    def __init__(self, traffic_signal):
        super(MaxPressurePhaseSelector, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        action_mask = self.controller.get_allowable_phase_switches()
        pressures = []
        if self.controller.current_phase_index in self.controller.green_phase_indices:
            for act, available in enumerate(action_mask):
                if available:
                    pressure = self._compute_pressure_for_phase(inp, act)
                else:
                    pressure = float("-inf")
                pressures.append((pressure, act))
            max_pressure_value = max(pressures, key=lambda x: x[0])[0]
            tied_actions = [
                act for pressure, act in pressures if pressure == max_pressure_value
            ]
            max_pressure_phase_index = np.random.choice(tied_actions)
        else:
            max_pressure_phase_index = self.controller.next_phase_index
        return max_pressure_phase_index

    def _compute_pressure_for_phase(self, inp, phase_index):
        """
        inp (dict): network.simulator.step_measurements["lane"]
        """
        pressure = 0
        phase = self.traffic_signal.config["phases"][phase_index]
        phase_inc_out_lanes = self.traffic_signal.config["phase_to_inc_out_lanes"][
            phase
        ]
        for inc_lane, out_lanes in phase_inc_out_lanes.items():
            inc_pos_mat, _ = inp["lane"][inc_lane]["position_speed_matrices"]
            inc_lane_vehicles = sum(inc_pos_mat[: self.visibility])
            out_lane_vehicles = 0
            for out_lane in out_lanes:
                out_pos_mat, _ = inp["lane"][out_lane]["position_speed_matrices"]
                out_lane_vehicles = sum(out_pos_mat[-self.visibility :])
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
        # EnvLogger.log_info(
        #     "SOTL parameters:\n"
        #     + f"\nmu: {self.mu} | phi = {self.phi_min} | theta: {self.theta}"
        # )

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
        phase_inc_out_lanes = self.traffic_signal.config["phase_to_inc_out_lanes"][
            phase
        ]
        for inc_lane in phase_inc_out_lanes.keys():
            inc_pos_mat, _ = inp["lane"][inc_lane]["position_speed_matrices"]
            total_vehicles += sum(inc_pos_mat[: self.visibility])
        return total_vehicles


class AnalyticPlusPhaseSelector(BasePhaseSelector):
    def __init__(self, traffic_signal):
        super(AnalyticPlusPhaseSelector, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        action_mask = self.controller.get_allowable_phase_switches()
        pressures = []
        if self.controller.current_phase_index in self.controller.green_phase_indices:
            for act, available in enumerate(action_mask):
                if available:
                    pressure = self._compute_pressure_for_phase(inp, act)
                else:
                    pressure = float("-inf")
                pressures.append((pressure, act))
            max_pressure_value = max(pressures, key=lambda x: x[0])[0]
            tied_actions = [
                act for pressure, act in pressures if pressure == max_pressure_value
            ]
            max_pressure_phase_index = np.random.choice(tied_actions)
        else:
            max_pressure_phase_index = self.controller.next_phase_index
        return max_pressure_phase_index

    def _compute_pressure_for_phase(self, inp, phase_index):
        pressure = 0
        phase = self.traffic_signal.config["phases"][phase_index]
        phase_inc_out_lanes = self.traffic_signal.config["phase_to_inc_out_lanes"][
            phase
        ]
        for inc_lane, out_lanes in phase_inc_out_lanes.items():
            inc_lane_vehicles = inp["lane"][inc_lane]["occupancy"]
            out_lane_vehicles = 0
            for out_lane in out_lanes:
                out_lane_vehicles += inp["lane"][out_lane]["occupancy"]
            pressure += np.abs(inc_lane_vehicles - out_lane_vehicles)
        return pressure


class RandomPhaseSelector(BasePhaseSelector):
    def __init__(self, traffic_signal):
        super(RandomPhaseSelector, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        action_mask = self.controller.get_allowable_phase_switches()
        available_actions = np.where(action_mask)[0]
        return np.random.choice(available_actions)

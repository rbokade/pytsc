from abc import ABC

import numpy as np


class BaseController(ABC):
    """
    Base class for traffic signal controllers.
    This class defines the interface for different traffic signal controllers
    and provides common functionality for controller management.
    Args:
        traffic_signal (TrafficSignal): Traffic signal object containing simulation parameters and network information.
        round_robin (bool): Flag to indicate if round-robin scheduling is used.
        **kwargs: Additional keyword arguments for controller configuration.
    """
    def __init__(self, traffic_signal, round_robin=True, **kwargs):
        self.traffic_signal = traffic_signal
        self.round_robin = round_robin
        self.visibility = traffic_signal.config["visibility"]

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.traffic_signal.id})"


class FixedTimeController(BaseController):
    """
    Fixed Time Controller.
    Args:
        traffic_signal (TrafficSignal): Traffic signal object containing simulation parameters and network information.
        green_time (int): Duration of the green phase in seconds.
    """
    def __init__(self, traffic_signal, green_time=25):
        super(FixedTimeController, self).__init__(traffic_signal)
        self.green_time = green_time
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        """
        Select the next action based on fixed time intervals.
        Args:
            inp (dict): Input data containing network and traffic signal information.
        Returns:
            int: Index of the selected action.
        """
        if (
            self.controller.current_phase_index in self.controller.green_phase_indices
            and self.controller.time_on_phase < self.green_time
        ):
            return self.controller.current_phase_index
        else:
            return self.controller.next_phase_index


class GreedyController(BaseController):
    """
    Greedy Controller.
    Args:
        traffic_signal (TrafficSignal): Traffic signal object containing simulation parameters and network information.
    """
    def __init__(self, traffic_signal):
        super(GreedyController, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        """
        Select the next action based on the greedy algorithm.
        Args:
            inp (dict): Input data containing network and traffic signal information.
        Returns:
            int: Index of the selected action.
        """
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
        """
        Compute the queue for a given phase.
        Args:
            phase_index (int): Index of the phase for which to compute the queue.
            inp (dict): network.simulator.step_measurements["lane"]
        Returns:
            int: Number of vehicles in the queue for the given phase.
        """
        inc_vehicles = 0
        phase = self.traffic_signal.config["phases"][phase_index]
        phase_inc_out_lanes = self.traffic_signal.config["phase_to_inc_out_lanes"][
            phase
        ]
        for inc_lane in phase_inc_out_lanes.keys():
            pos_mat = inp["lane"][inc_lane]["position_matrix"][-self.visibility :]
            n_queued = sum([p == 0.0 for p in pos_mat])
            inc_vehicles += n_queued
        return inc_vehicles


class MaxPressureController(BaseController):
    """
    Max Pressure Controller.
    Args:
        traffic_signal (TrafficSignal): Traffic signal object containing simulation parameters and network information.
    """
    def __init__(self, traffic_signal):
        super(MaxPressureController, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        """
        Select the next action based on the max pressure algorithm.
        Args:
            inp (dict): Input data containing network and traffic signal information.
        Returns:
            int: Index of the selected action.
        """
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
        Compute the pressure for a given phase.
        Args:
            phase_index (int): Index of the phase for which to compute pressure.
            inp (dict): network.simulator.step_measurements["lane"]
        Returns:
            float: Pressure for the given phase.
        """
        pressure = 0
        phase = self.traffic_signal.config["phases"][phase_index]
        phase_inc_out_lanes = self.traffic_signal.config["phase_to_inc_out_lanes"][
            phase
        ]
        for inc_lane, out_lanes in phase_inc_out_lanes.items():
            inc_pos_mat = inp["lane"][inc_lane]["position_matrix"][-self.visibility :]
            inc_lane_vehicles = sum([p >= 0.0 for p in inc_pos_mat])
            out_lane_vehicles = 0
            for out_lane in out_lanes:
                out_pos_mat = inp["lane"][out_lane]["position_matrix"][
                    : self.visibility
                ]
                out_lane_vehicles = sum([p >= 0.0 for p in out_pos_mat])
            pressure += np.abs(inc_lane_vehicles - out_lane_vehicles)
        return pressure


class SOTLController(BaseController):
    """
    mu: threshold for vehicles on green phase lanes
    theta: threshold for vehicles on red phase lanes
    phi_min: minimum green time
    """

    def __init__(self, traffic_signal, theta=3, mu=4, phi_min=5):
        super(SOTLController, self).__init__(traffic_signal)
        self.mu = mu
        self.theta = theta
        self.phi_min = phi_min
        self.last_vehicle_time = {}
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        """
        Select the next action based on the SOTL algorithm.
        Args:
            inp (dict): Input data containing network and traffic signal information.
        """
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
        """
        Calculate the flow for a given phase.
        Args::
            inp (dict): network.simulator.step_measurements["lane"]
            phase_index (int): Index of the phase for which to compute the flow.
        """
        total_vehicles = 0
        phase = self.traffic_signal.config["phases"][phase_index]
        phase_inc_out_lanes = self.traffic_signal.config["phase_to_inc_out_lanes"][
            phase
        ]
        for inc_lane in phase_inc_out_lanes.keys():
            inc_pos_mat = inp["lane"][inc_lane]["position_matrix"][-self.visibility :]
            total_vehicles += sum([p >= 0.0 for p in inc_pos_mat])
        return total_vehicles


class RandomController(BaseController):
    """
    Randomly selects an action from the available actions.
    
    Args:
        traffic_signal (TrafficSignal): Traffic signal object containing simulation parameters and network information.
    """
    def __init__(self, traffic_signal):
        super(RandomController, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller

    def get_action(self, inp):
        """
        Select a random action from the available actions.

        Args:
            inp (dict): Input data containing network and traffic signal information.

        Returns:
            int: Index of the selected action.
        """
        action_mask = self.controller.get_allowable_phase_switches()
        available_actions = np.where(action_mask)[0]
        return np.random.choice(available_actions)

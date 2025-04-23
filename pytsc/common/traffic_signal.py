from abc import ABC, abstractmethod

import pandas as pd

from pytsc.controllers import (
    FixedTimeController,
    GreedyController,
    MaxPressureController,
    SOTLController,
)


class BaseTrafficSignal(ABC):
    """
    Base class for traffic signals in the network.
    Args:
        id (str): Unique identifier for the traffic signal.
        config (dict): Configuration dictionary containing traffic signal parameters.
        simulator (Simulator): Simulator object containing simulation parameters and network information.
    """
    def __init__(self, id, config, simulator):
        self.id = id
        self.config = config
        self.simulator = simulator
        self.n_phases = config["n_phases"]

    def __repr__(self):
        return f"TrafficSignal ({self.id})"

    @abstractmethod
    def update_stats(self):
        """
        Update the statistics of the traffic signal.
        """
        raise NotImplementedError

    @abstractmethod
    def action_to_phase(self, action):
        """
        Convert action to phase index.
        Args:
            action (int): Action index.
        """
        raise NotImplementedError

    def init_rule_based_controllers(self):
        """
        Initialize rule-based controllers for the traffic signal.
        """
        self.controllers = {
            "fixed_time": FixedTimeController(self),
            "greedy": GreedyController(self),
            "max_pressure": MaxPressureController(self),
            "sotl": SOTLController(self),
        }

    # def get_controller_action(self, sub_results):
    #     return {k: v.get_action(sub_results) for k, v in self.controllers.items()}


class BaseTSProgram(ABC):
    """
    Base class for traffic signal programs.
    Args:
        id (str): Unique identifier for the traffic signal program.
        config (dict): Configuration dictionary containing traffic signal parameters.
        simulator (Simulator): Simulator object containing simulation parameters and network information.
    """
    def __init__(self, id, config, simulator):
        self.id = id
        self.phases = config["phases"]
        self.phase_indices = config["phase_indices"]
        self.phases_min_max_times = config["phases_min_max_times"]
        self.yellow_time = config["yellow_time"]

    @abstractmethod
    def _initialize_traffic_light_program(self):
        """
        Initialize the traffic light program.
        """
        raise NotImplementedError

    def set_initial_phase(self, phase_index):
        """
        Set the initial phase of the traffic signal program.
        Args:
            phase_index (int): Index of the initial phase.
        """        
        self.current_phase_index = phase_index
        self.current_phase = self.phases[phase_index]
        self.time_on_phase = 0
        self.norm_time_on_phase = 0

    def update_current_phase(self, phase_index):
        """
        Update the current phase of the traffic signal program.
        Args:
            phase_index (int): Index of the new phase.
        """        
        if phase_index == self.current_phase_index:
            self.phase_changed = False
            self.time_on_phase += self.yellow_time
        else:
            self.phase_changed = True
            self.time_on_phase = self.yellow_time
        self.current_phase_index = phase_index
        self.current_phase = self.phases[phase_index]
        max_phase_time = self.phases_min_max_times[self.current_phase]["max_time"]
        self.norm_time_on_phase = self.time_on_phase / max_phase_time

    def __repr__(self):
        return f"{self.__class__} ({self.id})"


class BaseTSController(ABC):
    """
    Base class for traffic signal controllers.
    Args:
        id (str): Unique identifier for the traffic signal controller.
        config (dict): Configuration dictionary containing traffic signal parameters.
        simulator (Simulator): Simulator object containing simulation parameters and network information.
    """    
    def __init__(self, id, config, simulator):
        self.id = id
        self.config = config
        self.simulator = simulator
        self.program = None

    @property
    def n_phases(self):
        """
        Get the number of phases in the traffic signal controller.
        Returns:
            int: Number of phases in the traffic signal controller.
        """        
        return self.config["n_phases"]

    @property
    def phase_indices(self):
        """
        Get the phase indices of the traffic signal controller.
        Returns:
            list: List of phase indices in the traffic signal controller.
        """        
        return self.config["phase_indices"]

    @property
    def green_phase_indices(self):
        """
        Get the green phase indices of the traffic signal controller.
        Returns:
            list: List of green phase indices in the traffic signal controller.
        """        
        return self.config["green_phase_indices"]

    @property
    def yellow_time(self):
        """
        Get the yellow time of the traffic signal controller.
        Returns:
            int: Yellow time of the traffic signal controller.
        """        
        return self.config["yellow_time"]

    @property
    def yellow_phase_indices(self):
        """
        Get the yellow phase indices of the traffic signal controller.
        Returns:
            list: List of yellow phase indices in the traffic signal controller.
        """        
        return self.config["yellow_phase_indices"]

    @property
    def phases_min_max_times(self):
        """
        Get the minimum and maximum times for each phase in the traffic signal controller.
        Returns:
            dict: Dictionary containing minimum and maximum times for each phase.
        """        
        return self.config["phases_min_max_times"]

    @property
    def current_phase(self):
        """
        Get the current phase of the traffic signal controller.
        Returns:
            str: Current phase of the traffic signal controller.
        """        
        if self.program is not None:
            return self.program.current_phase
        else:
            return None

    @property
    def current_phase_index(self):
        """
        Get the index of the current phase in the traffic signal controller.
        Returns:
            int: Index of the current phase in the traffic signal controller.
        """        
        if self.program is not None:
            return self.program.current_phase_index
        else:
            return None

    @property
    def next_phase_index(self):
        """
        Get the index of the next phase in the traffic signal controller.
        Returns:
            int: Index of the next phase in the traffic signal controller.
        """        
        if self.current_phase_index is not None:
            return (self.current_phase_index + 1) % self.n_phases
        else:
            return None

    @property
    def next_green_phase_index(self):
        """
        Get the index of the next green phase in the traffic signal controller.
        Returns:
            int: Index of the next green phase in the traffic signal controller.
        """        
        if self.current_phase_index is not None:
            return (self.current_phase_index + 2) % self.n_phases
        else:
            return None

    @property
    def time_on_phase(self):
        """
        Get the time spent on the current phase of the traffic signal controller.
        Returns:
            int: Time spent on the current phase of the traffic signal controller.
        """        
        if self.program is not None:
            return self.program.time_on_phase
        else:
            return None

    @property
    def norm_time_on_phase(self):
        """
        Get the normalized time spent on the current phase of the traffic signal controller.
        Returns:
            float: Normalized time spent on the current phase of the traffic signal controller.
        """        
        if self.program is not None:
            return self.program.norm_time_on_phase
        else:
            return None

    @property
    def phase_one_hot(self):
        """
        Get the one-hot encoding of the current phase of the traffic signal controller.
        Returns:
            list: One-hot encoding of the current phase of the traffic signal controller.
        """        
        phase_one_hot = [0 for _ in range(self.n_phases)]
        if self.current_phase_index is not None:
            phase_one_hot[self.current_phase_index] = 1
        return phase_one_hot

    def _instantiate_traffic_light_logic(self):
        """
        Initialize the traffic light logic based on the configuration.
        """        
        if self.config["round_robin"]:
            self.logic = TLSRoundRobinPhaseSelectLogic(self)
        else:
            self.logic = TLSFreePhaseSelectLogic(self)

    def get_allowable_phase_switches(self):
        """
        Get the allowable phase switches based on the current phase and time spent on it.
        Returns:
            list: List of allowable phase switches for the traffic signal controller.
        """        
        return self.logic.get_allowable_phase_switches(self.time_on_phase)

    @abstractmethod
    def switch_phase(self):
        """
        Switch the phase of the traffic signal controller.
        """        
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__} ({self.id})"


class TLSFreePhaseSelectLogic:
    """
    phase_dict: {phase_idx: {"min_time": int, "max_time": int}}}
    NOTES:
    - The phase indices must be ordered such that the phase list
    starts with a green phase and every green phase must be followed
    by a yellow phase.
    - Step size is fixed to be 5 seconds.
    """

    def __init__(self, controller):
        self.id = controller.id
        self.controller = controller
        # self.phases = controller.phases
        self.n_phases = controller.n_phases
        self.phase_indices = controller.phase_indices
        self.green_phase_indices = controller.green_phase_indices
        self.yellow_phase_indices = controller.yellow_phase_indices
        self.phases_min_max_times = controller.phases_min_max_times

    def get_allowable_phase_switches(self, time_on_phase):
        """
        Get the allowable phase switches based on the current phase and time spent on it.
        Args:
            time_on_phase (int): Time spent on the current phase.
        Returns:
            list: List of allowable phase switches for the traffic signal controller.
        """        
        mask = [0 for _ in range(self.n_phases)]
        if self.controller.current_phase_index in self.controller.green_phase_indices:
            min_max_times = self.phases_min_max_times[self.controller.current_phase]
            min_time = min_max_times["min_time"]
            max_time = min_max_times["max_time"]
            if time_on_phase < min_time:  # stay on current phase
                mask[self.controller.current_phase_index] = 1
                return mask
            elif (
                time_on_phase >= min_time and time_on_phase < max_time
            ):  # stay on the same phase or switch to the corr. yellow phase
                mask[self.controller.current_phase_index] = 1
                mask[self.controller.next_phase_index] = 1
            elif time_on_phase == max_time:  # switch to corr. yellow phase
                mask[self.controller.next_phase_index] = 1
            else:
                breakpoint()  # should never reach here
        else:
            for i in self.green_phase_indices:
                # Can switch to any green phases except the prev. green
                if i != self.controller.current_phase_index - 1:
                    mask[i] = 1
        return mask


class TLSRoundRobinPhaseSelectLogic(TLSFreePhaseSelectLogic):
    """
    Round robin phase selection logic for traffic signal controllers.
    Args:
        controller (BaseTSController): Traffic signal controller object.        
    """    
    def __init__(self, controller):
        super(TLSRoundRobinPhaseSelectLogic, self).__init__(controller)
        self.yellow_time = controller.yellow_time

    def get_allowable_phase_switches(self, time_on_phase):
        """
        Get the allowable phase switches based on the current phase and time spent on it.
        Args:
            time_on_phase (int): Time spent on the current phase.
        Returns:
            list: List of allowable phase switches for the traffic signal controller.
        """        
        mask = [0 for _ in range(self.n_phases)]
        if self.controller.current_phase_index in self.green_phase_indices:
            min_max_times = self.phases_min_max_times[self.controller.current_phase]
            min_time = min_max_times["min_time"]
            max_time = min_max_times["max_time"]
            if time_on_phase < min_time:  # stay on current phase
                mask[self.controller.current_phase_index] = 1
                return mask
            elif time_on_phase >= min_time and time_on_phase < max_time:
                # stay on the same phase or switch to the corr. yellow phase
                mask[self.controller.current_phase_index] = 1
                mask[self.controller.next_phase_index] = 1
            elif time_on_phase == max_time:
                # switch to corr. yellow phase
                mask[self.controller.next_phase_index] = 1
            else:
                breakpoint()  # should never reach here
        else:
            mask[self.controller.next_phase_index] = 1
        return mask

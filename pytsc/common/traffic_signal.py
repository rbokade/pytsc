from abc import ABC, abstractmethod

from pytsc.controllers import (
    FixedTimePhaseSelector,
    GreedyPhaseSelector,
    MaxPressurePhaseSelector,
    SOTLPhaseSelector,
)


class BaseTrafficSignal(ABC):
    def __init__(self, id, config, simulator):
        self.id = id
        self.config = config
        self.simulator = simulator
        self.n_phases = config["n_phases"]

    def __repr__(self):
        return f"TrafficSignal ({self.id})"

    @abstractmethod
    def update_stats(self):
        raise NotImplementedError

    @abstractmethod
    def action_to_phase(self, action):
        raise NotImplementedError

    def init_rule_based_controllers(self):
        self.controllers = {
            "fixed_time": FixedTimePhaseSelector(self),
            "greedy": GreedyPhaseSelector(self),
            "max_pressure": MaxPressurePhaseSelector(self),
            "sotl": SOTLPhaseSelector(self),
        }

    def get_controller_actions(self, sub_results):
        return {k: v.get_action(sub_results) for k, v in self.controllers.items()}


class BaseTSProgram(ABC):
    def __init__(self, id, config, simulator):
        self.id = id
        self.phases = config["phases"]
        self.phase_indices = config["phase_indices"]
        self.phases_min_max_times = config["phases_min_max_times"]
        self.yellow_time = config["yellow_time"]

    @abstractmethod
    def _initialize_traffic_light_program(self):
        raise NotImplementedError

    def set_initial_phase(self, phase_index):
        self.current_phase_index = phase_index
        self.current_phase = self.phases[phase_index]
        self.time_on_phase = 0
        self.norm_time_on_phase = 0

    def update_current_phase(self, phase_index):
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
    def __init__(self, id, config, simulator):
        self.id = id
        self.config = config
        self.simulator = simulator
        self.program = None

    @property
    def n_phases(self):
        return self.config["n_phases"]

    @property
    def phase_indices(self):
        return self.config["phase_indices"]

    @property
    def green_phase_indices(self):
        return self.config["green_phase_indices"]

    @property
    def yellow_time(self):
        return self.config["yellow_time"]

    @property
    def yellow_phase_indices(self):
        return self.config["yellow_phase_indices"]

    @property
    def phases_min_max_times(self):
        return self.config["phases_min_max_times"]

    @property
    def current_phase(self):
        if self.program is not None:
            return self.program.current_phase
        else:
            return None

    @property
    def current_phase_index(self):
        if self.program is not None:
            return self.program.current_phase_index
        else:
            return None

    @property
    def next_phase_index(self):
        if self.current_phase_index is not None:
            return (self.current_phase_index + 1) % self.n_phases
        else:
            return None

    @property
    def next_green_phase_index(self):
        if self.current_phase_index is not None:
            return (self.current_phase_index + 2) % self.n_phases
        else:
            return None

    @property
    def time_on_phase(self):
        if self.program is not None:
            return self.program.time_on_phase
        else:
            return None

    @property
    def norm_time_on_phase(self):
        if self.program is not None:
            return self.program.norm_time_on_phase
        else:
            return None

    @property
    def phase_one_hot(self):
        phase_one_hot = [0 for _ in range(self.n_phases)]
        if self.current_phase_index is not None:
            phase_one_hot[self.current_phase_index] = 1
        return phase_one_hot

    def _instantiate_traffic_light_logic(self):
        if getattr(self.config, "round_robin", False):
            self.logic = TLSFreePhaseSelectLogic(self)
        else:
            self.logic = TLSRoundRobinPhaseSelectLogic(self)

    def get_allowable_phase_switches(self):
        return self.logic.get_allowable_phase_switches(self.time_on_phase)

    @abstractmethod
    def switch_phase(self):
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
    def __init__(self, controller):
        super(TLSRoundRobinPhaseSelectLogic, self).__init__(controller)
        self.yellow_time = controller.yellow_time

    def get_allowable_phase_switches(self, time_on_phase):
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

from abc import ABC, abstractmethod


class BaseTSProgram(ABC):
    def __init__(self, id, config, simulator):
        for k, v in config.items():
            setattr(self, k, v)
        self.id = id

    @abstractmethod
    def _initialize_traffic_light_program(self):
        raise NotImplementedError

    def update_current_phase(self, phase_index):
        self.current_phase = self.phases[phase_index]
        self.current_phase_index = phase_index

    def __repr__(self):
        return f"{self.__class__} ({self.id})"


class BaseTSController(ABC):
    def __init__(self, id, config, simulator):
        for k, v in config.items():
            setattr(self, k, v)
        self.id = id
        self.time_on_phase = 0
        self.norm_time_on_phase = 0
        self.time_on_cycle = 0  # only applicable for round robin
        self.norm_time_on_cycle = 0
        self.last_cycle_length = 0  # only applicable for round robin
        self.phase_changed = False
        self.phase_history = []
        if getattr(self, "max_cycle_length", True):
            self.max_cycle_length = sum(
                v["max_time"] for v in self.phases_min_max_times.values()
            )

    @property
    def phase(self):
        return self.program.current_phase

    @property
    def phase_one_hot(self):
        phase_one_hot = [0 for _ in range(self.n_phases)]
        phase_one_hot[self.program.current_phase_index] = 1
        return phase_one_hot

    def _instantiate_traffic_light_logic(self):
        if getattr(self, "round_robin", False):
            self.logic = TLSRoundRobinPhaseSelectLogic(self)
        else:
            self.logic = TLSFreePhaseSelectLogic(self)
        self.logic.update_current_phase_index(
            self.program.current_phase_index, self.time_on_phase
        )
        self.phase_history.append(self.program.current_phase_index)

    def _update_cycle_time(self):
        self.time_on_cycle += self.yellow_time
        self.norm_time_on_cycle += self.yellow_time / self.max_cycle_length

    def _update_phase_time(self, phase_index):
        self.phase_history.append(self.program.current_phase_index)
        if phase_index == self.program.current_phase_index:
            self.phase_changed = False
            self.time_on_phase += self.yellow_time
            self.norm_time_on_phase += self.yellow_time / self.max_cycle_length
        else:
            if phase_index == 0:
                self.last_cycle_length = self.time_on_cycle
                self.time_on_cycle = 0
                self.norm_time_on_cycle = 0
            self.phase_changed = True
            self.time_on_phase = self.yellow_time
            self.norm_time_on_phase = self.yellow_time / self.max_cycle_length

    def get_allowable_phase_switches(self):
        return self.logic.get_allowable_phase_switches(
            self.time_on_phase, time_on_cycle=self.time_on_cycle
        )

    @abstractmethod
    def switch_phase(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__} ({self.id})"


class BaseTrafficSignal(ABC):
    def __init__(self, id, config, simulator):
        for k, v in config.items():
            setattr(self, k, v)
        self.id = id
        self.simulator = simulator

    def __repr__(self):
        return f"TrafficSignal ({self.id})"

    @abstractmethod
    def update_stats(self):
        raise NotImplementedError

    @abstractmethod
    def action_to_phase(self, action):
        raise NotImplementedError


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
        self.phases = controller.phases
        self.n_phases = controller.n_phases
        self.green_phase_indices = controller.green_phase_indices
        self.yellow_phase_indices = controller.yellow_phase_indices
        self.phases_min_max_times = controller.phases_min_max_times
        self.update_current_phase_index(
            controller.program.start_phase_index, 0
        )
        max_total_duration = sum(
            v["max_time"] for v in self.phases_min_max_times.values()
        )
        if getattr(controller, "max_cycle_length", True):
            self.max_cycle_length = max_total_duration
        else:
            min_total_duration = sum(
                v["min_time"] for v in self.phases_min_max_times.values()
            )
            assert (
                self.max_cycle_length % 5 == 0
            ), "Cycle length must be a multiple of 5"
            assert (
                self.max_cycle_length >= min_total_duration
                and self.max_cycle_length <= max_total_duration
            ), "Cycle length must <= the sum of all phase durations"

    def update_current_phase_index(self, current_phase_index, time_on_phase):
        self.current_phase = self.phases[current_phase_index]
        self.current_phase_index = current_phase_index
        self.time_on_phase = time_on_phase

    def get_allowable_phase_switches(self, time_on_phase, **kwargs):
        mask = [0 for _ in range(self.n_phases)]
        if self.current_phase_index in self.green_phase_indices:
            min_time = self.phases_min_max_times[self.current_phase][
                "min_time"
            ]
            max_time = self.phases_min_max_times[self.current_phase][
                "max_time"
            ]
            if time_on_phase < min_time:  # stay on current phase
                mask[self.current_phase_index] = 1
                return mask
            elif (
                time_on_phase >= min_time and time_on_phase < max_time
            ):  # stay on the same phase or switch to the corr. yellow phase
                mask[self.current_phase_index] = 1
                mask[self.current_phase_index + 1] = 1
            elif time_on_phase == max_time:  # switch to corr. yellow phase
                mask[self.current_phase_index + 1] = 1
            else:
                breakpoint()  # should never reach here
        else:
            for i in self.green_phase_indices:
                # Can switch to any green phases except the prev. green
                if i != self.current_phase_index - 1:
                    mask[i] = 1
        return mask


class TLSRoundRobinPhaseSelectLogic(TLSFreePhaseSelectLogic):
    def __init__(self, controller):
        super(TLSRoundRobinPhaseSelectLogic, self).__init__(controller)
        self.yellow_time = controller.yellow_time

    def get_allowable_phase_switches(self, time_on_phase, time_on_cycle):
        max_cycle_length_trimmed = self.max_cycle_length - self.yellow_time
        if self.current_phase_index in self.green_phase_indices:
            min_time = self.phases_min_max_times[self.current_phase][
                "min_time"
            ]
            max_time = self.phases_min_max_times[self.current_phase][
                "max_time"
            ]
            mask = [0 for _ in range(self.n_phases)]
            if time_on_phase < min_time:  # stay on current phase
                mask[self.current_phase_index] = 1
                return mask
            elif (
                time_on_phase >= min_time
                and time_on_phase < max_time
                and time_on_cycle < max_cycle_length_trimmed
            ):  # stay on the same phase or switch to the corr. yellow phase
                mask[self.current_phase_index] = 1
                mask[self.current_phase_index + 1] = 1
            elif (
                time_on_phase == max_time
                or time_on_cycle == max_cycle_length_trimmed
            ):  # switch to corr. yellow phase
                mask[self.current_phase_index + 1] = 1
            else:
                breakpoint()  # should never reach here
        else:
            mask = [0 for _ in range(self.n_phases)]
            mask[(self.current_phase_index + 1) % self.n_phases] = 1
        return mask

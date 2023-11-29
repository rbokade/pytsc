from abc import ABC, abstractmethod


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


class BaseTSProgram(ABC):
    def __init__(self, id, config, simulator):
        for k, v in config.items():
            setattr(self, k, v)
        self.id = id
        (
            self.min_cycle_length,
            self.max_cycle_length,
            self.cycle_lengths,
        ) = self._get_cycle_lengths()
        self.phase_history, self.cycle_length_history = [], []

    def _get_cycle_lengths(self):
        """
        NOTE: Looping over phase indices because a yellow phase might
        be common for all green phases (CityFlow).
        """
        min_cycle_length, max_cycle_length = 0, 0
        for phase_index in self.phase_indices:
            phase = self.phases[phase_index]
            min_time = self.phases_min_max_times[phase]["min_time"]
            max_time = self.phases_min_max_times[phase]["max_time"]
            min_cycle_length += min_time
            max_cycle_length += max_time
        cycle_lengths = [i for i in range(min_cycle_length, 240 + 5, 5)]
        return min_cycle_length, max_cycle_length, cycle_lengths

    @abstractmethod
    def _initialize_traffic_light_program(self):
        raise NotImplementedError

    def set_initial_phase(self, phase_index):
        self.current_phase_index = phase_index
        self.current_phase = self.phases[phase_index]
        self.time_on_phase = 0
        self.norm_time_on_phase = 0
        self.phase_history.append(phase_index)

    def set_initial_cycle_length(self, cycle_length_index):
        self.current_cycle_length_index = cycle_length_index
        self.current_cycle_length = self.cycle_lengths[cycle_length_index]
        self.time_on_cycle = 0
        self.norm_time_on_cycle = 0
        self.cycle_length_history.append(cycle_length_index)

    def update_current_phase(self, phase_index):
        if phase_index == self.current_phase_index:
            self.phase_changed = False
            self.time_on_phase += self.yellow_time
        else:
            self.phase_changed = True
            self.time_on_phase = self.yellow_time
        self.current_phase_index = phase_index
        self.current_phase = self.phases[phase_index]
        max_phase_time = self.phases_min_max_times[self.current_phase][
            "max_time"
        ]
        self.norm_time_on_phase = self.time_on_phase / max_phase_time
        self.phase_history.append(phase_index)

    def update_current_cycle_length(self, cycle_length_index):
        if self.time_on_cycle < self.current_cycle_length:
            # if cycle_length_index == self.current_cycle_length_index:
            self.cycle_length_changed = False
            self.time_on_cycle += self.yellow_time
        else:
            self.cycle_length_changed = True
            self.time_on_cycle = self.yellow_time
        self.current_cycle_length_index = cycle_length_index
        self.current_cycle_length = self.cycle_lengths[cycle_length_index]
        self.norm_time_on_cycle = (
            self.time_on_cycle / self.current_cycle_length
        )
        self.cycle_length_history.append(cycle_length_index)

    def __repr__(self):
        return f"{self.__class__} ({self.id})"


class BaseTSController(ABC):
    def __init__(self, id, config, simulator):
        for k, v in config.items():
            setattr(self, k, v)
        self.id = id

    @property
    def current_phase(self):
        return self.program.current_phase

    @property
    def current_phase_index(self):
        return self.program.current_phase_index

    @property
    def next_phase_index(self):
        return (self.current_phase_index + 1) % self.n_phases

    @property
    def time_on_phase(self):
        return self.program.time_on_phase

    @property
    def norm_time_on_phase(self):
        return self.program.norm_time_on_phase

    @property
    def current_cycle_length(self):
        return self.program.current_cycle_length

    @property
    def current_cycle_length_index(self):
        return self.program.current_cycle_length_index

    @property
    def time_on_cycle(self):
        return self.program.time_on_cycle

    @property
    def norm_time_on_cycle(self):
        return self.program.norm_time_on_cycle

    @property
    def remaining_time_on_cycle(self):
        return self.current_cycle_length - self.time_on_cycle

    @property
    def phase_one_hot(self):
        phase_one_hot = [0 for _ in range(self.n_phases)]
        phase_one_hot[self.current_phase_index] = 1
        return phase_one_hot

    def _instantiate_traffic_light_logic(self):
        if getattr(self, "round_robin", False):
            if getattr(self, "action_space") == "cycle_length_and_phase":
                self.logic = TLSRoundRobinCycleAndPhaseSelectLogic(self)
            else:
                self.logic = TLSRoundRobinPhaseSelectLogic(self)
        else:
            self.logic = TLSFreePhaseSelectLogic(self)

    def get_allowable_phase_switches(self):
        return self.logic.get_allowable_phase_switches(self.time_on_phase)

    def get_allowable_cycle_length_switches(self):
        return self.logic.get_allowable_cycle_length_switches(
            self.time_on_cycle
        )

    @abstractmethod
    def switch_phase(self):
        raise NotImplementedError

    @abstractmethod
    def switch_cycle_length(self):
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
        self.phases = controller.phases
        self.n_phases = controller.n_phases
        self.phase_indices = controller.phase_indices
        self.green_phase_indices = controller.green_phase_indices
        self.yellow_phase_indices = controller.yellow_phase_indices
        self.phases_min_max_times = controller.phases_min_max_times

    def get_allowable_phase_switches(self, time_on_phase):
        mask = [0 for _ in range(self.n_phases)]
        if (
            self.controller.current_phase_index
            in self.controller.green_phase_indices
        ):
            min_max_times = self.phases_min_max_times[
                self.controller.current_phase
            ]
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
            min_max_times = self.phases_min_max_times[
                self.controller.current_phase
            ]
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


class TLSRoundRobinCycleAndPhaseSelectLogic(TLSRoundRobinPhaseSelectLogic):
    def __init__(self, controller):
        super(TLSRoundRobinCycleAndPhaseSelectLogic, self).__init__(controller)

    @property
    def cycle_lengths(self):
        return self.controller.program.cycle_lengths

    def get_allowable_cycle_length_switches(self, time_on_cycle):
        if time_on_cycle < self.controller.current_cycle_length:
            mask = [0] * len(self.cycle_lengths)
            mask[self.controller.current_cycle_length_index] = 1
        else:  # allow switching to any cycle
            mask = [1] * len(self.cycle_lengths)
        return mask

    def get_allowable_phase_switches(self, time_on_phase):
        min_time_needed_on_cycle = (
            self.controller.n_phases - self.controller.next_phase_index
        ) * self.yellow_time
        if min_time_needed_on_cycle < self.controller.remaining_time_on_cycle:
            if (
                self.controller.remaining_time_on_cycle > self.yellow_time
                and self.controller.current_phase_index
                == self.controller.green_phase_indices[-1]
            ):
                mask = [0 for _ in range(self.n_phases)]
                mask[self.controller.green_phase_indices[-1]] = 1
            else:
                mask = super(
                    TLSRoundRobinCycleAndPhaseSelectLogic, self
                ).get_allowable_phase_switches(time_on_phase)
        else:
            mask = [0 for _ in range(self.n_phases)]
            mask[self.controller.next_phase_index] = 1
        # if self.id == "intersection_1_1":
        #     print(
        #         (
        #             min_time_needed_on_cycle,
        #             self.controller.remaining_time_on_cycle,
        #         )
        #     )
        return mask

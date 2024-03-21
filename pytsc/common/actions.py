from abc import ABC, abstractmethod

from pytsc.common.utils import pad_list


class BaseActionSpace(ABC):
    def __init__(self, config, traffic_signals):
        self.config = config
        self.traffic_signals = traffic_signals
        self.ts_ids = list(traffic_signals.keys())

    def _check_actions_type(self, actions):
        if not isinstance(actions, list):
            try:
                actions = actions.tolist()
            except Exception:
                print(
                    "Actions must be list or numpy array."
                    + f"Got {type(actions)}"
                )

    @abstractmethod
    def apply(self, actions):
        raise NotImplementedError

    @abstractmethod
    def get_mask(self):
        raise NotImplementedError

    @abstractmethod
    def get_size(self):
        raise NotImplementedError


class PhaseSelectionActionSpace(BaseActionSpace):
    """
    Actions are the phase index to switch to for each traffic signal.
    """

    def __init__(self, config, traffic_signals):
        super(PhaseSelectionActionSpace, self).__init__(
            config, traffic_signals
        )

    def apply(self, actions):
        # self._check_actions_type(actions)
        for ts_idx, ts in enumerate(self.traffic_signals.values()):
            ts.action_to_phase(actions[ts_idx])

    def get_size(self):
        return max(
            [ts.controller.n_phases for ts in self.traffic_signals.values()]
        )

    def get_mask(self):
        masks = []
        for ts in self.traffic_signals.values():
            mask = ts.controller.get_allowable_phase_switches()
            mask = pad_list(mask, self.get_size())
            masks.append(mask)
        return masks


class PhaseSwitchActionSpace(BaseActionSpace):
    """
    In this child class, actions are binary:
    0 for no switch (remain in the current phase)
    1 for switch (move to the next phase in a round-robin manner).
    """

    def __init__(self, config, traffic_signals):
        super(PhaseSwitchActionSpace, self).__init__(config, traffic_signals)

    def apply(self, actions):
        self._check_actions_type(actions)
        for ts_idx, ts in enumerate(self.traffic_signals.values()):
            current_phase = ts.controller.program.current_phase_index
            if actions[ts_idx] == 1:
                next_phase = (current_phase + 1) % ts.controller.n_phases
                ts.action_to_phase(next_phase)
            else:
                ts.action_to_phase(current_phase)

    def get_size(self):
        return 2

    def get_mask(self):
        masks = []
        for ts in self.traffic_signals.values():
            allowable_switches = ts.controller.get_allowable_phase_switches()
            n_phases = ts.controller.n_phases
            current_phase_index = ts.controller.program.current_phase_index
            next_phase_index = (current_phase_index + 1) % n_phases
            mask = [0, 0]
            if allowable_switches[current_phase_index]:
                mask[0] = 1
            if allowable_switches[next_phase_index]:
                mask[1] = 1
            masks.append(mask)
        return masks

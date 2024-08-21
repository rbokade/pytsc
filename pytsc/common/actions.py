from abc import ABC, abstractmethod
from itertools import product

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
                print("Actions must be list or numpy array." + f"Got {type(actions)}")

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
        super(PhaseSelectionActionSpace, self).__init__(config, traffic_signals)

    def apply(self, actions):
        # self._check_actions_type(actions)
        for ts_idx, ts in enumerate(self.traffic_signals.values()):
            ts.action_to_phase(actions[ts_idx])

    def get_size(self):
        return max([ts.controller.n_phases for ts in self.traffic_signals.values()])

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


class CentralizedActionSpace(BaseActionSpace):
    def __init__(self, individual_action_space):
        super(CentralizedActionSpace, self).__init__(
            individual_action_space.config,
            individual_action_space.traffic_signals,
        )
        self.individual_action_space = individual_action_space
        self.n_agents = len(self.traffic_signals)
        self.n_actions = self.individual_action_space.get_size()

    def apply(self, action):
        actions = self.decode_action(action.item())
        self.individual_action_space.apply(actions)

    def decode_action(self, action):
        actions = []
        current_action = action
        for _ in range(self.n_agents):
            actions.append(current_action % self.n_actions)
            current_action //= self.n_actions
        return actions

    def get_size(self):
        return self.individual_action_space.get_size() ** self.n_agents

    def get_mask(self):
        individual_masks = self.individual_action_space.get_mask()
        combinations = product(range(self.n_actions), repeat=self.n_agents)
        joint_mask = []
        for combination in combinations:
            is_valid = all(
                individual_masks[agent_idx][action] == 1
                for agent_idx, action in enumerate(combination)
            )
            joint_mask.append(is_valid)
        return joint_mask

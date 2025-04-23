from abc import ABC, abstractmethod
from itertools import product
import random

from pytsc.common.utils import pad_list


class BaseActionSpace(ABC):
    """
    Base class for action spaces in traffic signal control.
    This class defines the interface for different action spaces
    and provides common functionality for action space management.
    
    Args:
        config (Config): Configuration object containing simulation parameters.
        traffic_signals (dict): Dictionary of traffic signals in the network.
    """
    def __init__(self, config, traffic_signals):
        self.config = config
        self.traffic_signals = traffic_signals
        self.ts_ids = list(traffic_signals.keys())

    def _check_actions_type(self, actions):
        """
        Check if actions are in the correct format (list or numpy array).
        If not, try to convert them to a list.
        """
        if not isinstance(actions, list):
            try:
                actions = actions.tolist()
            except Exception:
                print("Actions must be list or numpy array." + f"Got {type(actions)}")

    @abstractmethod
    def apply(self, actions):
        """
        Apply the given actions to the traffic signals.
        Args:
            actions (list): List of actions to be applied to the traffic signals.
        """
        raise NotImplementedError

    @abstractmethod
    def get_mask(self):
        """
        Get the action mask for each traffic signal.
        Returns:
            list: List of action masks for each traffic signal.
        """
        raise NotImplementedError

    @abstractmethod
    def get_size(self):
        """
        Get the size of the action space.
        Returns:
            int: Size of the action space.
        """
        raise NotImplementedError

    def get_trad_controller_actions(self, controller):
        """
        Get the actions for traditional controllers.
        Args:
            controller (str): The type of controller to get actions for.
        Returns:
            list: List of actions for each traffic signal.
        """
        actions = []
        mask = self.get_mask()
        for i, ts in enumerate(self.traffic_signals.values()):
            if controller == "random":
                action = random.choices(range(ts.controller.n_phases), weights=mask[i])[
                    0
                ]
            else:
                action = ts.get_controller_action(controller)
            actions.append(action)
        return actions


class PhaseSelectionActionSpace(BaseActionSpace):
    """
    In this child class, actions are the phase index to switch to for each traffic signal.
    Args:
        config (Config): Configuration object containing simulation parameters.
        traffic_signals (dict): Dictionary of traffic signals in the network.        
    """

    def __init__(self, config, traffic_signals):
        super(PhaseSelectionActionSpace, self).__init__(config, traffic_signals)

    def apply(self, actions):
        """
        Apply the given actions to the traffic signals.
        Args:
            actions (list): List of actions to be applied to the traffic signals.
        """
        # self._check_actions_type(actions)
        for ts_idx, ts in enumerate(self.traffic_signals.values()):
            ts.action_to_phase(actions[ts_idx])

    def get_size(self):
        """
        Get the size of the action space.
        Returns:
            int: Size of the action space.
        """
        return max([ts.controller.n_phases for ts in self.traffic_signals.values()])

    def get_mask(self):
        """
        Get the action mask for each traffic signal.
        Returns:
            list: List of action masks for each traffic signal.
        """
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
        """
        Apply the given actions to the traffic signals.
        Args:
            actions (list): List of actions to be applied to the traffic signals.
        """
        self._check_actions_type(actions)
        for ts_idx, ts in enumerate(self.traffic_signals.values()):
            current_phase = ts.controller.program.current_phase_index
            if actions[ts_idx] == 1:
                next_phase = (current_phase + 1) % ts.controller.n_phases
                ts.action_to_phase(next_phase)
            else:
                ts.action_to_phase(current_phase)

    def get_size(self):
        """
        Get the size of the action space.
        Returns:
            int: Size of the action space.
        """
        return 2

    def get_mask(self):
        """
        Get the action mask for each traffic signal.
        Returns:
            list: List of action masks for each traffic signal.
        """
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

    def get_trad_controller_actions(self, controller):
        """
        Get the actions for traditional controllers.
        Args:
            controller (str): The type of controller to get actions for.
        Returns:
            list: List of actions for each traffic signal.
        """
        actions = []
        mask = self.get_mask()
        for i, ts in enumerate(self.traffic_signals.values()):
            if controller == "random":
                action = random.choices([0, 1], weights=mask[i])[0]
            else:
                action = ts.get_controller_action(controller)
            if ts.controller.program.current_phase_index != action:
                actions.append(1)
            else:
                actions.append(0)
        return actions


class CentralizedActionSpace(BaseActionSpace):
    """
    In this child class, actions are the phase index to switch to for each traffic signal.
    Args:
        individual_action_space (BaseActionSpace): The action space for each traffic signal.
    """
    def __init__(self, individual_action_space):
        super(CentralizedActionSpace, self).__init__(
            individual_action_space.config,
            individual_action_space.traffic_signals,
        )
        self.individual_action_space = individual_action_space
        self.n_agents = len(self.traffic_signals)
        self.n_actions = self.individual_action_space.get_size()

    def apply(self, action):
        """
        Apply the given action to the traffic signals.
        Args:
            action (int): The action to be applied to the traffic signals.
        """
        actions = self.decode_action(action.item())
        self.individual_action_space.apply(actions)

    def decode_action(self, action):
        """
        Decode the action into individual actions for each traffic signal.
        Args:
            action (int): The action to be decoded.
        Returns:
            list: List of individual actions for each traffic signal.
        """
        actions = []
        current_action = action
        for _ in range(self.n_agents):
            actions.append(current_action % self.n_actions)
            current_action //= self.n_actions
        return actions[::-1]

    def get_size(self):
        """
        Get the size of the action space.
        Returns:
            int: Size of the action space.
        """
        return self.individual_action_space.get_size() ** self.n_agents

    def get_mask(self):
        """
        Get the action mask for each traffic signal.
        Returns:
            list: List of action masks for each traffic signal.
        """
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

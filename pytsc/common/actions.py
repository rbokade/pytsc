from itertools import product

from pytsc.common.utils import pad_list


class BaseActionSpace:
    """
    Actions are the phase index to switch to for each traffic signal.
    Assumes homogenous action spaces for all traffic signals.
    """

    def __init__(self, traffic_signals):
        self.traffic_signals = traffic_signals
        self.ts_ids = list(traffic_signals.keys())

    def _check_actions_type(self, actions):
        if not isinstance(actions, list):
            try:
                actions = actions.tolist()
            except:
                print("Actions must be list or numpy array." + f"Got {type(actions)}")

    def apply(self, actions):
        self._check_actions_type(actions)
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


class BinaryActionSpace(BaseActionSpace):
    """
    In this child class, actions are binary:
    0 for no switch (remain in the current phase)
    1 for switch (move to the next phase in a round-robin manner).
    """

    def __init__(self, traffic_signals):
        super(BinaryActionSpace, self).__init__(traffic_signals)

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


class ActionSpaceWithOffset(BinaryActionSpace):
    offset_actions = [0, 1, 2, 3, 4, 5, 6]  # fixed for now

    def __init__(self, traffic_signals):
        super(ActionSpaceWithOffset, self).__init__(traffic_signals)

    def _split_actions(self, actions):
        phase_actions = [a // len(self.offset_actions) for a in actions]
        offset_actions = [a % len(self.offset_actions) for a in actions]
        return phase_actions, offset_actions

    def apply(self, actions):
        self._check_actions_type(actions)
        phase_actions, offset_actions = self._split_actions(actions)
        for ts_idx, ts in enumerate(self.traffic_signals.values()):
            current_phase = ts.controller.program.current_phase_index
            if phase_actions[ts_idx] == 1:
                next_phase = (current_phase + 1) % ts.controller.n_phases
                ts.action_to_phase(next_phase)
            else:
                ts.action_to_phase(current_phase)
            ts.store_offset(self.offset_actions[offset_actions[ts_idx]])

    def get_size(self):
        return 2 * len(self.offset_actions)

    def get_mask(self):
        extended_masks = []
        for ts in self.traffic_signals.values():
            n_phases = ts.controller.n_phases
            current_phase_index = ts.controller.program.current_phase_index
            allowable_switches = ts.controller.get_allowable_phase_switches()
            next_phase_index = (current_phase_index + 1) % n_phases
            mask = [0, 0]
            if allowable_switches[current_phase_index]:
                mask[0] = 1
            if allowable_switches[next_phase_index]:
                mask[1] = 1
            extended_mask = []
            for phase_allowed in mask:
                extended_mask.extend([phase_allowed] * len(self.offset_actions))
            extended_masks.append(extended_mask)
        return extended_masks


# class ActionSpaceWithOffset(BaseActionSpace):
#     offset_actions = [0, 1, 2, 3, 4, 5, 6]  # fixed

#     def __init__(self, traffic_signals):
#         super(ActionSpaceWithOffset, self).__init__(traffic_signals)

#     def _split_actions(self, actions):
#         phase_actions = [a // len(self.offset_actions) for a in actions]
#         offset_actions = [a % len(self.offset_actions) for a in actions]
#         return phase_actions, offset_actions

#     def apply(self, actions):
#         if not isinstance(actions, list):
#             try:
#                 actions = actions.tolist()
#             except:
#                 print(
#                     f"Actions must be list or numpy array. Got {type(actions)}"
#                 )
#         phase_actions, offset_actions = self._split_actions(actions)
#         for ts_idx, ts in enumerate(self.traffic_signals.values()):
#             ts.action_to_phase(phase_actions[ts_idx])
#             ts.store_offset(self.offset_actions[offset_actions[ts_idx]])

#     def get_size(self):
#         """
#         The total action space size is (# phases x # offsets).
#         """
#         return super(ActionSpaceWithOffset, self).get_size() * len(
#             self.offset_actions
#         )

#     def get_mask(self):
#         """
#         Create a mask that includes both phase switches and offset possibilities.
#         """
#         extended_masks = []
#         for ts in self.traffic_signals.values():
#             mask = ts.controller.get_allowable_phase_switches()
#             # mask = pad_list(
#             #     mask, super(ActionSpaceWithOffset, self).get_size()
#             # )
#             extended_mask = []
#             for phase_allowed in mask:
#                 extended_mask.extend(
#                     [phase_allowed] * len(self.offset_actions)
#                 )
#             extended_masks.append(extended_mask)
#         return extended_masks

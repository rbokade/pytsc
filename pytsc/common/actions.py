from itertools import product

import numpy as np

from pytsc.common.utils import pad_list


class BaseActionSpace:
    """
    Actions are the phase index to switch to for each traffic signal.
    Assumes homogenous action spaces for all traffic signals.
    """

    def __init__(self, config, traffic_signals):
        self.config = config
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

    def __init__(self, config, traffic_signals):
        super(BinaryActionSpace, self).__init__(config, traffic_signals)

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


class PhaseAndCycleLengthActionSpace(BaseActionSpace):
    """
    Actions are (for each traffic signal)
    (1) the phase index to switch to
    (2) the cycle length to switch to
    """

    def __init__(self, config, traffic_signals):
        super().__init__(config, traffic_signals)

    @property
    def network_max_cycle_length(self):
        return max(
            [len(ts.controller.cycle_lengths) for ts in self.traffic_signals.values()]
        )

    def apply(self, actions):
        for ts_idx, (ts_id, ts) in enumerate(self.traffic_signals.items()):
            action = actions[ts_idx]
            current_phase = ts.controller.program.current_phase_index
            if action % 2:  # phase switch == 1
                next_phase = (current_phase + 1) % ts.controller.n_phases
                phase_index = next_phase
            else:
                phase_index = current_phase
            cycle_length_index = action // 2
            ts.action_to_phase(phase_index, cycle_length_index=cycle_length_index)

    def get_size(self):
        return 2 * self.network_max_cycle_length

    def _get_phase_switch_mask(self, ts):
        allowable_switches = ts.controller.get_allowable_phase_switches()
        n_phases = ts.controller.n_phases
        current_phase_index = ts.controller.program.current_phase_index
        next_phase_index = (current_phase_index + 1) % n_phases
        phase_switch_mask = [0, 0]
        if allowable_switches[current_phase_index]:
            phase_switch_mask[0] = 1
        if allowable_switches[next_phase_index]:
            phase_switch_mask[1] = 1
        return phase_switch_mask

    def get_mask(self):
        mask = []
        for ts_id, ts in self.traffic_signals.items():
            phase_switch_mask = self._get_phase_switch_mask(ts)
            cycle_length_mask = ts.controller.get_allowable_cycle_length_switches()
            mask.append(
                [
                    int(cycle and phase)
                    for cycle, phase in product(cycle_length_mask, phase_switch_mask)
                ]
            )
        return mask


class ActionSpaceWithOffset(BinaryActionSpace):
    offset_actions = [0, 1, 2, 3, 4, 5, 6]  # fixed for now

    def __init__(self, config, traffic_signals):
        super(ActionSpaceWithOffset, self).__init__(config, traffic_signals)

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


# class KuramotoActionSpace(BaseActionSpace):
#     """
#     Action for each agent: coupling strength (0, 1) for each neighbor
#     Size of the action space: 2**max(node_degrees)
#     Action masking:
#         - Convert the actions into NxN matrix and multiply by the adjacency matrix
#     """

#     def __init__(self, config, traffic_signals):
#         super(KuramotoActionSpace, self).__init__(config, traffic_signals)
#         self.n_agents = len(self.traffic_signals)
#         self.threshold = self.config.misc_config.get("kuramoto_action_threshold", 0.0)

#     @property
#     def max_degree(self):
#         """
#         NOTE: ts.neighborhood_matrix: adjacency_matrix[ts_idx]
#         """
#         return max(
#             [
#                 int(ts.neighborhood_matrix.sum().item())
#                 for ts in self.traffic_signals.values()
#             ]
#         )

#     def get_mask(self):
#         return [[1 for _ in range(2**self.max_degree)] for _ in range(self.n_agents)]

#     def _index_to_binary_vector(self, index, length):
#         binary_str = format(index, f"0{length}b")
#         return [int(bit) for bit in binary_str]

#     def _actions_to_coupling_strength(self, actions):
#         """
#         Returns a coupling strength matrix based on actions taken by the agents.
#         args:
#             actions: [n_agents]
#         returns:
#             coupling_strength_matrix: [n_agents, n_agents]
#         """
#         coupling_strength_matrix = np.zeros((self.n_agents, self.n_agents))
#         for idx, (ts_id, action) in enumerate(
#             zip(self.traffic_signals.keys(), actions)
#         ):
#             binary_vector = self._index_to_binary_vector(action, self.max_degree)
#             neighbors = self.traffic_signals[ts_id].neighborhood_matrix
#             # Extracting indices of neighbors
#             neighbors_idx = [
#                 i for i, is_neighbor in enumerate(neighbors) if is_neighbor
#             ]
#             # Assigning coupling strengths based on the binary vector
#             for i, neigh_idx in enumerate(neighbors_idx):
#                 coupling_strength_matrix[idx][neigh_idx] = binary_vector[i]
#         return coupling_strength_matrix

#     def _get_phase_difference(self, ts, neigh_ts):
#         phase_angle = ts.controller.phase_and_cycle_history[-1][0]
#         phase_angle /= ts.controller.n_phases
#         phase_angle *= 2 * np.pi
#         # Compute neighbor's phase angle
#         offset = ts.neighbors_offsets[neigh_ts.id]
#         t = len(neigh_ts.controller.phase_and_cycle_history) - 1
#         offset_t = max(t - offset, 0)
#         neigh_phase_angle = neigh_ts.controller.phase_and_cycle_history[offset_t][0]
#         neigh_phase_angle /= neigh_ts.controller.n_phases
#         neigh_phase_angle *= 2 * np.pi
#         # Return phase difference
#         return neigh_phase_angle - phase_angle

#     def _compute_kuramotos_for_each_agent(self, coupling_strengths):
#         """
#         Returns a mean field kuramoto for each agent.
#         args:
#             coupling_strength: [n_agents, n_agents]
#         returns:
#             kuramotos: [n_agents]
#         """
#         kuramotos = np.zeros(self.n_agents)
#         for i, ts in enumerate(self.traffic_signals.values()):
#             total, count = 0, 0
#             for j, neigh_ts in enumerate(self.traffic_signals.values()):
#                 if coupling_strengths[i][j] > 0:
#                     phase_diff = self._get_phase_difference(ts, neigh_ts)
#                     total += coupling_strengths[i][j] * np.sin(phase_diff)
#                     count += 1
#             kuramotos[i] = total / count if count > 0 else 0
#         return kuramotos

#     def _kuramoto_to_phase_switch(self, kuramotos):
#         """
#         If the agent is lagging behind the mean phase then it should
#         switch to the next phase, otherwise it should stay at the
#         current phase.
#         """
#         for ts_idx, ts in enumerate(self.traffic_signals.values()):
#             allowable_switches = ts.controller.get_allowable_phase_switches()
#             n_phases = ts.controller.n_phases
#             current_phase_index = ts.controller.program.current_phase_index
#             next_phase_index = (current_phase_index + 1) % n_phases
#             # if allowable_switches[current_phase_index]:
#             # if allowable_switches[next_phase_index]:
#             if (
#                 kuramotos[ts_idx] >= self.threshold
#                 and allowable_switches[next_phase_index]
#             ) or not allowable_switches[
#                 current_phase_index
#             ]:  # switch to the next phase
#                 next_phase = (current_phase_index + 1) % ts.controller.n_phases
#                 ts.action_to_phase(next_phase)
#             else:  # stay on the current phase
#                 ts.action_to_phase(current_phase_index)

#     def apply(self, actions):
#         coupling_strengths = self._actions_to_coupling_strength(actions)
#         kuramotos = self._compute_kuramotos_for_each_agent(coupling_strengths)
#         self._kuramoto_to_phase_switch(kuramotos)

#     def get_size(self):
#         return 2**self.max_degree


class KuramotoActionSpace(BaseActionSpace):
    """
    Action for each agent: coupling strength (0, 1) for each neighbor
    Size of the action space: 2**max(node_degrees)
    Action masking:
        - Convert the actions into NxN matrix and multiply by the adjacency matrix
    """

    def __init__(self, config, traffic_signals):
        super(KuramotoActionSpace, self).__init__(config, traffic_signals)
        self.n_agents = len(self.traffic_signals)
        self.threshold = self.config.misc_config.get("kuramoto_action_threshold", 0.0)

    @property
    def max_degree(self):
        """
        NOTE: ts.neighborhood_matrix: adjacency_matrix[ts_idx]
        """
        return max(
            [
                int(ts.neighborhood_matrix.sum().item())
                for ts in self.traffic_signals.values()
            ]
        )

    def get_mask(self):
        masks = []
        for ts in self.traffic_signals.values():
            n_neighbors = int(ts.neighborhood_matrix.sum().item())
            ts_mask = [0] * self.get_size()
            for i in range(n_neighbors):
                ts_mask[i] = 1
            masks.append(ts_mask)
        return masks

    def _actions_to_coupling_strength(self, actions):
        coupling_strength_matrix = np.zeros((self.n_agents, self.n_agents))
        for idx, (ts_id, action) in enumerate(
            zip(self.traffic_signals.keys(), actions)
        ):
            neighbors = self.traffic_signals[ts_id].neighborhood_matrix
            neighbors_idx = [
                i for i, is_neighbor in enumerate(neighbors) if is_neighbor
            ]

            # Check if action corresponds to a valid neighbor
            if action < len(neighbors_idx):
                selected_neighbor_idx = neighbors_idx[action]
                coupling_strength_matrix[idx][selected_neighbor_idx] = 1
        return coupling_strength_matrix

    def _get_phase_difference(self, ts, neigh_ts):
        phase_angle = ts.controller.phase_and_cycle_history[-1][0]
        phase_angle /= ts.controller.n_phases
        phase_angle *= 2 * np.pi
        # Compute neighbor's phase angle
        offset = ts.neighbors_offsets[neigh_ts.id]
        t = len(neigh_ts.controller.phase_and_cycle_history) - 1
        offset_t = max(t - offset, 0)
        neigh_phase_angle = neigh_ts.controller.phase_and_cycle_history[offset_t][0]
        neigh_phase_angle /= neigh_ts.controller.n_phases
        neigh_phase_angle *= 2 * np.pi
        # Return phase difference
        return neigh_phase_angle - phase_angle

    def _compute_kuramotos_for_each_agent(self, coupling_strengths):
        """
        Returns a mean field kuramoto for each agent.
        args:
            coupling_strength: [n_agents, n_agents]
        returns:
            kuramotos: [n_agents]
        """
        kuramotos = np.zeros(self.n_agents)
        for i, ts in enumerate(self.traffic_signals.values()):
            total, count = 0, 0
            for j, neigh_ts in enumerate(self.traffic_signals.values()):
                if coupling_strengths[i][j] > 0:
                    phase_diff = self._get_phase_difference(ts, neigh_ts)
                    total += coupling_strengths[i][j] * np.sin(phase_diff)
                    count += 1
            kuramotos[i] = total / count if count > 0 else 0
        return kuramotos

    def _kuramoto_to_phase_switch(self, kuramotos):
        """
        If the agent is lagging behind the mean phase then it should
        switch to the next phase, otherwise it should stay at the
        current phase.
        """
        for ts_idx, ts in enumerate(self.traffic_signals.values()):
            allowable_switches = ts.controller.get_allowable_phase_switches()
            n_phases = ts.controller.n_phases
            current_phase_index = ts.controller.program.current_phase_index
            next_phase_index = (current_phase_index + 1) % n_phases
            # if allowable_switches[current_phase_index]:
            # if allowable_switches[next_phase_index]:
            if (
                kuramotos[ts_idx] >= self.threshold
                and allowable_switches[next_phase_index]
            ) or not allowable_switches[
                current_phase_index
            ]:  # switch to the next phase
                next_phase = (current_phase_index + 1) % ts.controller.n_phases
                ts.action_to_phase(next_phase)
            else:  # stay on the current phase
                ts.action_to_phase(current_phase_index)

    def apply(self, actions):
        coupling_strengths = self._actions_to_coupling_strength(actions)
        kuramotos = self._compute_kuramotos_for_each_agent(coupling_strengths)
        self._kuramoto_to_phase_switch(kuramotos)

    def get_size(self):
        return self.max_degree

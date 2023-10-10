class BaseActionSpace:
    """
    Actions are the phase index to switch to for each traffic signal.
    Assumes homogenous action spaces for all traffic signals.
    """

    def __init__(self, traffic_signals):
        self.traffic_signals = traffic_signals
        self.ts_ids = list(traffic_signals.keys())

    def apply(self, actions):
        for ts_idx, ts in enumerate(self.traffic_signals.values()):
            ts.action_to_phase(actions[ts_idx])

    def get_size(self):
        return max(
            [ts.program.n_phases for ts in self.traffic_signals.values()]
        )

    def get_mask(self):
        masks = []
        for ts in self.traffic_signals.values():
            mask = ts.controller.get_allowable_phase_switches()
            masks.append(mask)
        return masks


class PhaseAndCycleLengthActionSpace(BaseActionSpace):
    """
    Actions are (for each traffic signal)
    (1) the phase index to switch to
    (2) the cycle length to switch to
    """

    def __init__(self, traffic_signals):
        super().__init__(traffic_signals)

    def apply(self, actions):
        phases, cycle_lengths = actions
        for ts_idx, ts in enumerate(self.traffic_signals.values()):
            ts.action_to_cycle_length(cycle_lengths[ts_idx])
            ts.action_to_phase(phases[ts_idx])

    def get_size(self):
        return (
            max([ts.program.n_phases for ts in self.traffic_signals.values()]),
            max(
                [
                    ts.controller.n_cycle_lengths
                    for ts in self.traffic_signals.values()
                ]
            ),
        )

    def get_mask(self):
        phase_masks, cycle_length_masks = [], []
        for ts in self.traffic_signals.values():
            phase_masks.append(ts.controller.get_allowable_phase_switches())
            cycle_length_masks.append(
                ts.controller.get_allowable_cycle_length_switches()
            )
        return phase_masks, cycle_length_masks

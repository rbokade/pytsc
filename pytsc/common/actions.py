from pytsc.common.utils import pad_list


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
            [ts.controller.n_phases for ts in self.traffic_signals.values()]
        )

    def get_mask(self):
        masks = []
        for ts in self.traffic_signals.values():
            mask = ts.controller.get_allowable_phase_switches()
            mask = pad_list(mask, self.get_size())
            masks.append(mask)
        return masks

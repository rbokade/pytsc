from abc import ABC

from more_itertools import padded
import numpy as np

from pytsc.common.utils import pad_list


class BaseObservationSpace(ABC):
    """
    Standard observation space for traffic signal control.
    """

    def __init__(self, config, parsed_network, traffic_signals, simulator_backend):
        self.config = config
        self.parsed_network = parsed_network
        self.n_agents = len(traffic_signals)
        self.traffic_signals = traffic_signals
        self.simulator_backend = simulator_backend
        # Needed to get observation size and padding
        self.max_n_controlled_lanes = np.max(
            [len(ts.incoming_lanes) for ts in traffic_signals.values()]
        ).item()
        self.max_n_controlled_phases = np.max(
            [ts.n_phases for ts in traffic_signals.values()]
        ).item()
        self.pad_value = config.misc["pad_value"]

    @staticmethod
    def high():
        return 1

    @staticmethod
    def low():
        return -1


class PositionMatrix(BaseObservationSpace):
    """
    Each incoming lane is divided into bins of v_size + min_gap (7.5) meters.
    Visibility of each traffic signal is in terms of # of bins, not meters.
    -1: no vehicle in bin
    0: vehicle in bin, not moving
    >0: vehicle in bin, moving
    Observation:
        - position matrix of each incoming lane
        - one hot encoding of current phase
    """

    def __init__(self, config, parsed_network, traffic_signals, simulator_backend):
        super(PositionMatrix, self).__init__(
            config, parsed_network, traffic_signals, simulator_backend
        )
        self.visibility = config.signal["visibility"]
        self.max_mats_size = self.visibility * self.max_n_controlled_lanes
        # self.max_mats_size = 80

    def get_observations(self):
        observations = []
        for ts in self.traffic_signals.values():
            obs = []
            pos_mats = ts.inc_position_matrices
            for pos_mat in pos_mats.values():
                obs.extend(pos_mat)
            obs = pad_list(obs, self.max_mats_size, self.pad_value)
            obs.extend(ts.phase_id)
            observations.append(obs)
        return observations

    def get_observation_info(self):
        return None

    def get_size(self):
        return self.max_mats_size + self.max_n_controlled_phases

    def get_state(self):
        observations = self.get_observations()
        state = []
        for obs in observations:
            state.extend(obs)
        return state

    def get_state_size(self):
        return len(self.traffic_signals.keys()) * self.get_size()

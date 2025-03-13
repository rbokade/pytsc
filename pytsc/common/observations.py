from abc import ABC
from copy import deepcopy

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

    max_lanes_per_direction = 4
    max_lane_speed = 15.0  # m/s
    max_lane_length = 500  # m
    max_phases = 10

    def __init__(self, config, parsed_network, traffic_signals, simulator_backend):
        super(PositionMatrix, self).__init__(
            config, parsed_network, traffic_signals, simulator_backend
        )
        self.visibility = config.signal["visibility"]
        self.dropout_prob = config.signal.get("obs_dropout_prob", 0.0)
        self.max_mats_size = self.visibility * self.max_n_controlled_lanes
        self.lane_features = self._get_lane_features()

    def _get_lane_features(self):
        self.lane_lengths = self.parsed_network.lane_lengths
        self.lane_indices = self.parsed_network.lane_indices
        self.lane_angles = self.parsed_network.lane_angles
        self.lane_max_speeds = self.parsed_network.lane_max_speeds
        lane_features = {}
        for lane in self.parsed_network.lanes:
            one_hot_idx = [0.0 for _ in range(self.max_lanes_per_direction)]
            one_hot_idx[self.lane_indices[lane]] = 1.0
            lane_length = self.lane_lengths[lane] / self.max_lane_length
            lane_angle = self.lane_angles[lane] / np.pi
            lane_max_speed = self.lane_max_speeds[lane] / self.max_lane_speed
            lane_length = np.clip(lane_length, 0, 1)
            lane_angle = np.clip(lane_angle, -1, 1)
            lane_max_speed = np.clip(lane_max_speed, 0, 1)
            lane_features[lane] = [lane_length, lane_angle, lane_max_speed]
            lane_features[lane].extend(one_hot_idx)
        return lane_features

    def get_observations(self):
        observations = []
        for ts in self.traffic_signals.values():
            obs = []
            pos_mats = deepcopy(ts.inc_position_matrices)
            for lane, pos_mat in pos_mats.items():
                obs.extend(self.lane_features[lane])
                if self.dropout_prob > 0:
                    drop_idx = np.random.choice(
                        self.visibility,
                        int(self.visibility * self.dropout_prob),
                        replace=False,
                    )
                    for idx in drop_idx:
                        pos_mat[idx] = -1.0
                obs.extend(pos_mat)
            obs = pad_list(obs, self.get_size() - self.max_phases, self.pad_value)
            phase_id = pad_list(ts.phase_id, self.max_phases)
            obs.extend(phase_id)
            observations.append(obs)
        return observations

    def get_observation_info(self):
        info = {
            "lane_obs_dim": 7 + self.visibility,
            "max_n_controlled_lanes": self.max_n_controlled_lanes,
            "max_phases": self.max_phases,
        }
        return info

    def get_size(self):
        return self.max_n_controlled_lanes * (7 + self.visibility) + self.max_phases

    def get_state(self):
        observations = self.get_observations()
        state = []
        for obs in observations:
            state.extend(obs)
        return state

    def get_state_size(self):
        return len(self.traffic_signals.keys()) * self.get_size()

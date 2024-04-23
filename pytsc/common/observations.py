from abc import ABC
from turtle import position

import numpy as np

from pytsc.common.utils import compute_linearly_weighted_average, pad_array


class BaseObservationSpace(ABC):
    """
    Standard observation space for traffic signal control.
    """

    def __init__(
        self, config, parsed_network, traffic_signals, simulator_backend
    ):
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
        return 0


class LaneFeatures(BaseObservationSpace):
    def __init__(
        self, config, parsed_network, traffic_signals, simulator_backend
    ):
        super(LaneFeatures, self).__init__(
            config, parsed_network, traffic_signals, simulator_backend
        )

    def get_observations(self):
        """
        Returns a list of observations for each traffic signal in
        traffic_signals. The observations are obtained by padding
        and concatenating arrays of normalized queue lengths, densities,
        mean speeds, and mean wait times for each traffic signal.

        Returns:
        - observations: A list of observations for each traffic signal in
          traffic_signals.
        """
        observations = []
        for ts in self.traffic_signals.values():
            obs = pad_array(
                ts.norm_queue_lengths, self.max_n_controlled_lanes
            ).tolist()
            obs += pad_array(
                ts.norm_densities, self.max_n_controlled_lanes
            ).tolist()
            obs += pad_array(
                ts.norm_mean_speeds, self.max_n_controlled_lanes
            ).tolist()
            obs += pad_array(
                ts.norm_mean_wait_times, self.max_n_controlled_lanes
            ).tolist()
            obs += pad_array(
                ts.phase_id, self.max_n_controlled_phases
            ).tolist()
            observations.append(obs)
        return observations

    def get_size(self):
        return int(
            (4 * self.max_n_controlled_lanes) + self.max_n_controlled_phases
        )

    def get_state(self):
        observations = self.get_observations()
        state = []
        for obs in observations:
            state.extend(obs)
        return state

    def get_state_size(self):
        return self.get_size() * len(self.traffic_signals)


class PositionMatrix(LaneFeatures):
    def __init__(
        self, config, parsed_network, traffic_signals, simulator_backend
    ):
        super(PositionMatrix, self).__init__(
            config, parsed_network, traffic_signals, simulator_backend
        )

    def get_observations(self):
        observations = []
        for ts in self.traffic_signals.values():
            pos_mat = compute_linearly_weighted_average(ts.position_matrices)
            pos_mat = pad_array(
                pos_mat,
                self.max_n_controlled_lanes * self.config.signal["visibility"],
            )
            obs = np.concatenate((pos_mat, ts.phase_id), axis=0)
            observations.append(obs.tolist())
        return observations

    def get_size(self):
        return int(
            (self.max_n_controlled_lanes * self.config.signal["visibility"])
            + self.max_n_controlled_phases
        )

    def get_state(self):
        lane_features = super(PositionMatrix, self).get_observations()
        position_matrix = self.get_observations()
        return np.concatenate(lane_features + position_matrix).tolist()

    def get_state_size(self):
        lane_features_size = int(
            (4 * self.max_n_controlled_lanes) + self.max_n_controlled_phases
        ) * len(self.traffic_signals)
        pos_mat_size = self.get_size() * len(self.traffic_signals)
        return lane_features_size + pos_mat_size

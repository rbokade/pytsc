from abc import ABC
from turtle import position

import numpy as np

from pytsc.common.utils import compute_linearly_weighted_average, pad_array


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
        return 0


class LaneFeatures(BaseObservationSpace):
    def __init__(self, config, parsed_network, traffic_signals, simulator_backend):
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
            obs = pad_array(ts.norm_queue_lengths, self.max_n_controlled_lanes).tolist()
            obs += pad_array(ts.densities, self.max_n_controlled_lanes).tolist()
            obs += pad_array(ts.norm_mean_speeds, self.max_n_controlled_lanes).tolist()
            obs += pad_array(
                ts.norm_mean_wait_times, self.max_n_controlled_lanes
            ).tolist()
            obs += pad_array(ts.phase_id, self.max_n_controlled_phases).tolist()
            obs += [np.log(1 + ts.sim_step)]
            observations.append(obs)
        return observations

    def get_observation_info(self):
        feat_length = self.max_n_controlled_lanes
        queue_start, queue_end = 0, feat_length
        density_start, density_end = queue_end, queue_end + feat_length
        speed_start, speed_end = density_end, density_end + feat_length
        wait_start, wait_end = speed_end, speed_end + feat_length
        phase_start, phase_end = (
            wait_end,
            wait_end + self.max_n_controlled_phases,
        )
        return {
            "n_lanes": self.max_n_controlled_lanes,
            "n_phases": self.max_n_controlled_phases,
            "queue_idxs": range(queue_start, queue_end),
            "density_idxs": range(density_start, density_end),
            "speed_idxs": range(speed_start, speed_end),
            "wait_idxs": range(wait_start, wait_end),
            "phase_idxs": range(phase_start, phase_end),
        }

    def get_size(self):
        return int((4 * self.max_n_controlled_lanes) + self.max_n_controlled_phases + 1)

    def get_state(self):
        observations = self.get_observations()
        return np.concatenate(observations, axis=0).tolist()

    def get_state_size(self):
        return len(self.traffic_signals.keys()) * self.get_size()


class PositionMatrix(LaneFeatures):
    def __init__(self, config, parsed_network, traffic_signals, simulator_backend):
        super(PositionMatrix, self).__init__(
            config, parsed_network, traffic_signals, simulator_backend
        )

    def get_observations(self):
        observations = []
        for ts in self.traffic_signals.values():
            pos_mat = compute_linearly_weighted_average(ts.position_matrices)
            pos_mat = pad_array(
                pos_mat,
                2 * self.max_n_controlled_lanes * self.config.signal["visibility"],
            )
            speed_mat = compute_linearly_weighted_average(ts.speed_matrices)
            speed_mat = pad_array(
                speed_mat,
                2 * self.max_n_controlled_lanes * self.config.signal["visibility"],
            )
            obs = np.concatenate((pos_mat, speed_mat, ts.phase_id), axis=0)
            observations.append(obs.tolist() + [ts.sim_step])
        return observations

    def get_observation_info(self):
        mat_length = 2 * self.max_n_controlled_lanes * self.config.signal["visibility"]
        pos_start = 0
        pos_end = mat_length
        speed_start = pos_end
        speed_end = speed_start + mat_length
        phase_start = speed_end
        phase_end = phase_start + self.max_n_controlled_phases
        return {
            "visibility": self.config.signal["visibility"],
            "n_lanes": 2 * self.max_n_controlled_lanes,
            "n_phases": self.max_n_controlled_phases,
            "pos_mat_idxs": range(pos_start, pos_end),
            "speed_mat_idxs": range(speed_start, speed_end),
            "phase_idxs": range(phase_start, phase_end),
            "sim_step_idx": phase_end,
        }

    def get_size(self):
        size = int(
            4  # (pos, speed, inc and out lanes)
            * (self.max_n_controlled_lanes * self.config.signal["visibility"])
            + self.max_n_controlled_phases
            + 1
        )
        return size

    def get_state(self):
        return np.concatenate(
            (
                np.stack(self.get_observations()).flatten(),
                super(PositionMatrix, self).get_state(),
            ),
            axis=0,
        ).tolist()

    def get_state_size(self):
        size = super(PositionMatrix, self).get_state_size()
        size += len(self.traffic_signals.keys()) * self.get_size()
        return size

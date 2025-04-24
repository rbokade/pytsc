from abc import ABC
from copy import deepcopy

import numpy as np

from pytsc.common.utils import pad_list


class BaseObservationSpace(ABC):
    """
    Base class for observation spaces in traffic signal control.
    This class defines the interface for different observation spaces
    and provides common functionality for observation space management.
    
    Args:
        config (Config): Configuration object containing simulation parameters.
        parsed_network (ParsedNetwork): Parsed network object containing network information.
        traffic_signals (dict): Dictionary of traffic signals in the network.
        simulator_backend (str): The simulator backend to be used (e.g., "cityflow", "sumo").    
    """

    def __init__(self, config, parsed_network, traffic_signals, simulator_backend):
        self.config = config
        self.parsed_network = parsed_network
        self.n_agents = len(traffic_signals)
        self.traffic_signals = traffic_signals
        self.simulator_backend = simulator_backend
        # Needed to get observation size and padding
        # self.max_n_controlled_lanes = np.max(
        #     [len(ts.incoming_lanes) for ts in traffic_signals.values()]
        # ).item()
        # self.max_n_controlled_phases = np.max(
        #     [ts.n_phases for ts in traffic_signals.values()]
        # ).item()
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

    max_lanes_per_direction = 6
    max_lane_speed = 15.0  # m/s
    max_lane_length = 500  # m
    max_phases = 20
    max_n_controlled_lanes = 16

    def __init__(self, config, parsed_network, traffic_signals, simulator_backend):
        super(PositionMatrix, self).__init__(
            config, parsed_network, traffic_signals, simulator_backend
        )
        self.visibility = config.signal["visibility"]
        self.lane_features = self._get_lane_features()
        self.obs_noise_std = config.signal.get("obs_noise_std", 0.00)

    def _add_gaussian_noise(self, pos_mat, std=0.1):
        """
        Add Gaussian noise to the position matrix.
        Args:
            pos_mat (list): Position matrix of the lane.
            std (float): Standard deviation of the Gaussian noise.
        Returns:
            list: Noisy position matrix.
        """
        noisy = []
        for val in pos_mat:
            if val > 0:
                noise = np.random.normal(loc=0.0, scale=std)
                noisy_val = val + noise
                noisy_val = np.clip(noisy_val, 0.0, 1.0) if val >= 0 else -1
                noisy.append(noisy_val)
        return noisy

    def _get_lane_features(self):
        """
        Get lane features for each lane in the network.
        Lane features include:
            - lane length (normalized)
            - lane angle (normalized)
            - lane max speed (normalized)
            - one-hot encoding of lane index
        Returns:
            dict: Dictionary of lane features for each lane.
        """
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

    def get_per_agent_lane_features(self):
        """
        Get lane features for each agent (traffic signal).
        Each agent's lane features are padded to the maximum number of controlled lanes.
        Returns:
            list: List of lane features for each agent.
        """
        per_agent_lane_features = []
        for ts in self.traffic_signals.values():
            lane_features = []
            for lane in ts.incoming_lanes:
                lane_features.append(self.lane_features[lane])
            if len(ts.incoming_lanes) < self.max_n_controlled_lanes:
                n_pad_lanes = self.max_n_controlled_lanes - len(ts.incoming_lanes)
                for _ in range(n_pad_lanes):
                    lane_features.append(
                        [self.pad_value] * (self.max_lanes_per_direction + 3)
                    )
            per_agent_lane_features.append(lane_features)
        return per_agent_lane_features

    def get_observations(self):
        """
        Get observations for each traffic signal.
        Each observation includes the position matrix of each incoming lane
        and the one-hot encoding of the current phase.
        Returns:
            list: List of observations for each traffic signal.
        """
        observations = []
        for ts in self.traffic_signals.values():
            obs = []
            pos_mats = deepcopy(ts.inc_position_matrices)
            for lane, pos_mat in pos_mats.items():
                obs.extend(self.lane_features[lane])
                pos_mat = self._add_gaussian_noise(pos_mat, std=self.obs_noise_std)
                obs.extend(pos_mat)
            obs = pad_list(obs, self.get_size() - self.max_phases, -1)
            phase_id = pad_list(ts.phase_id, self.max_phases, -1)
            obs.extend(phase_id)
            observations.append(obs)
        return observations

    def get_observation_info(self):
        """
        Get information about the observation space.
        Returns:
            dict: Dictionary containing information about the observation space.
        """
        info = {
            "lane_obs_dim": 9 + self.visibility,
            "max_n_controlled_lanes": self.max_n_controlled_lanes,
            "max_phases": self.max_phases,
        }
        return info

    def get_size(self):
        """
        Get the size of the observation space.
        Returns:    
            int: Size of the observation space.
        """
        return self.max_n_controlled_lanes * (self.visibility + 9) + self.max_phases

    def get_state(self):
        """
        Get the state for each traffic signal.
        Each state includes the position matrix of each incoming lane
        and the one-hot encoding of the current phase.
        Returns:
            list: List of states for each traffic signal.
        """
        states = []
        for ts in self.traffic_signals.values():
            state = []
            for lane in ts.incoming_lanes:
                state.extend(self.lane_features[lane])
                lane_results = ts.sub_results["lane"][lane]
                n_queued = lane_results["n_queued"]
                occupancy = lane_results["occupancy"]
                mean_speed = lane_results["mean_speed"]
                state.extend([n_queued, occupancy, mean_speed])
            state = pad_list(state, self.get_state_size() - self.max_phases, -1)
            phase_id = pad_list(ts.phase_id, self.max_phases)
            state.extend(phase_id)
            states.append(state)
        return states

    def get_state_size(self):
        """
        Get the size of the state space.
        Returns:
            int: Size of the state space.
        """
        return self.max_n_controlled_lanes * (3 + 9) + self.max_phases


class LaneFeatures(BaseObservationSpace):
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

    max_lanes_per_direction = 6
    max_lane_speed = 15.0  # m/s
    max_lane_length = 500  # m
    max_phases = 20
    max_n_controlled_lanes = 16

    def __init__(self, config, parsed_network, traffic_signals, simulator_backend):
        super(LaneFeatures, self).__init__(
            config, parsed_network, traffic_signals, simulator_backend
        )
        self.visibility = config.signal["visibility"]
        self.dropout_prob = config.signal.get("obs_dropout_prob", 0.0)
        # self.max_mats_size = self.visibility * self.max_n_controlled_lanes
        self.lane_features = self._get_lane_features()
        self.reset_dropped_lanes()
        # self.dropout_prob lanes are dropped

    def reset_dropped_lanes(self):
        """
        Randomly drop lanes based on the dropout probability.
        This method is called at the beginning of each episode.
        """
        lanes = self.parsed_network.lanes
        self.dropped_lanes = np.random.choice(
            lanes, int(self.dropout_prob * len(lanes)), replace=False
        )

    def _get_lane_features(self):
        """
        Get lane features for each lane in the network.
        Lane features include:
            - lane length (normalized)
            - lane angle (normalized)
            - lane max speed (normalized)
            - one-hot encoding of lane index
        Returns:
            dict: Dictionary of lane features for each lane.
        """
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

    def get_per_agent_lane_features(self):
        """
        Get lane features for each agent (traffic signal).
        Each agent's lane features are padded to the maximum number of controlled lanes.
        Returns:
            list: List of lane features for each agent.
        """
        per_agent_lane_features = []
        for ts in self.traffic_signals.values():
            lane_features = []
            for lane in ts.incoming_lanes:
                lane_features.append(self.lane_features[lane])
            if len(ts.incoming_lanes) < self.max_n_controlled_lanes:
                n_pad_lanes = self.max_n_controlled_lanes - len(ts.incoming_lanes)
                for _ in range(n_pad_lanes):
                    lane_features.append(
                        [self.pad_value] * (self.max_lanes_per_direction + 3)
                    )
            per_agent_lane_features.append(lane_features)
        return per_agent_lane_features

    def get_observations(self):
        """
        Each observation includes the lane features of each incoming lane
        and the one-hot encoding of the current phase.
        Returns:
            list: List of observations for each traffic signal.
        """
        observations = []
        for ts in self.traffic_signals.values():
            observation = []
            for lane in ts.incoming_lanes:
                observation.extend(self.lane_features[lane])
                lane_results = ts.sub_results["lane"][lane]
                n_queued = lane_results["n_queued"]
                occupancy = lane_results["occupancy"]
                mean_speed = lane_results["mean_speed"]
                if lane in self.dropped_lanes:
                    observation.extend([0.0, 0.0, 0.0])
                else:
                    observation.extend([n_queued, occupancy, mean_speed])
            observation = pad_list(observation, self.get_size() - self.max_phases, -1)
            phase_id = pad_list(ts.phase_id, self.max_phases)
            observation.extend(phase_id)
            observations.append(observation)
        return observations

    def get_observation_info(self):
        """
        Get information about the observation space.
        Returns:
            dict: Dictionary containing information about the observation space.
        """
        info = {
            "lane_obs_dim": 9 + 3,
            "max_n_controlled_lanes": self.max_n_controlled_lanes,
            "max_phases": self.max_phases,
        }
        return info

    def get_size(self):
        """
        Get the size of the observation space.
        Returns:
            int: Size of the observation space.
        """
        return self.max_n_controlled_lanes * (3 + 9) + self.max_phases

    def get_state(self):
        """
        Get the state for each traffic signal.
        Each state includes the lane features of each incoming lane
        and the one-hot encoding of the current phase.
        Returns:
            list: List of states for each traffic signal.
        """
        states = []
        for ts in self.traffic_signals.values():
            state = []
            for lane in ts.incoming_lanes:
                state.extend(self.lane_features[lane])
                lane_results = ts.sub_results["lane"][lane]
                n_queued = lane_results["n_queued"]
                occupancy = lane_results["occupancy"]
                mean_speed = lane_results["mean_speed"]
                state.extend([n_queued, occupancy, mean_speed])
            state = pad_list(state, self.get_state_size() - self.max_phases, -1)
            phase_id = pad_list(ts.phase_id, self.max_phases)
            state.extend(phase_id)
            states.append(state)
        return states

    def get_state_size(self):
        """
        Get the size of the state space.
        Returns:
            int: Size of the state space.
        """
        return self.max_n_controlled_lanes * (3 + 9) + self.max_phases

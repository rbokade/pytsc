import numpy as np

from pytsc.common.utils import pad_array


class BaseObservationSpace:
    def __init__(
        self, config, parsed_network, traffic_signals, simulator_type
    ):
        self.config = config
        self.parsed_network = parsed_network
        self.traffic_signals = traffic_signals
        self.simulator_type = simulator_type
        # Needed to get observation size and padding
        self.max_n_controlled_lane = np.max(
            [len(ts.incoming_lanes) for ts in traffic_signals.values()]
        ).item()
        self.max_n_controlled_phases = np.max(
            [ts.n_phases for ts in traffic_signals.values()]
        ).item()
        self.pad_value = config.misc_config["pad_value"]

    def get_observations(self):
        observations = []
        for ts in self.traffic_signals.values():
            obs = pad_array(
                ts.norm_queue_lengths, self.max_n_controlled_lane
            ).tolist()
            obs += pad_array(
                ts.norm_densities, self.max_n_controlled_lane
            ).tolist()
            obs += pad_array(
                ts.norm_mean_speeds, self.max_n_controlled_lane
            ).tolist()
            obs += pad_array(
                ts.norm_mean_wait_times, self.max_n_controlled_lane
            ).tolist()
            obs += pad_array(
                ts.phase_id, self.max_n_controlled_phases
            ).tolist()
            obs += [ts.time_on_phase]
            obs += [ts.time_on_cycle]
            obs += [ts.sim_step]
            observations.append(obs)
        return observations

    def get_size(self):
        return int(
            4 * self.max_n_controlled_lane + self.max_n_controlled_phases + 3
        )

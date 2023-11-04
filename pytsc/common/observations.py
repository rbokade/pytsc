import numpy as np

from pytsc.common.utils import pad_array


class BaseObservationSpace:
    """
    Standard observation space for traffic signal control.
    """

    def __init__(
        self, config, parsed_network, traffic_signals, simulator_type
    ):
        self.config = config
        self.parsed_network = parsed_network
        self.n_agents = len(traffic_signals)
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
        adjacency_matrix = self.parsed_network.adjacency_matrix
        phase_ids = []
        time_on_phases = []
        time_on_cycles = []
        last_step_offsets = []
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
            # obs += pad_array(
            #     ts.phase_id, self.max_n_controlled_phases
            # ).tolist()
            # obs += [ts.time_on_phase]
            # obs += [ts.time_on_cycle]
            phase_ids.extend(ts.phase_id)
            time_on_phases.append(ts.time_on_phase)
            time_on_cycles.append(ts.time_on_cycle)
            if len(ts.offsets):
                last_step_offsets.append(ts.offsets[-1])
            else:
                last_step_offsets.append(0)
            obs += [ts.sim_step]
            observations.append(obs)
        for obs in observations:
            obs.extend(phase_ids)
            obs.extend(time_on_phases)
            obs.extend(time_on_cycles)
            obs.extend(last_step_offsets)
        return observations

    def get_size(self):
        return int(
            4 * self.max_n_controlled_lane
            + self.n_agents * (self.max_n_controlled_phases + 3)
            + 1
        )

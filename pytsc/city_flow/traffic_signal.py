import logging

from collections import deque

import numpy as np

from pytsc.common.traffic_signal import (
    BaseTSProgram,
    BaseTSController,
    BaseTrafficSignal,
)

logger = logging.getLogger(__name__)


class TSProgram(BaseTSProgram):
    start_phase_index = 0
    start_cycle_length_index = 0

    def __init__(self, id, config, simulator):
        super(TSProgram, self).__init__(id, config, simulator)
        self.engine = simulator.engine
        self._initialize_traffic_light_program()

    def _initialize_traffic_light_program(self):
        self.engine.set_tl_phase(self.id, self.phases[self.start_phase_index])
        self.set_initial_phase(self.start_phase_index)
        self.set_initial_cycle_length(self.start_cycle_length_index)


class TSController(BaseTSController):
    def __init__(self, id, config, simulator):
        super(TSController, self).__init__(id, config, simulator)
        self.program = TSProgram(id, config, simulator)
        self.engine = simulator.engine
        self._instantiate_traffic_light_logic()

    def switch_phase(self, phase_index):
        self.engine.set_tl_phase(self.id, self.phases[phase_index])
        self.program.update_current_phase(phase_index)

    def switch_cycle_length(self, cycle_length_index):
        self.program.update_current_cycle_length(cycle_length_index)


class TrafficSignal(BaseTrafficSignal):
    def __init__(self, id, config, simulator):
        super(TrafficSignal, self).__init__(id, config, simulator)
        self.config = config
        self.controller = TSController(id, config, simulator)
        self.neighborhood_matrix = (
            self.simulator.parsed_network.adjacency_matrix[
                self.simulator.parsed_network.traffic_signal_ids.index(id)
            ]
        )
        self.neighbors_lanes = self.simulator.parsed_network.neighbors_lanes[
            id
        ]
        self.neighbors_offsets = (
            self.simulator.parsed_network.neighbors_offsets[id]
        )
        # TODO: Make these deques of size `neighbors_offsets`
        self.offsets = []
        self.incoming_lane_speeds = {
            n_ts_id: deque(maxlen=int(offset))
            for n_ts_id, offset in self.neighbors_offsets.items()
        }
        self.phase_ids = []

    def update_stats(self, sub_results):
        (
            queue_lengths,
            densities,
            mean_speeds,
            norm_queue_lengths,
            norm_densities,
            norm_mean_speeds,
        ) = ([], [], [], [], [], [])
        for lane in self.incoming_lanes:
            lane_results = sub_results["lane"][lane]
            queue_lengths.append(lane_results["n_queued"])
            densities.append(lane_results["n_vehicles"])
            mean_speeds.append(lane_results["mean_speed"])
            norm_densities.append(lane_results["occupancy"])
            norm_queue_lengths.append(lane_results["norm_queue_length"])
            norm_mean_speeds.append(lane_results["norm_mean_speed"])
        self.queue_lengths = np.asarray(queue_lengths)
        self.densities = np.asarray(densities)
        self.mean_speeds = np.asarray(mean_speeds)
        self.norm_queue_lengths = np.asarray(norm_queue_lengths)
        self.norm_densities = np.asarray(norm_densities)
        self.norm_mean_speeds = np.asarray(norm_mean_speeds)
        self.norm_mean_wait_times = np.zeros_like(norm_mean_speeds)
        self.time_on_phase = self.controller.norm_time_on_phase
        self.time_on_cycle = self.controller.norm_time_on_cycle
        self.phase_id = np.asarray(self.controller.phase_one_hot)
        self.sim_step = self.simulator.sim_step / 3600
        # Save logs
        self.phase_ids.append(self.phase_id)
        phase_out_lanes = [
            pair[1]
            for pair in self.phase_to_inc_out_lanes[
                self.controller.program.current_phase
            ]
        ]
        for n_ts_id, lanes in self.neighbors_lanes.items():
            norm_mean_speed = 0
            for lane in lanes:
                lane_results = sub_results["lane"][lane]
                if (
                    lane_results["n_queued"] == 0
                    and lane_results["n_vehicles"] >= 5
                ):
                    norm_mean_speed += lane_results["norm_mean_speed"]
            norm_mean_speed /= len(lanes)
            self.incoming_lane_speeds[n_ts_id].append(norm_mean_speed)

    def action_to_phase(self, phase_index):
        self.controller.switch_phase(phase_index)

    def action_to_cycle_length(self, cycle_length_index):
        self.controller.switch_cycle_length(cycle_length_index)

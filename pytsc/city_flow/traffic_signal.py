import logging

import numpy as np

from pytsc.common.traffic_signal import (
    BaseTLSProgram,
    BaseTLSController,
    BaseTrafficSignal,
)

logger = logging.getLogger(__name__)


class TLSProgram(BaseTLSProgram):
    start_phase_index = 0

    def __init__(self, id, config, simulator):
        super(TLSProgram, self).__init__(id, config, simulator)
        self.engine = simulator.engine
        self._initialize_traffic_light_program()

    def _initialize_traffic_light_program(self):
        self.engine.set_tl_phase(self.id, self.phases[self.start_phase_index])
        self.update_current_phase(self.start_phase_index)


class TLSController(BaseTLSController):
    def __init__(self, id, config, simulator):
        super(TLSController, self).__init__(id, config, simulator)
        self.program = TLSProgram(id, config, simulator)
        self.engine = simulator.engine
        self._instantiate_traffic_light_logic()

    def switch_phase(self, phase_index):
        self._update_phase_time(phase_index)
        self._update_cycle_time()
        self.engine.set_tl_phase(self.id, self.phases[phase_index])
        self.program.update_current_phase(phase_index)
        self.logic.update_current_phase_index(phase_index, self.time_on_phase)


class TrafficSignal(BaseTrafficSignal):
    def __init__(self, id, config, simulator):
        super(TrafficSignal, self).__init__(id, config, simulator)
        self.controller = TLSController(id, config, simulator)

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

    def action_to_phase(self, action):
        self.controller.switch_phase(action)

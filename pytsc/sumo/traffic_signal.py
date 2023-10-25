import logging

import numpy as np

from pytsc.common.traffic_signal import (
    BaseTSProgram,
    BaseTSController,
    BaseTrafficSignal,
)

logger = logging.getLogger(__name__)


class TSProgram(BaseTSProgram):
    start_phase_index = 0

    def __init__(self, id, config, simulator):
        super(TSProgram, self).__init__(id, config, simulator)
        self.traci = simulator.traci.trafficlight
        self._initialize_traffic_light_program()

    @property
    def program(self):
        return self.traci.getAllProgramLogics(self.id)[0]

    def _initialize_traffic_light_program(self):
        self._set_caps_on_current_program(self.program)
        self.traci.setPhase(self.id, self.phases[self.start_phase_index])
        self.update_current_phase(self.start_phase_index)

    def _set_caps_on_current_program(self, current_program):
        phase_definitions = []
        for phase in current_program.phases:
            if "y" in phase.state:
                phase_definitions.append(
                    self.traci.Phase(
                        duration=self.yellow_time,
                        state=phase.state,
                        minDur=self.yellow_time,
                        maxDur=self.yellow_time,
                    )
                )
            else:
                phase_definitions.append(
                    self.traci.Phase(
                        duration=self.max_green_time,
                        state=phase.state,
                        minDur=self.min_green_time,
                        maxDur=self.max_green_time,
                    )
                )
        new_program = self.traci.Logic(
            programID=current_program.programID,
            type=current_program.type,
            currentPhaseIndex=current_program.currentPhaseIndex,
            phases=tuple(phase_definitions),
        )
        self.traci.setProgramLogic(self.id, new_program)


class TSController(BaseTSController):
    def __init__(self, id, config, simulator):
        super(TSController, self).__init__(id, config, simulator)
        self.program = TSProgram(id, config, simulator)
        self.traci = simulator.traci.trafficlight
        self._instantiate_traffic_light_logic()

    def switch_phase(self, phase_index):
        self._update_phase_time(phase_index)
        self._update_cycle_time()
        self.traci.setPhase(self.id, self.program.phases[phase_index])
        self.program.update_current_phase(phase_index)
        self.logic.update_current_phase_index(phase_index, self.time_on_phase)


class TrafficSignal(BaseTrafficSignal):
    def __init__(self, id, config, simulator):
        super(TrafficSignal, self).__init__(id, config, simulator)
        self.controller = TSController(id, config, simulator)

    def update_stats(self, sub_results):
        (
            queue_lengths,
            densities,
            mean_speeds,
            mean_wait_times,
            norm_queue_lengths,
            norm_mean_speeds,
            norm_mean_wait_times,
        ) = ([], [], [], [], [], [], [])
        for lane in self.incoming_lanes:
            lane_results = sub_results["lane"][lane]
            queue_lengths.append(lane_results["n_queued"])
            densities.append(lane_results["occupancy"])
            mean_speeds.append(lane_results["mean_speed"])
            mean_wait_times.append(lane_results["mean_wait_time"])
            norm_queue_lengths.append(lane_results["norm_queue_length"])
            norm_mean_speeds.append(lane_results["norm_mean_speed"])
            norm_mean_wait_times.append(lane_results["norm_mean_wait_time"])
        self.queue_lengths = queue_lengths
        self.densities = densities
        self.mean_speeds = mean_speeds
        self.mean_wait_times = mean_wait_times
        self.norm_densities = np.asarray(densities)
        self.norm_queue_lengths = np.asarray(norm_queue_lengths)
        self.norm_mean_speeds = np.asarray(norm_mean_speeds)
        self.norm_mean_wait_times = np.asarray(norm_mean_wait_times)
        self.time_on_phase = self.controller.norm_time_on_phase
        self.time_on_cycle = self.controller.norm_time_on_cycle
        self.phase_id = np.asarray(self.controller.phase_one_hot)
        self.sim_step = self.simulator.sim_step / 3600

    def action_to_phase(self, action):
        self.controller.switch_phase(action)

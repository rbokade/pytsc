from collections import deque

import numpy as np

from pytsc.common.traffic_signal import (
    BaseTSProgram,
    BaseTSController,
    BaseTrafficSignal,
)


class TSProgram(BaseTSProgram):
    start_phase_index = 0

    def __init__(self, id, config, simulator):
        super(TSProgram, self).__init__(id, config, simulator)
        self.max_green_time = config["max_green_time"]
        self.min_green_time = config["min_green_time"]
        self.traci = simulator.traci.trafficlight
        self._initialize_traffic_light_program()

    def _initialize_traffic_light_program(self):
        # self._set_caps_on_current_program(self.program)
        self.traci.setPhase(self.id, self.phases[self.start_phase_index])
        self.set_initial_phase(self.start_phase_index)

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
        self.traci.setPhase(self.id, self.config["phases"][phase_index])
        self.program.update_current_phase(phase_index)


class TrafficSignal(BaseTrafficSignal):
    def __init__(self, id, config, simulator):
        super(TrafficSignal, self).__init__(id, config, simulator)
        self.controller = TSController(id, config, simulator)
        self.incoming_lanes = config["incoming_lanes"]
        self.outgoing_lanes = config["outgoing_lanes"]
        self.position_matrices = deque(maxlen=self.config["input_n_avg"])
        # self.init_rule_based_controllers()

    def update_stats(self, sub_results):
        pos_mat = []
        for lane in self.incoming_lanes:
            lane_results = sub_results["lane"][lane]
            lane_pos_mat = np.zeros(
                self.config["visibility"], dtype=np.float32
            )
            vehicle_bin_idxs = lane_results["vehicles_bin_idxs"]
            if len(vehicle_bin_idxs):
                for i in vehicle_bin_idxs:
                    lane_pos_mat[i] = 1.0
            pos_mat.append(lane_pos_mat)
        self.position_matrices.append(np.concatenate(pos_mat, axis=0))
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
        outgoing_densities = []
        for lane in self.outgoing_lanes:
            lane_results = sub_results["lane"][lane]
            outgoing_densities.append(lane_results["occupancy"])
        self.queue_lengths = queue_lengths
        self.densities = densities
        self.mean_speeds = mean_speeds
        self.mean_wait_times = mean_wait_times
        self.norm_densities = np.asarray(densities)
        self.norm_queue_lengths = np.asarray(norm_queue_lengths)
        self.norm_mean_speeds = np.asarray(norm_mean_speeds)
        self.norm_mean_wait_times = np.asarray(norm_mean_wait_times)
        self.time_on_phase = self.controller.norm_time_on_phase
        self.phase_id = np.asarray(self.controller.phase_one_hot)
        self.pressure = np.abs(
            np.mean(densities) - np.mean(outgoing_densities)
        )
        self.sim_step = self.simulator.sim_step / 3600

    def action_to_phase(self, phase_index):
        self.controller.switch_phase(phase_index)

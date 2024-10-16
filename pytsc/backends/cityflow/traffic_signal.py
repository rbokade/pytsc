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
        self.engine = simulator.engine
        self._initialize_traffic_light_program()

    def _initialize_traffic_light_program(self):
        self.engine.set_tl_phase(self.id, self.phases[self.start_phase_index])
        self.set_initial_phase(self.start_phase_index)


class TSController(BaseTSController):
    def __init__(self, id, config, simulator):
        super(TSController, self).__init__(id, config, simulator)
        self.phases = config["phases"]
        self.program = TSProgram(id, config, simulator)
        self.engine = simulator.engine
        self._instantiate_traffic_light_logic()

    def switch_phase(self, phase_index):
        self.engine.set_tl_phase(self.id, self.phases[phase_index])
        self.program.update_current_phase(phase_index)


class TrafficSignal(BaseTrafficSignal):
    debug = False

    def __init__(self, id, config, simulator):
        super(TrafficSignal, self).__init__(id, config, simulator)
        self.config = config
        self.controller = TSController(id, config, simulator)
        self.incoming_lanes = config["incoming_lanes"]
        self.outgoing_lanes = config["outgoing_lanes"]
        self.position_matrices = deque(maxlen=self.config["input_n_avg"])
        self.speed_matrices = deque(maxlen=self.config["input_n_avg"])
        self.sub_results = None
        self.init_rule_based_controllers()

    def get_controller_action(self, controller):
        inp = self.simulator.step_measurements
        inp.update(
            {
                "time": self.simulator.sim_time,
                "current_phase_index": self.controller.program.current_phase_index,
                "time_on_phase": self.controller.time_on_phase,
            }
        )
        return self.controllers[controller].get_action(inp)

    def update_stats(self, sub_results):
        self.sub_results = sub_results
        # if self.config["observation_space"] == "position_matrix":
        pos_mat, speed_mat = [], []
        # self.lane_pos_mats, self.lane_speed_mats = {}, {}
        for lane in self.incoming_lanes:
            lane_results = sub_results["lane"][lane]
            lane_pos_mat = lane_results["position_speed_matrices"][0][
                : self.config["visibility"]
            ]
            lane_speed_mat = lane_results["position_speed_matrices"][1][
                : self.config["visibility"]
            ]
            pos_mat.append(lane_pos_mat)
            speed_mat.append(lane_speed_mat)
        for lane in self.outgoing_lanes:
            lane_results = sub_results["lane"][lane]
            lane_pos_mat = lane_results["position_speed_matrices"][0][
                -self.config["visibility"] :
            ]
            lane_speed_mat = lane_results["position_speed_matrices"][1][
                -self.config["visibility"] :
            ]
            pos_mat.append(lane_pos_mat)
            speed_mat.append(lane_speed_mat)
        self.position_matrices.append(np.concatenate(pos_mat, axis=0))
        self.speed_matrices.append(np.concatenate(speed_mat, axis=0))
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
            densities.append(lane_results["occupancy"])
            mean_speeds.append(lane_results["mean_speed"])
            norm_densities.append(lane_results["occupancy"])
            norm_queue_lengths.append(lane_results["norm_queue_length"])
            norm_mean_speeds.append(lane_results["norm_mean_speed"])
        outgoing_densities = []
        for lane in self.outgoing_lanes:
            lane_results = sub_results["lane"][lane]
            outgoing_densities.append(lane_results["occupancy"])
        self.queue_lengths = np.asarray(queue_lengths)
        self.densities = np.asarray(densities)
        self.mean_speeds = np.asarray(mean_speeds)
        self.norm_queue_lengths = np.asarray(norm_queue_lengths)
        self.norm_densities = np.asarray(norm_densities)
        self.norm_mean_speeds = np.asarray(norm_mean_speeds)
        self.norm_mean_wait_times = np.zeros_like(norm_mean_speeds)
        self.time_on_phase = self.controller.norm_time_on_phase
        self.phase_id = np.asarray(self.controller.phase_one_hot)
        self.pressure = np.abs(
            np.mean(densities) - np.mean(outgoing_densities)
        )
        self.sim_step = self.simulator.sim_step / 3600

    def action_to_phase(self, phase_index):
        self.controller.switch_phase(phase_index)

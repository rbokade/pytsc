from turtle import pos

import numpy as np

from pytsc.common.traffic_signal import (
    BaseTrafficSignal,
    BaseTSController,
    BaseTSProgram,
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
        # self.init_rule_based_controllers()

    def update_stats(self, sub_results):
        self.sub_results = sub_results
        # Compute intersection stats
        self.n_queued = 0
        self.occupancy = 0
        self.mean_speed = 0
        self.mean_delay = 0
        self.average_wait_time = 0
        self.average_travel_time = 0
        self.inc_position_matrices = {}
        max_speeds = self.simulator.parsed_network.lane_max_speeds
        for lane in self.incoming_lanes:
            lane_results = sub_results["lane"][lane]
            self.n_queued += lane_results["n_queued"]
            self.occupancy += lane_results["occupancy"]
            self.mean_speed += lane_results["mean_speed"]
            self.mean_delay += 1 - lane_results["mean_speed"] / max_speeds[lane]
            self.average_travel_time += lane_results["average_travel_time"]
            self.average_wait_time += lane_results["average_wait_time"]
            pos_mat = sub_results["lane"][lane]["position_matrix"]
            self.inc_position_matrices[lane] = pos_mat[-self.config["visibility"] :]
        self.occupancy /= len(self.incoming_lanes)
        self.mean_speed /= len(self.incoming_lanes)
        self.mean_delay /= len(self.incoming_lanes)
        self.average_travel_time /= len(self.incoming_lanes)
        self.average_wait_time /= len(self.incoming_lanes)

        self.outgoing_occupancy = 0
        self.out_position_matrices = {}
        for lane in self.outgoing_lanes:
            lane_results = sub_results["lane"][lane]
            self.outgoing_occupancy += lane_results["occupancy"]
            pos_mat = sub_results["lane"][lane]["position_matrix"]
            self.out_position_matrices[lane] = pos_mat[: self.config["visibility"]]
        self.outgoing_occupancy /= len(self.outgoing_lanes)

        self.time_on_phase = self.controller.norm_time_on_phase
        self.phase_id = self.controller.phase_one_hot
        self.pressure = np.abs(self.occupancy - self.outgoing_occupancy).item()
        self.sim_step = self.simulator.sim_step / 3600

    def action_to_phase(self, phase_index):
        self.controller.switch_phase(phase_index)

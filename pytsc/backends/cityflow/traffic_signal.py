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
        # Compute intersection stats
        self.n_queued = 0
        self.occupancy = 0
        self.mean_speed = 0
        self.mean_delay = 0
        self.inc_position_matrices = {}
        lane_max_speeds = self.simulator.parsed_network.lane_max_speeds
        for lane in self.incoming_lanes:
            lane_results = sub_results["lane"][lane]
            self.n_queued += lane_results["n_queued"]
            self.occupancy += lane_results["occupancy"]
            self.mean_speed += lane_results["mean_speed"]
            self.mean_delay += 1 - lane_results["mean_speed"] / lane_max_speeds[lane]
            pos_mat = sub_results["lane"][lane]["position_matrix"]
            self.inc_position_matrices[lane] = pos_mat[-self.config["visibility"] :]
        self.occupancy /= len(self.incoming_lanes)
        self.mean_speed /= len(self.incoming_lanes)
        self.mean_delay /= len(self.incoming_lanes)

        self.outgoing_occupancy = 0
        self.out_position_matrices = {}
        for lane in self.outgoing_lanes:
            lane_results = sub_results["lane"][lane]
            self.outgoing_occupancy += lane_results["occupancy"]
            pos_mat = sub_results["lane"][lane]["position_matrix"]
            self.out_position_matrices[lane] = pos_mat[: self.config["visibility"]]
        self.outgoing_occupancy /= len(self.outgoing_lanes)

        self.time_on_phase = self.controller.norm_time_on_phase
        self.phase_id = np.asarray(self.controller.phase_one_hot)
        self.pressure = np.abs(self.occupancy - self.outgoing_occupancy).item()
        self.sim_step = self.simulator.sim_step / 3600

    def action_to_phase(self, phase_index):
        self.controller.switch_phase(phase_index)

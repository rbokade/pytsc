import os
import sys

from pytsc.common.utils import calculate_vehicle_bin_index

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    import traci.constants as tc
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


from pytsc.common.retriever import BaseRetriever


class Retriever(BaseRetriever):
    """
    For retrieving results from TraCI subscriptions.
    """

    tc = {
        "n_vehicles": tc.LAST_STEP_VEHICLE_NUMBER,
        "n_queued": tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
        "mean_speed": tc.LAST_STEP_MEAN_SPEED,
        "occupancy": tc.LAST_STEP_OCCUPANCY,
        "wait_time": tc.VAR_WAITING_TIME,
        "travel_time": tc.VAR_CURRENT_TRAVELTIME,
        "time_step": tc.VAR_TIME,
        "n_loaded": tc.VAR_LOADED_VEHICLES_NUMBER,
        "n_departed": tc.VAR_DEPARTED_VEHICLES_NUMBER,
        "n_teleported": tc.VAR_TELEPORT_STARTING_VEHICLES_NUMBER,
        "n_arrived": tc.VAR_ARRIVED_VEHICLES_NUMBER,
        "n_colliding": tc.VAR_COLLIDING_VEHICLES_NUMBER,
        "n_emergency_brakes": tc.VAR_EMERGENCYSTOPPING_VEHICLES_NUMBER,
        "phase": tc.TL_CURRENT_PHASE,
        "vehicle_id": tc.LAST_STEP_VEHICLE_ID_LIST,
    }

    def __init__(self, simulator):
        super().__init__(simulator)
        self.traci = simulator.traci
        self.visibility = self.config.signal["visibility"]
        self.v_size = self.config.simulator["veh_size_min_gap"]
        self.lane_lengths = self.simulator.parsed_network.lane_lengths
        self.lane_max_speeds = self.simulator.parsed_network.lane_max_speeds

    def _subscribe_to_ts_vars(self):
        for ts in self.parsed_network.traffic_signal_ids:
            self.traci.trafficlight.subscribe(ts, [self.tc["phase"]])

    def _subscribe_to_lane_vars(self):
        for lane in self.parsed_network.lanes:
            self.traci.lane.subscribe(
                lane,
                [
                    self.tc["n_vehicles"],
                    self.tc["n_queued"],
                    self.tc["mean_speed"],
                    self.tc["occupancy"],
                    self.tc["wait_time"],
                    self.tc["travel_time"],
                    self.tc["vehicle_id"],
                ],
            )

    def _subscribe_to_sim_vars(self):
        self.traci.simulation.subscribe(
            [
                self.tc["time_step"],
                self.tc["n_loaded"],
                self.tc["n_departed"],
                self.tc["n_teleported"],
                self.tc["n_arrived"],
                self.tc["n_colliding"],
                self.tc["n_emergency_brakes"],
            ]
        )

    def _compute_lane_position_matrix(self, lane_sub_results, lane):
        bin_count = int(self.lane_lengths[lane] / self.v_size)
        if bin_count > 0:
            pos_mat = [-1.0] * bin_count
            for v in lane_sub_results[lane][self.tc["vehicle_id"]]:
                vehicle_position = self.traci.vehicle.getLanePosition(v)
                bin_idx = calculate_vehicle_bin_index(
                    n_bins=bin_count,
                    lane_length=self.lane_lengths[lane],
                    vehicle_position=vehicle_position,
                )
                if bin_idx is not None:
                    speed = self.traci.vehicle.getSpeed(v)
                    max_speed = self.lane_max_speeds[lane]
                    norm_speed = speed / max_speed
                    pos_mat[bin_idx] += 1.0
                    pos_mat[bin_idx] += norm_speed
            if len(pos_mat) < self.visibility:
                pos_mat += [-1.0] * (self.visibility - len(pos_mat))
        else:
            pos_mat = [-1.0] * self.visibility
        return pos_mat

    def _compute_lane_measurements(self, lane_sub_results):
        lane_measurements = {}
        for lane in self.parsed_network.lanes:
            position_matrix = self._compute_lane_position_matrix(lane_sub_results, lane)
            n_vehicles = lane_sub_results[lane][self.tc["n_vehicles"]] + 1e-6
            n_queued = lane_sub_results[lane][self.tc["n_queued"]] + 1e-6
            mean_speed = lane_sub_results[lane][self.tc["mean_speed"]] + 1e-6
            occupancy = lane_sub_results[lane][self.tc["occupancy"]] + 1e-6
            tt = lane_sub_results[lane][self.tc["travel_time"]] + 1e-6
            wt = lane_sub_results[lane][self.tc["wait_time"]] + 1e-6
            att = tt / n_vehicles
            awt = wt / n_queued if n_queued > 0 else 0
            lane_measurements[lane] = {
                "n_vehicles": n_vehicles,
                "n_queued": n_queued,
                "mean_speed": mean_speed,
                "occupancy": occupancy,
                "average_travel_time": att,
                "average_wait_time": awt,
                "position_matrix": position_matrix,
            }

        return lane_measurements

    def subscribe(self):
        self._subscribe_to_ts_vars()
        self._subscribe_to_lane_vars()
        self._subscribe_to_sim_vars()

    def retrieve_lane_measurements(self):
        lane_sub_results = {
            lane: self.traci.lane.getSubscriptionResults(lane)
            for lane in self.parsed_network.lanes
        }
        return self._compute_lane_measurements(lane_sub_results)

    def retrieve_sim_measurements(self):
        sim_sub_results = self.traci.simulation.getSubscriptionResults()
        return {
            "time_step": sim_sub_results[self.tc["time_step"]],
            "n_loaded": sim_sub_results[self.tc["n_loaded"]],
            "n_departed": sim_sub_results[self.tc["n_departed"]],
            "n_teleported": sim_sub_results[self.tc["n_teleported"]],
            "n_arrived": sim_sub_results[self.tc["n_arrived"]],
            "n_colliding": sim_sub_results[self.tc["n_colliding"]],
            "n_emergency_brakes": sim_sub_results[self.tc["n_emergency_brakes"]],
        }

    def retrieve_ts_measurements(self):
        ts_measurements = {}
        for ts in self.parsed_network.traffic_signal_ids:
            results = self.traci.trafficlight.getSubscriptionResults(ts)
            ts_measurements[ts] = {"phase": results[self.tc["phase"]]}
        return ts_measurements

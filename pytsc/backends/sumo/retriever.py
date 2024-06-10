import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    import traci.constants as tc
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from pytsc.common.retriever import BaseRetriever
from pytsc.common.utils import calculate_bin_index


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
        "average_travel_time": tc.VAR_CURRENT_TRAVELTIME,
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
                    self.tc["average_travel_time"],
                    self.tc["vehicle_id"],
                    # self.tc["position"],
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

    def subscribe(self):
        self._subscribe_to_ts_vars()
        self._subscribe_to_lane_vars()
        self._subscribe_to_sim_vars()

    def retrieve_ts_measurements(self):
        ts_measurements = {}
        for ts in self.parsed_network.traffic_signal_ids:
            results = self.traci.trafficlight.getSubscriptionResults(ts)
            ts_measurements[ts] = {"phase": results[self.tc["phase"]]}
        return ts_measurements

    def retrieve_lane_measurements(self):
        lane_measurements = {}
        for lane in self.parsed_network.lanes:
            results = self.traci.lane.getSubscriptionResults(lane)
            lane_vehicles_bin_idxs = []
            for v in results[self.tc["vehicle_id"]]:
                lane_position = self.traci.vehicle.getLanePosition(v)
                bin_idx = calculate_bin_index(
                    n_bins=self.config.signal["visibility"],
                    bin_size=self.config.simulator["veh_size_min_gap"],
                    lane_length=self.parsed_network.lane_lengths[lane],
                    lane_position=lane_position,
                )
                if bin_idx is not None:
                    speed = self.traci.vehicle.getSpeed(v)
                    norm_speed = (
                        speed / self.parsed_network.lane_max_speeds[lane]
                    )
                    lane_vehicles_bin_idxs.append((bin_idx, norm_speed))
            lane_measurements[lane] = {}
            lane_measurements[lane].update(
                {
                    "n_vehicles": results[self.tc["n_vehicles"]] + 1e-6,
                    "n_queued": results[self.tc["n_queued"]] + 1e-6,
                    "mean_speed": results[self.tc["mean_speed"]] + 1e-6,
                    "occupancy": results[self.tc["occupancy"]] + 1e-6,
                    "wait_time": results[self.tc["wait_time"]] + 1e-6,
                    "vehicles_bin_idxs": lane_vehicles_bin_idxs,
                }
            )
            lane_measurements[lane].update(
                {
                    "average_travel_time": (
                        results[self.tc["average_travel_time"]] + 1e-6
                    )
                    / lane_measurements[lane]["n_vehicles"],
                }
            )
            lane_measurements[lane].update(
                {
                    "mean_wait_time": (
                        lane_measurements[lane]["wait_time"]
                        / lane_measurements[lane]["n_queued"]
                    ),
                }
            )
            lane_measurements[lane].update(
                {
                    "norm_queue_length": (
                        lane_measurements[lane]["n_queued"]
                        / self.parsed_network.lane_lengths[lane]
                    ),
                    "norm_mean_speed": (
                        lane_measurements[lane]["mean_speed"]
                        / self.parsed_network.lane_max_speeds[lane]
                    ),
                    "norm_mean_wait_time": (
                        lane_measurements[lane]["mean_wait_time"]
                        / self.config.misc["max_wait_time"]
                    ),
                }
            )
        return lane_measurements

    def retrieve_sim_measurements(self):
        results = self.traci.simulation.getSubscriptionResults()
        sim_measurements = {
            "time_step": results[self.tc["time_step"]],
            "n_loaded": results[self.tc["n_loaded"]],
            "n_departed": results[self.tc["n_departed"]],
            "n_teleported": results[self.tc["n_teleported"]],
            "n_arrived": results[self.tc["n_arrived"]],
            "n_colliding": results[self.tc["n_colliding"]],
            "n_emergency_brakes": results[self.tc["n_emergency_brakes"]],
        }
        return sim_measurements

from pytsc.common.retriever import BaseRetriever
from pytsc.common.utils import calculate_vehicle_bin_index


class Retriever(BaseRetriever):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.engine = simulator.engine
        self.visibility = self.config.signal["visibility"]
        self.v_size = self.config.simulator["veh_size_min_gap"]
        self.lane_lengths = self.simulator.parsed_network.lane_lengths
        self.lane_max_speeds = self.simulator.parsed_network.lane_max_speeds

    def _compute_lane_position_matrix(self, lane_sub_results, lane):
        bin_count = int(self.lane_lengths[lane] / self.v_size)
        lane_vehicles = lane_sub_results["lane_vehicles"][lane]
        if bin_count > 0 and len(lane_vehicles) > 0:
            pos_mat = [-1.0] * bin_count
            for v in lane_vehicles:
                vehicle_info = self.engine.get_vehicle_info(v)
                vehicle_position = float(vehicle_info["distance"])
                bin_idx = calculate_vehicle_bin_index(
                    n_bins=bin_count,
                    lane_length=self.lane_lengths[lane],
                    vehicle_position=vehicle_position,
                )
                if bin_idx is not None:
                    norm_speed = (
                        float(vehicle_info["speed"]) / self.lane_max_speeds[lane]
                    )
                    pos_mat[bin_idx] += 1.0
                    pos_mat[bin_idx] += norm_speed
            if len(pos_mat) < self.visibility:
                pos_mat += [-1.0] * (self.visibility - len(pos_mat))
        else:
            pos_mat = [-1.0] * self.visibility
        return pos_mat

    def _compute_lane_measurements(self, lane_results):
        v_size = self.config.simulator["veh_size_min_gap"]
        lane_measurements = {}
        for lane, vehicles_on_lane in lane_results["lane_vehicles"].items():
            n_queued = lane_results["lane_n_queued"][lane]
            total_lane_speed = 0
            n_vehicles = len(vehicles_on_lane)
            if n_vehicles == 0:
                mean_speed = 0.0
            else:
                for vehicle in vehicles_on_lane:
                    total_lane_speed += lane_results["vehicle_speeds"][vehicle]
                mean_speed = total_lane_speed / n_vehicles
            lane_length = self.parsed_network.lane_lengths[lane] / v_size
            occupancy = n_vehicles / lane_length
            position_matrix = self._compute_lane_position_matrix(lane_results, lane)
            lane_measurements[lane] = {
                "n_vehicles": n_vehicles,
                "n_queued": n_queued,
                "occupancy": occupancy,
                "mean_speed": mean_speed,
                "position_matrix": position_matrix,
            }
        return lane_measurements

    def retrieve_lane_measurements(self):
        lane_results = {
            "lane_n_queued": self.engine.get_lane_waiting_vehicle_count(),
            "lane_vehicles": self.engine.get_lane_vehicles(),
            "vehicle_speeds": self.engine.get_vehicle_speed(),
        }
        return self._compute_lane_measurements(lane_results)

    def retrieve_sim_measurements(self):
        return {
            "n_vehicles": self.engine.get_vehicle_count(),
            "average_travel_time": self.engine.get_average_travel_time(),
            "time_step": self.engine.get_current_time(),
        }

    def retrieve_ts_measurements(self):
        pass

from pytsc.common.retriever import BaseRetriever


class Retriever(BaseRetriever):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.engine = simulator.engine

    def _compute_lane_measurements(self):
        v_size = self.config.cityflow_config["veh_size_min_gap"]
        lane_n_queued = self.engine.get_lane_waiting_vehicle_count()
        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_speeds = self.engine.get_vehicle_speed()
        lane_measurements = {}
        for lane_id, vehicles_on_lane in lane_vehicles.items():
            n_queued = lane_n_queued[lane_id]
            total_lane_speed = 0
            n_vehicles = len(vehicles_on_lane)
            if n_vehicles == 0:
                mean_speed = 0.0
            else:
                for vehicle_id in vehicles_on_lane:
                    total_lane_speed += vehicle_speeds[vehicle_id]
                mean_speed = total_lane_speed / n_vehicles
            lane_length = self.parsed_network.lane_lengths[lane_id] / v_size
            lane_max_speed = self.parsed_network.lane_max_speeds[lane_id]
            occupancy = n_vehicles / lane_length
            norm_queue_length = n_queued / lane_length
            norm_mean_speed = mean_speed / lane_max_speed
            lane_measurements[lane_id] = {
                "n_vehicles": n_vehicles,
                "n_queued": n_queued,
                "mean_speed": mean_speed,
                "occupancy": occupancy,
                "norm_queue_length": norm_queue_length,
                "norm_mean_speed": norm_mean_speed,
            }
        return lane_measurements

    def retrieve_lane_measurements(self):
        return self._compute_lane_measurements()

    def retrieve_sim_measurements(self):
        return {
            "n_vehicles": self.engine.get_vehicle_count(),
            "average_travel_time": self.engine.get_average_travel_time(),
            "time": self.engine.get_current_time(),
        }

    def retrieve_ts_measurements(self):
        pass

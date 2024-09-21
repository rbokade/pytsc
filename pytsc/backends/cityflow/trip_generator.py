import argparse
import json
import logging
import os
import random

import networkx as nx
import numpy as np

from pytsc.backends.cityflow.config import Config, DisruptedConfig
from pytsc.backends.cityflow.network_parser import NetworkParser
from pytsc.common.trip_generator import TripGenerator
from pytsc.common.utils import EnvLogger, generate_weibull_flow_rates

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../..",
    "scenarios",
    "cityflow",
)

# EnvLogger.set_log_level(logging.WARNING)


def detect_turn_direction(prev_road, curr_road):
    prev_split = prev_road.split('_')
    curr_split = curr_road.split('_')
    if prev_split[1] == curr_split[1]: 
        return "go_straight"
    elif int(curr_split[1]) > int(prev_split[1]): 
        return "turn_right"
    else: 
        return "turn_left"




class CityFlowTripGenerator(TripGenerator):
    turns = ["turn_left", "turn_right", "go_straight"]
    vehicle_data = {
        "length": 5.0,
        "width": 2.0,
        "maxPosAcc": 2.0,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.0,
        "usualNegAcc": 4.5,
        "minGap": 2.5,
        "maxSpeed": 11.11,
        "headwayTime": 1.5,
    }
    """
    NOTE: Traffic signal network is assumed to be a grid network.
    """

    def __init__(
        self,
        scenario,
        start_time,
        end_time,
        inter_mu,
        inter_sigma,
        disrupted=False,
        edge_weights=None,
        turn_probs=[0.1, 0.3, 0.6],
        **kwargs,
    ):
        self.scenario = scenario
        if disrupted:
            self.config = DisruptedConfig(scenario, **kwargs)
        else:
            self.config = Config(scenario)
        self.parsed_network = NetworkParser(self.config)
        self.start_time = start_time
        self.end_time = end_time
        self.inter_mu = inter_mu
        self.inter_sigma = inter_sigma
        self.turn_probabilities = turn_probs
        self.max_trip_length = self._get_max_trip_length()
        self.lane_connectivity_map = self._get_lane_connectivity_map()
        self._set_edge_weights(edge_weights)

    def _get_max_trip_length(self):
        """
        Max trip length is assumed to be the length traveled by a vehicle
        that goes from one corner of the grid to the opposite corner.
        (n - 1) + (m - 1) + 2
        """
        G = self.parsed_network._get_networkx_representation()
        try:
            return nx.diameter(G) + 2
        except:
            return len(self.parsed_network.traffic_signal_ids) + 1

    def _set_edge_weights(self, input_edge_weights=None):
        """
        Sets the edge weights for route calculation. If edge weights are not
        provided, edge weights are calculated according to the maximum allowed
        speed for each lane, such that higher the (norm) max speed of a lane
        higher the probability of the vehicle turning into that lane.
        NOTE: Edge implies a link between intersection i and j, such that it
        includes both lanes_ij and lanes_ji
        """
        self.edge_weights = {}
        if input_edge_weights is not None:
            for road in self.parsed_network.roads:
                road_id = road["id"]
                if road_id in input_edge_weights:
                    self.edge_weights[road_id] = input_edge_weights[road_id]
                else:
                    self.edge_weights[road_id] = 0.0
            return
        lane_max_speeds = self.parsed_network.lane_max_speeds
        global_max_speed = max(lane_max_speeds.values())
        for road in self.parsed_network.roads:
            lane_speeds = [
                lane_max_speeds[f"{road['id']}_{i}"] for i in range(len(road["lanes"]))
            ]
            avg_max_speed = sum(lane_speeds) / len(lane_speeds)
            normalized_weight = avg_max_speed / global_max_speed
            self.edge_weights[road["id"]] = np.round(normalized_weight, 2)

    def _find_fringe_edges(self):
        """
        NOTE: Fringe edges are based on the assumption that only one
        incoming lane (from the boundary) is connected to the intersection.
        """
        incoming_fringe_edges = []
        outgoing_fringe_edges = []
        virtual_intersections = set(
            [
                intersection["id"]
                for intersection in self.parsed_network.intersections
                if intersection["virtual"]
            ]
        )
        for road in self.parsed_network.roads:
            start_virtual = road["startIntersection"] in virtual_intersections
            end_virtual = road["endIntersection"] in virtual_intersections
            # If the road starts from a virtual intersection and ends at
            # a non-virtual, it's incoming
            if start_virtual and not end_virtual:
                incoming_fringe_edges.append(road["id"])
            # If the road starts from a non-virtual intersection and ends at
            # a virtual, it's outgoing
            if not start_virtual and end_virtual:
                outgoing_fringe_edges.append(road["id"])
        return incoming_fringe_edges, outgoing_fringe_edges

    def _get_lane_connectivity_map(self):
        lane_connectivity_map = {}
        # Logic to build the lane_connectivity_map from the road network data
        for intersection in self.parsed_network.intersections:
            for road_link in intersection.get("roadLinks", []):
                start_road = road_link["startRoad"]
                direction = road_link["type"]
                end_road = road_link["endRoad"]
                if start_road not in lane_connectivity_map:
                    lane_connectivity_map[start_road] = {}
                lane_connectivity_map[start_road][direction] = end_road
        return lane_connectivity_map

    def _choose_next_edge(self, current_edge):
        if current_edge not in self.lane_connectivity_map:
            return None
        next_edge_candidates = []
        combined_weights = []
        for i, direction in enumerate(self.turns):
            next_edge = self.lane_connectivity_map[current_edge].get(direction)
            if next_edge:
                next_edge_candidates.append(next_edge)
                combined_weights.append(
                    self.turn_probabilities[i] * self.edge_weights.get(next_edge, 1.0)
                )
        if not next_edge_candidates:
            return None
        # Normalize combined weights
        total_weight = sum(combined_weights)
        if total_weight == 0:
            return None
        normalized_weights = [w / total_weight for w in combined_weights]
        next_edge = random.choices(
            next_edge_candidates, weights=normalized_weights, k=1
        )[0]
        return next_edge

    def _generate_route(self, start_edge):
        _, outgoing_edges = self._find_fringe_edges()
        route = [start_edge]
        current_edge = start_edge
        while True:
            next_edge = self._choose_next_edge(current_edge)
            loop_attempts = 0
            while next_edge in route:
                next_edge = self._choose_next_edge(current_edge)
                loop_attempts += 1
                if loop_attempts >= len(self.turns):
                    next_edge = None
                    break
            if next_edge is None:
                break
            route.append(next_edge)
            current_edge = next_edge
            if next_edge in outgoing_edges:
                break
        return route

    def generate_flows(self, filepath, replicate_no=None):
        incoming_edges, _ = self._find_fringe_edges()
        flows = []
        for start_edge in incoming_edges:
            current_time = self.start_time
            while current_time < self.end_time:
                interarrival_time = np.random.normal(self.inter_mu, self.inter_sigma)
                interarrival_time = max(0, interarrival_time)
                vehicle_start_time = int(current_time + interarrival_time)
                if vehicle_start_time >= self.end_time:
                    break
                route = [start_edge]
                while len(route) <= 1 or len(route) > self.max_trip_length:
                    route = self._generate_route(start_edge)
                flow_entry = {
                    "vehicle": self.vehicle_data,
                    "route": route,
                    "interval": 1.0,
                    "startTime": vehicle_start_time,
                    "endTime": vehicle_start_time,
                }
                flows.append(flow_entry)
                current_time = vehicle_start_time
        sorted_flows = sorted(flows, key=lambda x: x["startTime"])
        flow_rate = (self.end_time - self.start_time) / self.inter_mu
        filename = f"{self.scenario}__gaussian_{int(flow_rate)}_flows.json"
        if "replicate_no" in self.config._additional_config:
            filename = f"{self.config._additional_config['replicate_no']}__{filename}"
        if replicate_no is not None:
            filename = f"{replicate_no}__{filename}"
        filepath = os.path.join(filepath, filename)
        with open(filepath, "w") as f:
            json.dump(sorted_flows, f, indent=4)


class IntervalCityFlowTripGenerator(CityFlowTripGenerator):
    def generate_flows(
        self,
        filepath,
        replicate_no,
        interval_duration=360,
        shape=1.5,
        scale=300,
    ):
        """
        Generate flows with the mean flow rate specified for every
        interval_duration seconds.
        """
        n_segments = int(3600 / interval_duration)
        flow_rate_segment = generate_weibull_flow_rates(
            shape, scale, self.inter_mu, n_segments
        )
        incoming_edges, _ = self._find_fringe_edges()
        flows = []
        num_intervals = (self.end_time - self.start_time) // interval_duration
        for start_edge in incoming_edges:
            current_time = self.start_time
            for i, interval in enumerate(range(num_intervals)):
                interval_mean = flow_rate_segment[i]
                while current_time < (
                    self.start_time + (interval + 1) * interval_duration
                ):
                    interarrival_time = np.random.normal(
                        interval_mean, self.inter_sigma
                    )
                    interarrival_time = max(0, interarrival_time)
                    vehicle_start_time = int(current_time + interarrival_time)
                    if vehicle_start_time >= (
                        self.start_time + (interval + 1) * interval_duration
                    ):
                        break
                    if vehicle_start_time >= self.end_time:
                        break
                    route = [start_edge]
                    while len(route) <= 1 or len(route) > self.max_trip_length:
                        route = self._generate_route(start_edge)
                    flow_entry = {
                        "vehicle": self.vehicle_data,
                        "route": route,
                        "interval": 1.0,
                        "startTime": vehicle_start_time,
                        "endTime": vehicle_start_time,
                    }
                    flows.append(flow_entry)
                    current_time = vehicle_start_time
        sorted_flows = sorted(flows, key=lambda x: x["startTime"])
        flow_rate = (self.end_time - self.start_time) / self.inter_mu
        filename = f"{self.scenario}__interval_{int(flow_rate)}_flows.json"
        if "replicate_no" in self.config._additional_config:
            filename = f"{self.config._additional_config['replicate_no']}__{filename}"
        else:
            filename = f"{replicate_no}__{filename}"
        filepath = os.path.join(filepath, filename)
        with open(filepath, "w") as f:
            json.dump(sorted_flows, f, indent=4)


class VariableDemandTripGenerator(CityFlowTripGenerator):
    def __init__(
        self,
        scenario,
        start_time,
        end_time,
        inter_mus,
        inter_sigmas,
        edge_weights,
        disrupted=False,
        turn_probs=[1 / 3, 1 / 3, 1 / 3],
        **kwargs,
    ):
        self.scenario = scenario
        if disrupted:
            self.config = DisruptedConfig(scenario, **kwargs)
        else:
            self.config = Config(scenario)
        self.parsed_network = NetworkParser(self.config)
        self.start_time = start_time
        self.end_time = end_time
        self.inter_mus = inter_mus
        self.inter_sigmas = inter_sigmas
        self.turn_probabilities = turn_probs
        self.max_trip_length = self._get_max_trip_length()
        self.lane_connectivity_map = self._get_lane_connectivity_map()
        self._set_edge_weights(edge_weights)
        self.demand_profile = [
            0.5,
            0.6,
            0.75,
            1.0,
            1.0,
            0.5,
            0.5,
            0.3,
            0.3,
            1e-6,
        ]

    def _get_interarrival_time(self, edge_id, current_time):
        time_slot = (current_time % 3600) // 600
        mu = self.inter_mus[edge_id] / self.demand_profile[time_slot]
        sigma = self.inter_sigmas[edge_id] / self.demand_profile[time_slot]
        interarrival_time = np.random.normal(mu, sigma)
        return max(0, interarrival_time)

    def generate_flows(self, filepath, replicate_no=None):
        incoming_edges, _ = self._find_fringe_edges()
        flows = []
        for start_edge in incoming_edges:
            if start_edge in self.inter_mus.keys():
                current_time = self.start_time
                while current_time < self.end_time:
                    interarrival_time = self._get_interarrival_time(
                        start_edge, current_time
                    )
                    vehicle_start_time = int(current_time + interarrival_time)
                    if vehicle_start_time >= self.end_time:
                        break
                    route = [start_edge]
                    while len(route) <= 1 or len(route) > self.max_trip_length:
                        route = self._generate_route(start_edge)
                    flow_entry = {
                        "vehicle": self.vehicle_data,
                        "route": route,
                        "interval": 1.0,
                        "startTime": vehicle_start_time,
                        "endTime": vehicle_start_time,
                    }
                    flows.append(flow_entry)
                    current_time = vehicle_start_time
        sorted_flows = sorted(flows, key=lambda x: x["startTime"])
        flow_rate = (self.end_time - self.start_time) / np.mean(
            list(self.inter_mus.values())
        )
        filename = f"{self.scenario}__gaussian_{int(flow_rate)}_flows.json"
        if "replicate_no" in self.config._additional_config:
            filename = f"{self.config._additional_config['replicate_no']}__{filename}"
        if replicate_no is not None:
            filename = f"{replicate_no}__{filename}"
        filepath = os.path.join(filepath, filename)
        with open(filepath, "w") as f:
            json.dump(sorted_flows, f, indent=4)


class CityFlowOneWayTripGenerator(CityFlowTripGenerator):
    def __init__(
        self,
        scenario,
        start_time,
        end_time,
        inter_mu_ns,
        inter_sigma_ns,
        inter_mu_ew,
        inter_sigma_ew,
        disrupted=False,
        edge_weights=None,
        **kwargs,
    ):
        self.inter_mu_ns = inter_mu_ns
        self.inter_sigma_ns = inter_sigma_ns
        self.inter_mu_ew = inter_mu_ew
        self.inter_sigma_ew = inter_sigma_ew
        super(CityFlowOneWayTripGenerator, self).__init__(
            scenario,
            start_time,
            end_time,
            inter_mu_ns,  # Default values for the parent's constructor
            inter_sigma_ns,  # Default values for the parent's constructor
            disrupted=disrupted,
            edge_weights=edge_weights,
            turn_probs=[0.0, 0.0, 1.0],
            **kwargs,
        )

    def _is_ns_road(self, road):
        start_point = road["points"][0]
        end_point = road["points"][-1]
        return start_point["x"] == end_point["x"] and start_point["y"] > end_point["y"]

    def _is_ew_road(self, road):
        start_point = road["points"][0]
        end_point = road["points"][-1]
        return start_point["y"] == end_point["y"] and start_point["x"] > end_point["x"]

    def generate_flows(self, filepath, replicate_no=None):
        incoming_edges, _ = self._find_fringe_edges()
        ns_edges = [
            road["id"]
            for road in self.parsed_network.roads
            if self._is_ns_road(road) and road["id"] in incoming_edges
        ]
        ew_edges = [
            road["id"]
            for road in self.parsed_network.roads
            if self._is_ew_road(road) and road["id"] in incoming_edges
        ]
        flows = []

        def generate_flow_for_edges(edges, inter_mu, inter_sigma):
            for start_edge in edges:
                current_time = self.start_time
                while current_time < self.end_time:
                    interarrival_time = np.random.normal(inter_mu, inter_sigma)
                    interarrival_time = max(0, interarrival_time)
                    vehicle_start_time = int(current_time + interarrival_time)
                    if vehicle_start_time >= self.end_time:
                        break
                    route = [start_edge]
                    while len(route) <= 1 or len(route) > self.max_trip_length:
                        route = self._generate_route(start_edge)
                    flow_entry = {
                        "vehicle": self.vehicle_data,
                        "route": route,
                        "interval": 1.0,
                        "startTime": vehicle_start_time,
                        "endTime": vehicle_start_time,
                    }
                    flows.append(flow_entry)
                    current_time = vehicle_start_time

        generate_flow_for_edges(ns_edges, self.inter_mu_ns, self.inter_sigma_ns)
        generate_flow_for_edges(ew_edges, self.inter_mu_ew, self.inter_sigma_ew)
        sorted_flows = sorted(flows, key=lambda x: x["startTime"])
        flow_rate_ns = (self.end_time - self.start_time) / self.inter_mu_ns
        flow_rate_ew = (self.end_time - self.start_time) / self.inter_mu_ew
        filename = f"{self.scenario}__oneway_{int(flow_rate_ns)}_NS_{int(flow_rate_ew)}_EW_flows.json"
        if "replicate_no" in self.config._additional_config:
            filename = f"{self.config._additional_config['replicate_no']}__{filename}"
        if replicate_no is not None:
            filename = f"{replicate_no}__{filename}"
        filepath = os.path.join(filepath, filename)
        with open(filepath, "w") as f:
            json.dump(sorted_flows, f, indent=4)


class CityFlowRandomizedTripGenerator(CityFlowTripGenerator):
    turns = ["turn_left", "turn_right", "go_straight"]
    vehicle_data = {
        "length": 5.0,
        "width": 2.0,
        "maxPosAcc": 2.0,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.0,
        "usualNegAcc": 4.5,
        "minGap": 2.5,
        "maxSpeed": 11.11,
        "headwayTime": 1.5,
    }
    
    def __init__(self, scenario, start_time, end_time, **kwargs):
        self.scenario = scenario
        self.config = Config(scenario)
        self.parsed_network = NetworkParser(self.config)
        self.start_time = start_time
        self.end_time = end_time
        self.lane_connectivity_map = self._get_lane_connectivity_map()
        turn_ratios, self.flow_info, self.stored_routes, self.route_proportions = self.get_flow_rates()
        self.turn_probabilities = [
            turn_ratios["turn_left"], turn_ratios["turn_right"], turn_ratios["go_straight"]
        ]
        self.max_trip_length = self._get_max_trip_length()
        self._set_edge_weights(None)

    def _get_max_trip_length(self):
        return max([v['max_route_length'] for v in self.flow_info.values()])

    def get_flow_rates(self):
        flow_file_dir = os.path.join(CONFIG_DIR, self.scenario, self.config.simulator['flow_file'])
        with open(flow_file_dir, "r") as flow_file:
            flow_data = json.load(flow_file)
        
        road_counts = {}
        start_times = []
        road_start_times = {}
        route_lengths = {}
        stored_routes = {}
        total_routes_per_start = {}  # New dictionary to store total routes per start road
        turning_ratios = {"go_straight": 0, "turn_right": 0, "turn_left": 0}

        for vehicle in flow_data:
            route = vehicle["route"]
            start_road = route[0]
            
            # Initialize stored_routes and total_routes_per_start
            if start_road not in stored_routes:
                stored_routes[start_road] = []
            if start_road not in total_routes_per_start:
                total_routes_per_start[start_road] = {}
            
            # Count the route occurrences for each start_road
            if tuple(route) not in total_routes_per_start[start_road]:
                total_routes_per_start[start_road][tuple(route)] = 0
            total_routes_per_start[start_road][tuple(route)] += 1
            
            if route not in stored_routes[start_road]:
                stored_routes[start_road].append(route)

            start_time = vehicle["startTime"]
            road_counts[start_road] = road_counts.get(start_road, 0) + 1
            start_times.append(start_time)

            if start_road not in road_start_times:
                road_start_times[start_road] = []
            if start_road not in route_lengths:
                route_lengths[start_road] = []
            road_start_times[start_road].append(start_time)
            route_lengths[start_road].append(len(route))

            # Update turning ratios
            for i in range(1, len(route)):
                turn_direction = detect_turn_direction(route[i-1], route[i])
                turning_ratios[turn_direction] += 1
        
        # Normalize the route proportions for each start road
        route_proportions = {}
        for start_road, route_count in total_routes_per_start.items():
            road_total_routes = sum(route_count.values())
            route_proportions[start_road] = {route: count / road_total_routes for route, count in route_count.items()}
        
        turning_ratios = {k: v / sum(turning_ratios.values()) for k, v in turning_ratios.items()}
        total_time_hours = (max(start_times) - min(start_times)) / 3600
        flow_rates = {road: count / total_time_hours for road, count in road_counts.items()}
        road_arrival_diff_stats = {}
        for road, times in road_start_times.items():
            diffs = np.diff(sorted(times))
            road_arrival_diff_stats[road] = (np.mean(diffs), np.std(diffs))
        
        combined_results = {}
        for road in road_counts:
            combined_results[road] = {
                "flow_rate": flow_rates[road],
                "arrival_diff_mean": road_arrival_diff_stats[road][0],
                "arrival_diff_std": road_arrival_diff_stats[road][1],
                "mean_route_length": np.mean(route_lengths[road]),
                "min_route_length": np.min(route_lengths[road]),
                "max_route_length": np.max(route_lengths[road]),
            }
        
        return turning_ratios, combined_results, stored_routes, route_proportions

    def generate_flows(self, filepath, replicate_no=None):
        incoming_edges, _ = self._find_fringe_edges()
        flows = []
        total_route_length = 0 
        num_routes = 0  
        target_mean_route_length = sum(v['mean_route_length'] for v in self.flow_info.values()) / len(self.flow_info)

        for start_edge in incoming_edges:
            if start_edge not in self.flow_info:
                continue
            current_time = self.start_time
            while current_time < self.end_time:
                interarrival_time = np.random.normal(
                    self.flow_info[start_edge]['arrival_diff_mean'], 
                    self.flow_info[start_edge]['arrival_diff_std'],
                )
                interarrival_time = max(0, interarrival_time)
                vehicle_start_time = int(current_time + interarrival_time)
                if vehicle_start_time >= self.end_time:
                    break
                
                if start_edge in self.stored_routes:
                    routes = self.stored_routes[start_edge]
                    
                    # Extract weights from route_proportions
                    weights = [self.route_proportions[start_edge][tuple(route)] for route in routes]
                    
                    # Use weights to select a route
                    route = random.choices(routes, weights=weights, k=1)[0]
                    
                    flow_entry = {
                        "vehicle": self.vehicle_data,
                        "route": route,
                        "interval": 1.0,
                        "startTime": vehicle_start_time,
                        "endTime": vehicle_start_time,
                    }
                    flows.append(flow_entry)
                    total_route_length += len(route)
                    num_routes += 1
                    current_time = vehicle_start_time

        sorted_flows = sorted(flows, key=lambda x: x["startTime"])
        filename = f"{self.scenario}__gaussian_flows.json"
        if "replicate_no" in self.config._additional_config:
            filename = f"{self.config._additional_config['replicate_no']}__{filename}"
        if replicate_no is not None:
            filename = f"{replicate_no}__{filename}"
        os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(filepath, filename)
        with open(filepath, "w") as f:
            json.dump(sorted_flows, f, indent=4)
        
        final_mean_route_length = total_route_length / num_routes if num_routes > 0 else 0
        EnvLogger.log_info(
            f"Final mean route length: {final_mean_route_length}, Target: {target_mean_route_length}"
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=int,
        default="1x1_cityflow_grid",
        help="Name of the scenario",
    )
    parser.add_argument(
        "--flow-rate-mean",
        dest="flow_rate_mean",
        type=float,
        default=600,
        help="Mean flow rate per incoming lane",
    )
    parser.add_argument(
        "--flow-rate-sigma",
        dest="flow_rate_sigma",
        type=float,
        default=0.8,
        help="Std of the flow rate",
    )
    args = parser.parse_args()

    start_time = 0
    end_time = 3600
    mu = end_time / args.flow_rate_mean
    turn_probs = [0.1, 0.3, 0.6]

    flow_generator = CityFlowTripGenerator(
        scenario=args.scenario,
        start_time=start_time,
        end_time=end_time,
        inter_mu=mu,
        inter_sigma=args.flow_rate_sigma,
        turn_probs=turn_probs,
    )
    flow_generator.generate_flows(filepath="/Users/rohitbokade/repos/pytsc/pytsc/tests")



import argparse
import json
import os
import random

import networkx as nx
import numpy as np

from pytsc.backends.cityflow.config import Config
from pytsc.backends.cityflow.network_parser import NetworkParser
from pytsc.common.trip_generator import TripGenerator


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
        edge_weights=None,
        turn_probs=[0.1, 0.3, 0.6],
    ):
        self.scenario = scenario
        self.config = Config(scenario)
        self.parsed_network = NetworkParser(self.config)
        self.start_time = start_time
        self.end_time = end_time
        self.inter_mu = inter_mu
        self.inter_sigma = inter_sigma
        self.turn_probabilities = turn_probs
        self.max_trip_length = self._get_max_trip_length()
        print(f"Max trip length: {self.max_trip_length}")
        self.lane_connectivity_map = self._get_lane_connectivity_map()
        self._set_edge_weights(edge_weights)

    def _get_max_trip_length(self):
        """
        Max trip length is assumed to be the length traveled by a vehicle
        that goes from one corner of the grid to the opposite corner.
        (n - 1) + (m - 1) + 2
        """
        G = self.parsed_network._get_networkx_representation()
        return nx.diameter(G) + 2

    def _set_edge_weights(self, input_edge_weights=None):
        """
        Sets the edge weights for route calculation. If edge weights are not
        provided, edge weights are calculated according to the maximum allowed
        speed for each lane, such that higher the (norm) max speed of a lane
        higher the probability of the vehicle turning into that lane.
        NOTE: Edge implies a link between intersection i and j, such that it
        includes both lanes_ij and lanes_ji
        """
        if input_edge_weights is not None:
            self.edge_weights = input_edge_weights
            return
        self.edge_weights = {}
        lane_max_speeds = self.parsed_network.lane_max_speeds
        global_max_speed = max(lane_max_speeds.values())
        for road in self.parsed_network.roads:
            lane_speeds = [
                lane_max_speeds[f"{road['id']}_{i}"]
                for i in range(len(road["lanes"]))
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
                    self.turn_probabilities[i]
                    * self.edge_weights.get(next_edge, 1.0)
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

    def generate_flows(self, filepath):
        incoming_edges, _ = self._find_fringe_edges()
        flows = []
        for start_edge in incoming_edges:
            current_time = self.start_time
            while current_time < self.end_time:
                interarrival_time = np.random.normal(
                    self.inter_mu, self.inter_sigma
                )
                interarrival_time = max(0, interarrival_time)
                vehicle_start_time = int(current_time + interarrival_time)
                if vehicle_start_time >= self.end_time:
                    break
                route = [start_edge]
                while len(route) <= 2 or len(route) > self.max_trip_length:
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
        filename = os.path.join(
            filepath,
            f"{self.scenario}__gaussian_flow_rate_{int(flow_rate)}_flows.json",
        )
        with open(filename, "w") as f:
            json.dump(sorted_flows, f, indent=4)


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
    flow_generator.generate_flows(
        filepath="/home/rohitbokade/repos/pytsc/pytsc/tests"
    )

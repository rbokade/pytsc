import json
import os
import random

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

    def __init__(
        self,
        scenario,
        start_time,
        end_time,
        inter_mu,
        inter_sigma,
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
        lane_connectivity_map = self._get_lane_connectivity_map()
        if current_edge not in lane_connectivity_map:
            return None
        chosen_direction = random.choices(
            self.turns, weights=self.turn_probabilities, k=1
        )[0]
        next_edge = lane_connectivity_map[current_edge].get(
            chosen_direction, None
        )
        return next_edge

    def _generate_route(self, start_edge):
        _, outgoing_edges = self._find_fringe_edges()
        route = [start_edge]
        current_edge = start_edge
        while True:
            next_edge = self._choose_next_edge(current_edge)
            if not next_edge or next_edge in outgoing_edges:
                break
            route.append(next_edge)
            current_edge = next_edge
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
                route = self._generate_route(start_edge)
                flow_entry = {
                    "vehicle": self.vehicle_data,
                    "route": route,
                    "startTime": vehicle_start_time,
                    "endTime": vehicle_start_time,
                }
                flows.append(flow_entry)
                current_time = vehicle_start_time
        sorted_flows = sorted(flows, key=lambda x: x["startTime"])
        flow_rate = (self.end_time - self.start_time) / self.inter_mu
        filename = os.path.join(
            filepath, f"{self.scenario}__flow_rate_{flow_rate}_flows.json"
        )
        with open(filename, "w") as f:
            json.dump(sorted_flows, f, indent=4)

import argparse
import json
import os

import numpy as np


class FlowGenerator:
    def __init__(self, roadnet_file, mean_flow_rate):
        self.mean_flow_rate = mean_flow_rate
        self.roadnet = json.load(open(roadnet_file, "r"))
        self.fringe_roads = self._find_fringe_roads()

    def _find_fringe_roads(self):
        fringe_roads = []
        for road in self.roadnet["roads"]:
            if self.is_fringe_intersection(
                road["startIntersection"]
            ) or self.is_fringe_intersection(road["endIntersection"]):
                fringe_roads.append(road)
        return fringe_roads

    def is_fringe_intersection(self, intersection_id):
        intersection = next(
            (
                i
                for i in self.roadnet["intersections"]
                if i["id"] == intersection_id
            ),
            None,
        )
        return intersection and intersection["virtual"]

    def _find_route(self, intersection_id, direction):
        pass

    def _find_route_to_exit(self, intersection_id):
        pass

    def _generate_vehicles(self, route):
        # Sample the number of vehicles from a Gaussian distribution
        number_of_vehicles = int(
            np.random.normal(loc=self.mean_flow_rate, scale=50) / 3600
        )
        vehicles = []
        for i in range(number_of_vehicles):
            vehicle = {
                "route": route,
                "departureTime": i,
                # Other vehicle attributes
            }
            vehicles.append(vehicle)
        return vehicles

    def _choose_next_edge(self, current_edge):
        lane_connectivity_map = self._get_lane_connectivity_map()
        if current_edge in lane_connectivity_map:
            movement_options = lane_connectivity_map[current_edge]
            # Now, instead of randomly choosing based on turn_probabilities,
            # you check the available movements from current_edge
            available_moves = [
                move for move in self.turns if movement_options[move]
            ]
            if not available_moves:
                return None
            chosen_move = random.choice(available_moves)
            next_edges = movement_options[chosen_move]
            # If there are multiple options for a move, randomly select one.
            next_edge = random.choice(next_edges) if next_edges else None
            return next_edge
        return None

    def _generate_route(self, start_edge):
        _, outgoing_edges = self._find_fringe_roads()
        route = [start_edge]
        current_edge = start_edge

        while True:
            next_edge = self._choose_next_edge(current_edge)
            loop_attempts = 0
            while next_edge in route:
                next_edge = self._choose_next_edge(current_edge)
                loop_attempts += 1
                # If too many attempts have been made, stop trying to find a new edge
                if loop_attempts > len(self.turns):
                    next_edge = None  # Indicate failure to find a non-looping next edge
                    break

            # If next_edge is None, it means a suitable next edge wasn't found; break the loop
            if next_edge is None:
                break

            # Append next_edge to route
            route.append(next_edge)
            current_edge = next_edge

            # If the next_edge is an outgoing edge, it's a valid termination point; exit the loop
            if next_edge in outgoing_edges:
                break

        return route

    def generate_flow(self):
        flow = []
        for road in self.fringe_roads:
            for lane_index in range(3):
                route = self._generate_route(road, lane_index)
                vehicles = self._generate_vehicles(route)
                flow.extend(vehicles)
        return flow

    def generate_flow_file(self, output_file):
        flows = self.generate_flow()
        flow_data = {"flows": flows}
        with open(output_file, "w") as file:
            json.dump(flow_data, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--road-net",
        dest="roadnet_file",
        help="Path to the roadnet json file",
        type=str,
        default="/home/rohitbokade/pytsc/pytsc/scenarios/syn_1x2_random/roadnet_1X2.json",
    )
    parser.add_argument(
        "--mean-flow-rate",
        dest="mean_flow_rate",
        help="Mean flow rate (vehicles/hour/lane)",
        type=float,
        default=500,
    )
    args = parser.parse_args()

    scenarios_path = os.path.dirname(args.roadnet_file)
    output_file = os.path.join(
        scenarios_path,
        f"{os.path.basename(args.roadnet_file)}_{args.mean_flow_rate}_1h.json",
    )

    trip_generator = FlowGenerator(args.roadnet_file, args.mean_flow_rate)
    trip_generator.generate_flow_file(output_file)

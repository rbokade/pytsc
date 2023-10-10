import argparse
import json
import os

import numpy as np


class FlowGenerator:
    def __init__(self, roadnet_file):
        self.mean_flow_rate = 500
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
                for i in self.network["intersections"]
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

    def generate_flow(self):
        flow = []
        for road in self.fringe_roads:
            for lane_index in range(3):  # Three lanes per approach
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

import json
import os
import random

from pytsc.backends.cityflow.config import Config
from pytsc.backends.cityflow.network_parser import NetworkParser

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../..",
    "scenarios",
    "cityflow",
)


class CityFlowGridDisruptor:
    """
    Disruptor for CityFlow grid networks.
    This class modifies the speed of lanes in a grid network based on a specified
    disruption ratio and speed reduction factor.
    
    Args:
        scenario (str): The name of the scenario.
        disruption_ratio (float): The ratio of traffic signals to disrupt.
        speed_reduction_factor (float): The factor by which to reduce the speed of disrupted lanes.
        replicate_no (int): The replicate number for the disrupted network.
    """

    def __init__(
        self, scenario, disruption_ratio, speed_reduction_factor, replicate_no
    ):
        self.scenario = scenario
        self.disruption_ratio = disruption_ratio
        self.speed_reduction_factor = speed_reduction_factor
        self.replicate_no = replicate_no
        self.config = Config(scenario)
        self.parsed_network = NetworkParser(self.config)
        self.roadnet = self._load_roadnet()
        self._create_disrupted_scenario_folder()

    def _load_roadnet(self):
        """
        Load the roadnet file for the specified scenario.

        Returns:
            dict: Parsed roadnet data.
        """
        roadnet_dir = os.path.join(
            CONFIG_DIR, self.scenario, self.config.simulator["roadnet_file"]
        )
        with open(roadnet_dir, "r") as f:
            return json.load(f)

    def _create_disrupted_scenario_folder(self):
        """
        Create a directory for the disrupted scenario files.
        """
        scenario_dir = os.path.join(CONFIG_DIR, self.scenario)
        settings = f"r_{self.disruption_ratio}"
        settings += "__"
        settings += f"p_{self.speed_reduction_factor}"
        self.disrupted_scenario_dir = os.path.join(scenario_dir, "disrupted", settings)
        os.makedirs(self.disrupted_scenario_dir, exist_ok=True)

    def _select_edges_to_disrupt(self):
        """
        Select edges to disrupt based on the disruption ratio.

        Returns:
            list: List of edges to disrupt.
        """
        n_traffic_signals = len(self.parsed_network.traffic_signal_ids)
        n_edges_to_disrupt = int(n_traffic_signals * self.disruption_ratio)
        # Ensure all traffic signals have at least one disrupted link
        disruptable_edges = []
        used_intersections = set()
        for road in self.roadnet["roads"]:
            start, end = road["startIntersection"], road["endIntersection"]
            if (
                start in self.parsed_network.traffic_signal_ids
                and end in self.parsed_network.traffic_signal_ids
            ):  # ensure it is not a fringe edge
                if start not in used_intersections:
                    disruptable_edges.append(road["id"])
                    used_intersections.add(start)
                elif end not in used_intersections:
                    disruptable_edges.append(road["id"])
                    used_intersections.add(end)
        disrupted_edges = random.sample(disruptable_edges, n_edges_to_disrupt)
        return disrupted_edges

    def _lower_the_speed_of_disrupted_lanes(self, edges_to_disrupt):
        """
        Lower the speed of lanes in the disrupted edges.

        Args:
            edges_to_disrupt (list): List of edges to disrupt.
        """
        for road in self.roadnet["roads"]:
            for edge in edges_to_disrupt:
                if edge == road["id"]:
                    for lane in road["lanes"]:
                        max_speed = lane["maxSpeed"]
                        lane["maxSpeed"] = max_speed * self.speed_reduction_factor

    def _save_disrupted_network(self):
        """
        Save the disrupted roadnet to a file.
        """
        filename = f"{self.replicate_no}__{self.config.simulator['roadnet_file']}"
        output_file = os.path.join(self.disrupted_scenario_dir, filename)
        with open(output_file, "w") as f:
            json.dump(self.roadnet, f, indent=4)

    def generate_disrupted_network(self):
        """
        Generate the disrupted network by selecting edges to disrupt,
        lowering their speed, and saving the disrupted network to a file.
        """
        edges_to_disrupt = self._select_edges_to_disrupt()
        self._lower_the_speed_of_disrupted_lanes(edges_to_disrupt)
        self._save_disrupted_network()

import argparse
import os
import random
import sys
import yaml

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    from sumolib.net import readNet
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# from pytsc.sumo.config import Config

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "scenarios",
)


class SUMOGridGenerator:
    """
    Creates grid network using SUMO's netgenerate tool

    First create a scenario folder in the scenarios directory and add the configs in the `config.yaml` file
    """

    _detector_distance_to_tls = 0.1

    def __init__(self, scenario):
        self.scenario = scenario
        # self.config = Config(scenario, add_config={})
        self.config = self._load_config()
        self.data_dir = os.path.join(CONFIG_DIR, "sumo", scenario)
        self.netfile_dir = os.path.join(
            self.data_dir, f"{self.scenario}.net.xml"
        )
        self.cfg_file_dir = os.path.join(
            self.data_dir, f"{self.scenario}.sumocfg"
        )
        grid_generator_config = self.config["grid_generator"]
        signal_config = self.config["signal"]

        self.grid_length = grid_generator_config["grid_length"]
        self.grid_x_number = grid_generator_config["grid_x_number"]
        self.grid_y_number = grid_generator_config["grid_y_number"]
        self.grid_attach_length = grid_generator_config["grid_attach_length"]
        self.add_turn_lane = grid_generator_config["add_turn_lane"]
        self.max_green_time = signal_config["max_green_time"]
        self.yellow_time = signal_config["yellow_time"]
        self.turn_lane_length = max(self.grid_length, self.grid_attach_length)
        self.vision = signal_config["vision"]

    def _load_config(self):
        scenario_file_path = os.path.join(
            CONFIG_DIR, "sumo", self.scenario, "config.yaml"
        )
        if os.path.exists(scenario_file_path):
            with open(scenario_file_path, "r") as f:
                scenario_config = yaml.safe_load(f)
        return scenario_config

    def generate_grid(self, regular=True):
        gridgen = "netgenerate --grid"
        gridgen += f" --grid.x-number {self.grid_x_number}"
        gridgen += f" --grid.y-number {self.grid_y_number}"
        gridgen += f" --grid.length {self.grid_length}"
        gridgen += f" --grid.attach-length {self.grid_attach_length}"
        gridgen += f" --output-file {self.netfile_dir}"
        # gridgen += f" --save-configuration {self.cfg_file_dir}"
        gridgen += " --default.lanenumber 2"
        gridgen += " --no-turnarounds"
        gridgen += " --tls.guess"
        gridgen += " --tls.guess.threshold 50"
        gridgen += " --tls.group-signals True"
        gridgen += " --tls.default-type delay_based"
        gridgen += f" --tls.green.time {self.max_green_time}"
        gridgen += f" --tls.yellow.time {self.yellow_time}"
        if self.add_turn_lane:
            gridgen += f" --turn-lanes.length {self.turn_lane_length}"
            gridgen += " --turn-lanes 1"
        if not regular:
            edges_to_be_removed = self._get_edges_to_be_removed()
            gridgen += f" --remove {edges_to_be_removed}"
        print(gridgen)
        os.system(gridgen)
        self._generate_lane_area_detectors()

    def _generate_lane_area_detectors(self):
        script = tools + "/output/generateTLSE2Detectors.py"
        additional_files_dir = os.path.join(
            self.data_dir, f"{self.scenario}.add.xml"
        )
        ladgen = f"python {script}"
        ladgen += f" --net-file {self.netfile_dir}"
        ladgen += f" --output {additional_files_dir}"
        ladgen += f" --detector-length {self.vision}"
        ladgen += " --frequency 1"
        print(ladgen)
        os.system(ladgen)

    def _get_edges_to_be_removed(self, n=2):
        self.generate_grid()
        self.net = readNet(self.net_file)
        edges = self.net.getEdges()
        edge_pairs_without_fringes = self._filter_out_fringes(edges)
        edges_to_be_removed = random.sample(edge_pairs_without_fringes, n)
        edges_to_be_removed_str = self._convert_edge_list_to_str(
            edges_to_be_removed
        )
        os.remove(self.net_file)
        return edges_to_be_removed_str

    def _convert_edge_list_to_str(self, edge_pairs_list):
        edges_str = ""
        for edge_pair in edge_pairs_list:
            for edge in edge_pair:
                edges_str += ", " + edge
        return edges_str[2:]

    def _filter_out_fringes(self, edges):
        filtered_edges = []
        children = []
        for child in edges:
            if not child.is_fringe() and child._id not in children:
                _to = child._to._id
                _from = child._from._id
                filtered_edges.append([_to + _from, _from + _to])
                children.extend([_to + _from, _from + _to])
        return filtered_edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        help="scenario name",
        type=str,
        default="2x1_regular_grid",
    )
    args = parser.parse_args()

    grid = SUMOGridGenerator(args.scenario)
    grid.generate_grid()

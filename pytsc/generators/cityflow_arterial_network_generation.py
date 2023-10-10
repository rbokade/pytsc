import argparse
import os
import json


CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "scenarios",
)
CITYFLOW_GRID_GEN_SCRIPT = (
    "/home/rohitbokade/CityFlow/tools/generator/generate_grid_scenario.py"
)


class ArterialNetwork:
    nrows = 1
    col_distance = 300
    row_distance = 300
    intersection_width = 20
    n_left_lanes = 0
    n_right_lanes = 0
    n_straight_lanes = 1
    lane_max_speed = 10
    veh_max_speed = 10
    veh_min_gap = 2.5
    veh_headway_time = 2

    def __init__(self, ncols, mean_flow_rates=(500, 600, 700)):
        self.ncols = ncols
        self.mean_flow_rates = mean_flow_rates

    def _create_scenario_folder(self):
        self.scenario_dir = os.path.join(
            CONFIG_DIR, f"{self.nrows}x{self.ncols}_cityflow_grid"
        )
        os.makedirs(self.scenario_dir, exist_ok=True)

    def _generate_roadnet_file(
        self, mean_flow_rate, update_side_street_interval=True
    ):
        self._create_scenario_folder()
        self.roadnet_file = os.path.join(
            self.scenario_dir, f"{self.nrows}x{self.ncols}_roadnet.json"
        )
        self.flow_file = os.path.join(
            self.scenario_dir,
            f"{self.nrows}x{self.ncols}_roadnet_{mean_flow_rate}.json",
        )
        arterial_interval = 3600 / mean_flow_rate
        side_street_interval = 3600 / (mean_flow_rate * 3 / 5)
        gridgen = f"python {CITYFLOW_GRID_GEN_SCRIPT}"
        gridgen += f" {self.nrows} {self.ncols}"
        gridgen += f" --rowDistance={self.row_distance}"
        gridgen += f" --columnDistance={self.col_distance}"
        gridgen += f" --intersectionWidth={self.intersection_width}"
        gridgen += f" --numLeftLanes={self.n_left_lanes}"
        gridgen += f" --numRightLanes={self.n_right_lanes}"
        gridgen += f" --numStraightLanes={self.n_straight_lanes}"
        gridgen += f" --laneMaxSpeed={self.lane_max_speed}"
        gridgen += f" --vehMinGap={self.veh_min_gap}"
        gridgen += f" --vehMaxSpeed={self.veh_max_speed}"
        gridgen += f" --vehHeadwayTime={self.veh_headway_time}"
        gridgen += f" --flowFile {self.flow_file}"
        gridgen += f" --roadnetFile {self.roadnet_file}"
        gridgen += f" --interval {arterial_interval}"
        gridgen += f" --tlPlan"
        print(gridgen)
        os.system(gridgen)
        if update_side_street_interval:
            self._update_side_street_intervals(side_street_interval)

    def _update_side_street_intervals(self, side_street_interval):
        """
        Updates the interval of side street flows.
        """
        with open(self.flow_file, "r") as json_file:
            data = json.load(json_file)
        for flow in data:
            if len(flow.get("route", [])) <= 2:  # side streeets
                flow["interval"] = side_street_interval
        with open(self.flow_file, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def generate_roadnet_files(self, update_side_street_interval=True):
        for mean_flow_rate in self.mean_flow_rates:
            self._generate_roadnet_file(
                mean_flow_rate,
                update_side_street_interval=update_side_street_interval,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Number of columns in the arterial network",
    )
    parser.add_argument(
        "--mean_flow_rates",
        type=int,
        nargs="+",
        default=[500, 600, 700],
        help="Mean flow rates for the arterial and side streets",
    )
    args = parser.parse_args()
    arterial_network = ArterialNetwork(args.ncols, args.mean_flow_rates)
    arterial_network.generate_roadnet_files()

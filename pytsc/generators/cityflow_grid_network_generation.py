import argparse
import os


CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "scenarios",
    "cityflow",
)
CITYFLOW_GRID_GEN_SCRIPT = (
    "/home/rohitbokade/CityFlow/tools/generator/generate_grid_scenario.py"
)


class GridNetwork:
    col_distance = 400
    row_distance = 400
    intersection_width = 20
    n_left_lanes = 1
    n_right_lanes = 1
    n_straight_lanes = 1
    lane_max_speed = 13.89  # m/s
    veh_max_speed = 13.89  # m/s
    veh_min_gap = 2.5  # meters
    veh_headway_time = 2  # meters

    def __init__(self, nrows, ncols, mean_flow_rates=(500, 600, 700)):
        self.nrows = nrows
        self.ncols = ncols
        self.mean_flow_rates = mean_flow_rates

    def _create_scenario_folder(self):
        self.scenario_dir = os.path.join(
            CONFIG_DIR, f"{self.nrows}x{self.ncols}_cityflow_grid"
        )
        os.makedirs(self.scenario_dir, exist_ok=True)

    def _generate_roadnet_file(self, mean_flow_rate):
        self._create_scenario_folder()
        self.roadnet_file = os.path.join(
            self.scenario_dir, f"{self.nrows}x{self.ncols}_roadnet.json"
        )
        self.flow_file = os.path.join(
            self.scenario_dir,
            f"{self.nrows}x{self.ncols}_roadnet_{mean_flow_rate}.json",
        )
        arterial_interval = 3600 / mean_flow_rate
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
        gridgen += " --tlPlan"
        gridgen += " --turn"
        print(gridgen)
        os.system(gridgen)

    def generate_roadnet_files(self):
        for mean_flow_rate in self.mean_flow_rates:
            self._generate_roadnet_file(mean_flow_rate)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nrows",
        type=int,
        default=4,
        help="Number of rows in the arterial network",
    )
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
    arterial_network = GridNetwork(
        args.nrows, args.ncols, args.mean_flow_rates
    )
    arterial_network.generate_roadnet_files()

import argparse
import json
import os


CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../..",
    "scenarios",
    "cityflow",
)


class CityFlowGridNetworkGenerator:
    col_distance = 200
    row_distance = 200
    intersection_width = 20
    n_left_lanes = 0
    n_right_lanes = 0
    n_straight_lanes = 1
    lane_max_speed = 11.11  # m/s
    veh_max_speed = 11.11  # m/s
    veh_min_gap = 2.5  # meters
    veh_headway_time = 2  # meters

    def __init__(self, cityflow_dir, nrows, ncols, mean_flow_rates=[600]):
        self.nrows = nrows
        self.ncols = ncols
        self.mean_flow_rates = mean_flow_rates
        self.cityflow_gridgen_script = os.path.join(
            cityflow_dir,
            "tools/generator/generate_one_way_grid_scenario.py",
        )

    def _create_scenario_folder(self):
        self.scenario_dir = os.path.join(
            CONFIG_DIR, f"syn_{self.nrows}x{self.ncols}"
        )
        os.makedirs(self.scenario_dir, exist_ok=True)

    def _generate_roadnet_file(self, mean_flow_rate):
        self._create_scenario_folder()
        self.roadnet_file = os.path.join(
            self.scenario_dir, f"syn_{self.nrows}x{self.ncols}_roadnet.json"
        )
        self.flow_file = os.path.join(
            self.scenario_dir,
            f"{self.nrows}x{self.ncols}_roadnet_{mean_flow_rate}.json",
        )
        gridgen = f"python {self.cityflow_gridgen_script}"
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
        gridgen += f" --roadnetFile {self.roadnet_file}"
        gridgen += " --tlPlan"
        # gridgen += " --turn"
        print(gridgen)
        os.system(gridgen)

    def generate_roadnet_files(self):
        for mean_flow_rate in self.mean_flow_rates:
            self._generate_roadnet_file(mean_flow_rate)


class CityFlowOneWayGridNetworkGenerator(CityFlowGridNetworkGenerator):
    col_distance = 200
    row_distance = 200
    intersection_width = 20
    n_left_lanes = 0
    n_right_lanes = 0
    n_straight_lanes = 2
    lane_max_speed = 11.11  # m/s
    veh_max_speed = 11.11  # m/s
    veh_min_gap = 2.5  # meters
    veh_headway_time = 2  # meters

    def __init__(self, cityflow_dir, nrows, ncols, mean_flow_rates=[600]):
        super(CityFlowOneWayGridNetworkGenerator, self).__init__(
            cityflow_dir, nrows, ncols, mean_flow_rates
        )

    def _create_scenario_folder(self):
        self.scenario_dir = os.path.join(
            CONFIG_DIR, f"syn_{self.nrows}x{self.ncols}_oneway"
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
        gridgen = f"python {self.cityflow_gridgen_script}"
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
        gridgen += f" --roadnetFile {self.roadnet_file}"
        gridgen += " --tlPlan"
        print(gridgen)
        os.system(gridgen)
        # self._make_grid_one_way()

    def _make_grid_one_way(self):
        with open(self.roadnet_file, "r") as file:
            roadnet = json.load(file)
        for intersection in roadnet["intersections"]:
            intersection["roads"] = filter_roads(intersection["roads"])
            intersection["roadLinks"] = filter_road_links(
                intersection["roadLinks"], intersection["roads"]
            )
        with open(self.roadnet_file, "w") as file:
            json.dump(roadnet, file, indent=4)


def filter_roads(roads):
    filtered_roads = [road for road in roads if "NS" in road or "EW" in road]
    return filtered_roads


def filter_road_links(road_links, filtered_roads):
    filtered_road_links = [
        link
        for link in road_links
        if link["startRoad"] in filtered_roads
        and link["endRoad"] in filtered_roads
    ]
    return filtered_road_links


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-cityflow",
        dest="cityflow_dir",
        type=str,
        default="/home/rohitbokade/CityFlow/",
        help="Path to CityFlow installation",
    )
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
        default=[600],
        help="Mean flow rates for the arterial and side streets",
    )
    args = parser.parse_args()

    network_generator = CityFlowGridNetworkGenerator(
        args.cityflow_dir, args.nrows, args.ncols, args.mean_flow_rates
    )
    network_generator.generate_roadnet_files()

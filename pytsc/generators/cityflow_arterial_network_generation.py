import argparse
import os
import json
import random

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scenarios", "cityflow"
)
CITYFLOW_GRID_GEN_SCRIPT = (
    "/Users/rohitbokade/CityFlow/tools/generator/generate_grid_scenario.py"
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
    end_time = 360

    def __init__(
        self,
        ncols,
        mean_flow_rates=(500, 600, 700),
        n_bursts=1,
        burst_size=15,
        burst_interval=2,
    ):
        self.ncols = ncols
        self.mean_flow_rates = mean_flow_rates
        self.n_bursts = n_bursts
        self.burst_size = burst_size
        self.burst_interval = burst_interval
        self.seed = random.randint(1, 100)
        random.seed(self.seed)

    def _create_scenario_folder(self):
        self.scenario_dir = os.path.join(
            CONFIG_DIR, f"syn_{self.nrows}x{self.ncols}_uniform"
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
        self._add_burst_noise()

    def _update_side_street_intervals(self, side_street_interval):
        """
        Updates the interval of side street flows.
        """
        with open(self.flow_file, "r") as json_file:
            data = json.load(json_file)
        for flow in data:
            if len(flow.get("route", [])) <= 2:  # side streeets
                flow["interval"] = side_street_interval
            flow["endTime"] = self.end_time
        with open(self.flow_file, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def _add_burst_noise(self):
        with open(self.flow_file, "r") as json_file:
            flow_data = json.load(json_file)
        arterial_routes = [
            flow
            for flow in flow_data
            if len(flow["route"]) > 2  # arterial streets
        ]
        occupied_slots = []
        for _ in range(self.n_bursts):
            while True:
                random_arterial_route = random.choice(arterial_routes)
                random_burst_size = random.choice(
                    [self.burst_size - 5, self.burst_size, self.burst_size + 5]
                )
                burst_start_min = 30  # warmup time
                burst_start_max = (
                    self.end_time
                    - burst_start_min
                    - (random_burst_size * self.burst_interval)
                )
                burst_start_time = random.randint(
                    burst_start_min, burst_start_max
                )
                burst_end_time = (
                    burst_start_time + random_burst_size * self.burst_interval
                )
                overlapping = any(
                    start <= burst_end_time and end >= burst_start_time
                    for start, end in occupied_slots
                )
                if not overlapping:
                    occupied_slots.append((burst_start_time, burst_end_time))
                    break
            burst_flow = {
                "vehicle": random_arterial_route["vehicle"],
                "route": random_arterial_route["route"],
                "interval": self.burst_interval,
                "startTime": burst_start_time,
                "endTime": burst_end_time,
            }
            flow_data.append(burst_flow)
        burst_flow_file = self.flow_file.replace(
            ".json",
            f"_burst_{self.n_bursts}_{random_burst_size}_{self.burst_interval}_{self.seed}.json",
        )
        with open(burst_flow_file, "w") as json_file:
            json.dump(flow_data, json_file, indent=4)

    def generate_roadnet_files(self, update_side_street_interval=True):
        if isinstance(self.mean_flow_rates, list):
            for mean_flow_rate in self.mean_flow_rates:
                self._generate_roadnet_file(
                    mean_flow_rate,
                    update_side_street_interval=update_side_street_interval,
                )
        else:
            self._generate_roadnet_file(
                self.mean_flow_rates,
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
        default=(700),
        help="Mean flow rates for the arterial and side streets",
    )
    parser.add_argument(
        "--n_bursts",
        type=int,
        default=2,
        help="Number of bursts to generate",
    )
    parser.add_argument(
        "--burst_size",
        type=int,
        default=15,
        help="# of vehicles in the platoon",
    )
    parser.add_argument(
        "--burst_interval",
        type=int,
        default=2,
        help="1 vehicle per burst_interval second",
    )
    args = parser.parse_args()
    arterial_network = ArterialNetwork(
        args.ncols,
        args.mean_flow_rates,
        args.n_bursts,
        args.burst_size,
        args.burst_interval,
    )
    arterial_network.generate_roadnet_files()

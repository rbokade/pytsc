import json
import os
import random
import tempfile
import time

from itertools import cycle

from pytsc.common.config import BaseConfig
from pytsc.common.utils import EnvLogger

# Set the path to the config.yaml file
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../..",
    "scenarios",
    "cityflow",
)


class Config(BaseConfig):
    def __init__(self, scenario, **kwargs):
        super().__init__(scenario, **kwargs)
        self._load_config("cityflow")
        # Simulator files
        scenario_path = os.path.join(CONFIG_DIR, scenario)
        self.cityflow_roadnet_file = os.path.abspath(
            os.path.join(
                scenario_path,
                f"{self.simulator['roadnet_file']}",
            )
        )
        self.dir = os.path.join(os.path.abspath(scenario_path), "")
        self.temp_dir = tempfile.mkdtemp()
        self.cityflow_cfg_file = None
        self._set_flow_file()
        self._check_assertions()

    def _check_assertions(self):
        assert (
            self.signal["yellow_time"] == self.simulator["delta_time"]
        ), "Delta time and yellow times must be fixed to 5 seconds."

    def create_and_save_cityflow_cfg(self):
        self._set_flow_file()
        # Delete the old config file if it exists
        if self.cityflow_cfg_file and os.path.exists(self.cityflow_cfg_file):
            os.remove(self.cityflow_cfg_file)
        # Create a unique filename with a timestamp suffix
        unique_suffix = str(int(time.time()))
        filename = f"{self.scenario}_cfg_{unique_suffix}.json"
        self.cityflow_cfg_file = os.path.join(self.temp_dir, filename)
        # Configuration details
        cityflow_cfg = {
            "dir": self.dir,
            "roadnetFile": self.simulator["roadnet_file"],
            "flowFile": self.flow_file,
            "interval": self.simulator["interval"],
            "rlTrafficLight": self.simulator["rl_traffic_light"],
            "laneChange": self.simulator["lane_change"],
            "seed": self.simulator["seed"],
            "saveReplay": self.simulator["save_replay"],
            "replayLogFile": self.simulator["replay_log_file"],
            "roadnetLogFile": self.simulator["roadnet_log_file"],
        }
        # Save cityflow_cfg to file
        with open(self.cityflow_cfg_file, "w") as f:
            json.dump(cityflow_cfg, f, indent=4)
        EnvLogger.log_info(f"Loaded flow file: {self.flow_file}")

    def _set_flow_file(self):
        # Get the flow file structure
        self.flow_rate_type = getattr(self, "flow_rate_type", "constant")
        if self.flow_rate_type == "constant":
            self.flow_file = self.simulator["flow_file"]
        elif self.flow_rate_type == "random":
            self.flows = random.choice(self.simulator["flow_files"])
        elif self.flow_rate_type == "sequential":
            self.flows = cycle(self.simulator["flow_files"])
        else:
            raise ValueError(
                "Flow files order is not supported. "
                + "Flow files order must be `random` or `constant`"
            )

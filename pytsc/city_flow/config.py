import os
import json
import logging
import random
import tempfile
import time

from itertools import cycle

from pytsc.common.config import BaseConfig

# Set the path to the config.yaml file
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scenarios", "cityflow"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config(BaseConfig):
    def __init__(self, scenario, add_config):
        super().__init__(scenario, add_config)
        self._update_with_scenario_config("cityflow")
        # Simulator files
        self.cityflow_roadnet_file = os.path.abspath(
            os.path.join(
                CONFIG_DIR, scenario, f"{self.cityflow_config['roadnet_file']}"
            )
        )
        self.dir = os.path.join(
            os.path.abspath(os.path.join(CONFIG_DIR, scenario)), ""
        )
        self.tempdir = self.temp_dir = tempfile.mkdtemp()
        self.cityflow_cfg_file = None
        # Set seeds
        if isinstance(self.cityflow_config["seed"], int):
            self.seed = self.cityflow_config["seed"]
        elif self.cityflow_config["seed"] == "random":
            self.seed = random.randint(0, 1000)
        else:
            raise ValueError(
                "Seed {} is not supported. Seed must be an `int` or `random` (str)".format(
                    self.cityflow_config["seed"]
                )
            )
        # Get the flow file structure
        self.flow_rate_type = getattr(self, "flow_rate_type", "constant")
        if self.flow_rate_type == "constant":
            self.flow_file = self.cityflow_config["flow_file"]
        elif self.flow_rate_type == "random":
            self.flows = self.cityflow_config["flow_files"]
        elif self.flow_rate_type == "sequential":
            self.flows = cycle(self.cityflow_config["flow_files"])
        else:
            raise ValueError(
                "Flow files order is not supported. "
                + "Flow files order must be `random` or `constant`"
            )
        self._check_assertions()

    def _check_assertions(self):
        assert (
            self.signal_config["yellow_time"]
            == self.cityflow_config["delta_time"]
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
            "roadnetFile": self.cityflow_config["roadnet_file"],
            "flowFile": self.flow_file,
            "interval": self.cityflow_config["interval"],
            "rlTrafficLight": self.cityflow_config["rl_traffic_light"],
            "laneChange": self.cityflow_config["lane_change"],
            "seed": self.seed,
            "saveReplay": self.cityflow_config["save_replay"],
            "replayLogFile": self.cityflow_config["replay_log_file"],
            "roadnetLogFile": self.cityflow_config["roadnet_log_file"],
        }
        # Save cityflow_cfg to file
        with open(self.cityflow_cfg_file, "w") as f:
            json.dump(cityflow_cfg, f, indent=4)
        logger.info(f"Loaded flow file: {self.flow_file}")

    def _set_flow_file(self):
        if self.flow_rate_type == "constant":
            self.flow_file = self.cityflow_config["flow_file"]
        elif self.flow_rate_type == "random":
            self.flow_file = random.choice(self.flows)
        elif self.flow_rate_type == "sequential":
            self.flow_file = next(self.flows)
        else:
            raise ValueError(
                "Flow files order is not supported. "
                + "Flow files order must be `random` or `constant`"
            )

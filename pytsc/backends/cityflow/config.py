import dis
import json
import logging
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

# EnvLogger.set_log_level(logging.WARNING)


class Config(BaseConfig):
    def __init__(self, scenario, **kwargs):
        super().__init__(scenario, **kwargs)
        self._load_config("cityflow")
        # Simulator files
        scenario_path = os.path.join(CONFIG_DIR, scenario)
        self._set_roadnet_file(scenario_path, **kwargs)
        self.dir = os.path.join(os.path.abspath(scenario_path), "")
        self.temp_dir = tempfile.mkdtemp()
        self.cityflow_cfg_file = None
        self.flow_files_cycle = cycle(self.simulator.get("flow_files", []))
        # self._set_flow_file()
        self._check_assertions()
        # random.seed(self.simulator["seed"])

    def _set_roadnet_file(self, scenario_path, **kwargs):
        self.cityflow_roadnet_file = os.path.abspath(
            os.path.join(
                scenario_path,
                f"{self.simulator['roadnet_file']}",
            )
        )

    def _check_assertions(self):
        assert (
            self.signal["yellow_time"] == self.simulator["delta_time"]
        ), "Delta time and yellow times must be fixed to 5 seconds."

    def _set_flow_file(self):
        # Get the flow file structure
        self.flow_rate_type = self.simulator.get("flow_rate_type", "constant")
        if self.flow_rate_type == "constant":
            self.flow_file = self.simulator["flow_file"]
        elif self.flow_rate_type == "random":
            self.flow_file = random.choice(self.simulator["flow_files"])
        elif self.flow_rate_type == "sequential":
            self.flow_file = next(self.flow_files_cycle)
        else:
            raise ValueError(
                "Flow files order is not supported. "
                + "Flow files order must be `random` or `constant`"
            )

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


class DisruptedConfig(Config):
    def __init__(self, scenario, mode="train", **kwargs):
        self.scenario = scenario
        self.mode = mode
        self._additional_config = kwargs
        self._load_config("cityflow")
        # self.mode = kwargs.get("mode", "train")
        # self.domain = kwargs.get("domain", "flow_disrupted")
        # Simulator files
        scenario_path = os.path.join(CONFIG_DIR, scenario)
        self._set_roadnet_file(scenario_path, **kwargs)
        self.dir = os.path.join(os.path.abspath(scenario_path), "")
        self.temp_dir = tempfile.mkdtemp()
        self.cityflow_cfg_file = None
        # e.g., ['flow_disrupted', 'link_disrupted']
        self.domains = list(self.simulator[mode].keys())
        # e.g., { 'flow_disrupted': ['600', ...], 'link_disrupted': ['0_1', ...] }
        self.disrup_values = {
            domain: list(self.simulator[mode][domain].keys()) for domain in self.domains
        }
        self.domain_classes = self._get_domain_classes()
        self._check_assertions()
        self.current_domain_class = None
        # random.seed(self.simulator["seed"])

    def _get_domain_classes(self):
        combined_labels = []
        for domain in self.domains:
            for disrup_value in self.disrup_values[domain]:
                combined_labels.append((domain, disrup_value))
        return combined_labels

    def _set_flow_file(self):
        self.flow_rate_type = self.simulator.get("flow_rate_type", "constant")
        self._select_random_flow_file()

    def _select_random_flow_file(self):
        random_domain = random.choice(self.domains)
        random_disrup_value = random.choice(self.disrup_values[random_domain])
        self.current_domain_class = self.domain_classes.index(
            (random_domain, random_disrup_value)
        )
        flow_file = random.choice(
            self.simulator[self.mode][random_domain][random_disrup_value]
        )
        self.flow_file = os.path.join(
            self.mode, random_domain, random_disrup_value, flow_file
        )
        print(
            f"Randomly selected flow file: {self.flow_file} from"
            + f"domain {random_domain}, disruption {random_disrup_value}"
        )

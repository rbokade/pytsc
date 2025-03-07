import logging
import os
import pprint
import random
from abc import ABC

import yaml

from pytsc.common.utils import EnvLogger, recursively_update_dict

# Set the path to the config.yaml file
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "scenarios",
)

# EnvLogger.set_log_level(logging.WARNING)


class BaseConfig(ABC):
    def __init__(self, scenario, **kwargs):
        self.scenario = scenario
        self._additional_config = kwargs
        random.seed(self.simulator["seed"])

    def _load_config(self, simulator_backend):
        # Load default config parameters
        default_file_path = os.path.join(CONFIG_DIR, "default", "config.yaml")
        with open(default_file_path, "r") as f:
            default_config = yaml.safe_load(f)

        # Load scenario-specific config parameters
        scenario_file_path = os.path.join(
            CONFIG_DIR, simulator_backend, self.scenario, "config.yaml"
        )
        EnvLogger.log_info(f"Scenario config file: {scenario_file_path}")
        if os.path.exists(scenario_file_path):
            with open(scenario_file_path, "r") as f:
                scenario_config = yaml.safe_load(f)
        recursively_update_dict(default_config, scenario_config)

        # Update the config with input config parameters
        if self._additional_config is not None:
            recursively_update_dict(default_config, self._additional_config)

        # Add the final configs to the class
        self.network = default_config["network"]
        self.signal = default_config["signal"]
        self.misc = default_config["misc"]
        self.simulator = default_config[simulator_backend]

        formatted_config = pprint.pformat(default_config, indent=4)
        EnvLogger.log_info(f"Config:\n{formatted_config}")

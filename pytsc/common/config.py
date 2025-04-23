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
    """
    Base class for configuration management in traffic signal control.
    This class handles loading and merging configuration files for different
    scenarios and simulator backends.
    Args:
        scenario (str): Name of the scenario for which the configuration is being loaded.
        debug (bool): Flag to enable debug mode. If True, detailed logs will be printed.
        **kwargs: Additional configuration parameters to override default values.
    """
    def __init__(self, scenario, debug=False, **kwargs):
        self.debug = debug
        self.scenario = scenario
        self._additional_config = kwargs

    def _load_config(self, simulator_backend):
        """
        Load the configuration files for the specified scenario and simulator backend.
        This method loads the default configuration, merges it with scenario-specific
        configuration, and applies any additional configuration parameters provided.
        Args:
            simulator_backend (str): The simulator backend to be used (e.g., "cityflow", "sumo").
        """
        # Load default config parameters
        default_file_path = os.path.join(CONFIG_DIR, "default", "config.yaml")
        with open(default_file_path, "r") as f:
            default_config = yaml.safe_load(f)

        # Load scenario-specific config parameters
        scenario_file_path = os.path.join(
            CONFIG_DIR, simulator_backend, self.scenario, "config.yaml"
        )
        if not self.debug:
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

        random.seed(self.simulator["seed"])

        if not self.debug:
            formatted_config = pprint.pformat(default_config, indent=4)
            EnvLogger.log_info(f"Config:\n{formatted_config}")

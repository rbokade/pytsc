import os
import yaml

from abc import ABC

from pytsc.common.utils import recursively_update_dict

# Set the path to the config.yaml file
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "scenarios",
)


class BaseConfig(ABC):
    def __init__(self, scenario, add_config):
        self.scenario = scenario
        self.add_config = add_config
        self._get_default_config()

    def _get_default_config(self):
        # Load default config parameters
        default_file_path = os.path.join(CONFIG_DIR, "default", "config.yaml")
        with open(default_file_path, "r") as f:
            default_config = yaml.safe_load(f)
        self._config = default_config

    def _update_with_scenario_config(self, simulator_type):
        # Load scenario-specific config parameters
        scenario_file_path = os.path.join(
            CONFIG_DIR, simulator_type, self.scenario, "config.yaml"
        )
        if os.path.exists(scenario_file_path):
            with open(scenario_file_path, "r") as f:
                scenario_config = yaml.safe_load(f)
        # Update the default config with scenario-specific parameters
        recursively_update_dict(self._config, scenario_config)
        # Update the config with input config parameters
        recursively_update_dict(self._config, self.add_config)
        # Set the config as attributes
        self._set_config()

    def _set_config(self):
        for key, value in self._config.items():
            setattr(self, f"{key}_config", value)

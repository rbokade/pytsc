import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from pytsc.common.config import BaseConfig

# Set the path to the config.yaml file
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scenarios", "sumo"
)


class Config(BaseConfig):
    def __init__(self, scenario, add_config):
        super().__init__(scenario, add_config)
        self._update_with_scenario_config("sumo")
        # Simulator files
        self.sumo_cfg_dir = os.path.join(
            CONFIG_DIR, scenario, f"{scenario}.sumocfg"
        )
        self.net_dir = os.path.join(
            CONFIG_DIR, scenario, f"{scenario}.net.xml"
        )
        self._check_assertions()

    def _check_assertions(self):
        assert (
            self.signal_config["yellow_time"] == self.sumo_config["delta_time"]
        ), "Delta time and yellow times must be fixed to 5 seconds."

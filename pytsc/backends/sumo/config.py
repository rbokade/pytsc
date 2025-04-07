import os
import random
import sys
import xml.etree.ElementTree as ET

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from pytsc.common.config import BaseConfig

# Set the path to the config.yaml file
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../..", "scenarios", "sumo"
)


class Config(BaseConfig):
    def __init__(self, scenario, **kwargs):
        super().__init__(scenario, **kwargs)
        self._load_config("sumo")
        # if not self.simulator["random_game"]:
        #     self.sumo_cfg_dir = os.path.join(
        #         CONFIG_DIR, scenario, self.simulator["sumo_config_file"]
        #     )
        # else:
        #     self.sumo_cfg_dirs = [
        #         os.path.join(CONFIG_DIR, scenario, f)
        #         for f in self.simulator["sumo_config_files"]
        #     ]
        #     self.reset_config()
        if kwargs["sumo"].get("sumo_config_file", None) is not None:
            self.simulator["sumo_config_file"] = kwargs["sumo"]["sumo_config_file"]
        self.sumo_cfg_dir = os.path.join(
            CONFIG_DIR, scenario, self.simulator["sumo_config_file"]
        )
        self.net_dir = os.path.join(
            CONFIG_DIR, scenario, self.simulator["sumo_net_file"]
        )
        self._check_assertions()
        self._get_start_and_end_times()

    def reset_config(self, **kwargs):
        if self.simulator["random_game"]:
            self.sumo_cfg_dir = random.choice(self.sumo_cfg_dirs)

    def _check_assertions(self):
        assert (
            self.signal["yellow_time"] == self.simulator["delta_time"]
        ), "Delta time and yellow times must be fixed to 5 seconds."

    def _get_start_and_end_times(self):
        # Load the XML file
        tree = ET.parse(self.sumo_cfg_dir)
        root = tree.getroot()
        # Access the 'time' element and extract 'begin' and 'end' values
        time_element = root.find("time")
        self.begin_time = int(time_element.find("begin").get("value"))
        self.end_time = int(time_element.find("end").get("value"))

    def _set_netdir(self, netfile):
        self.net_dir = os.path.join(CONFIG_DIR, self.scenario, netfile)

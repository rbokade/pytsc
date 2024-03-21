import os
import sys
import argparse
from lxml import etree

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from pytsc.backends.sumo.config import Config


CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "scenarios",
)


class LaneAreaDetectorGenerator:
    def __init__(self, scenario):
        self.scenario = scenario
        self.config = Config(scenario, additional_config=None)
        self.data_dir = os.path.join(CONFIG_DIR, "sumo", scenario)
        self.netfile_dir = os.path.join(
            self.data_dir, f"{self.scenario}.net.xml"
        )

    def generate_lane_area_detectors(self):
        detector_file_dir = os.path.join(
            self.data_dir, f"{self.scenario}.add.xml"
        )
        script = tools + "/output/generateTLSE2Detectors.py"
        ladgen = "python {}".format(script)
        ladgen += " --net-file {}".format(self.netfile_dir)
        ladgen += " --output {}".format(detector_file_dir)
        ladgen += " --detector-length {}".format(
            self.config.signal["visibility"]
        )
        ladgen += " --frequency 1"
        print(ladgen)
        os.system(ladgen)
        self._add_to_config_file()
        self._disable_detector_logs()

    def _add_to_config_file(self):
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(self.config.sumo_cfg_dir, parser)
        root = tree.getroot()
        input_tag = root.find("input")
        additional_files_element = etree.Element("additional-files")
        additional_files_element.set("value", f"{self.scenario}.add.xml")
        input_tag.append(additional_files_element)
        tree.write(
            self.config.sumo_cfg_dir,
            pretty_print=True,
            encoding="utf-8",
            xml_declaration=True,
        )

    def _disable_detector_logs(self):
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(self.config.sumo_cfg_dir, parser)
        root = tree.getroot()
        existing_report = root.find("report")
        if existing_report is not None:
            root.remove(existing_report)
        report = etree.SubElement(root, "report")
        verbose = etree.SubElement(report, "verbose")
        verbose.set("value", "true")
        no_step_log = etree.SubElement(report, "no-step-log")
        no_step_log.set("value", "true")
        tree.write(
            self.config.sumo_cfg_dir,
            pretty_print=True,
            xml_declaration=True,
            encoding="UTF-8",
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        help="Name of SUMO scenario",
        type=str,
        default="2x2_sumo_grid",
    )
    args = parser.parse_args()
    netgenerator = LaneAreaDetectorGenerator(args.scenario)
    netgenerator.generate_lane_area_detectors()

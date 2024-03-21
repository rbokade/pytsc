import argparse
import os
import random
import xml.etree.ElementTree as ET

from pytsc.backends.sumo.config import Config
from pytsc.backends.sumo.network_parser import NetworkParser

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "scenarios",
)


class TripGenerator:
    def __init__(self, scenario):
        self.scenario = scenario
        self.config = Config(scenario)
        self.parsed_network = NetworkParser(self.config)
        self.data_dir = os.path.join(CONFIG_DIR, scenario)
        self.netfile_dir = os.path.join(
            self.data_dir, f"{self.scenario}.net.xml"
        )
        self.routesfile_dir = os.path.join(
            self.data_dir, f"{self.scenario}.rou.xml"
        )
        self.odpairsfile_dir = os.path.join(
            self.data_dir, f"{self.scenario}_odpairs.txt"
        )
        self.begin = 0
        self.end = 3600

    def find_fringe_edges(self):
        incoming_fringe_edges = []
        outgoing_fringe_edges = []
        for edge in self.parsed_network.net.getEdges():
            if edge.is_fringe():
                if edge.getFromNode().getFringe() is not None:
                    outgoing_fringe_edges.append(edge)
                elif edge.getToNode().getFringe() is not None:
                    incoming_fringe_edges.append(edge)
        return incoming_fringe_edges, outgoing_fringe_edges

    def _create_od_pairs(self):
        incoming_fringe_edges, outgoing_fringe_edges = self.find_fringe_edges()
        # Calculate all combinations of incoming and outgoing edges
        od_pairs = [
            (in_edge.getID(), out_edge.getID())
            for in_edge in incoming_fringe_edges
            for out_edge in outgoing_fringe_edges
        ]
        # Write the origin-destination pairs to the file
        with open(self.odpairsfile_dir, "w") as f:
            for in_edge_id, out_edge_id in od_pairs:
                f.write(f"{in_edge_id} {out_edge_id}\n")

    def generate_flows(self):
        # self._create_od_pairs()
        # # Use od2trips to generate the trips directly in the routes file
        # cmd = "od2trips"
        # cmd += f" --net-file {self.netfile_dir}"
        # cmd += f" --flow-output {self.routesfile_dir}"
        # cmd += f" --begin {self.begin}"
        # cmd += f" --end {self.end}"
        # cmd += f" --od-matrix-files {self.odpairsfile_dir}"
        # cmd += " --verbose"
        # os.system(cmd)
        self._validate_flows()

    # def generate_flows(self):
    #     incoming_fringe_edges, outgoing_fringe_edges = self.find_fringe_edges()
    #     root = ET.Element("routes")
    #     for i in range(self.config.trip_generator_config["n_flows"]):
    #         flow = ET.SubElement(root, "flow")
    #         flow.set("id", f"flow_{i}")
    #         flow.set("from", random.choice(incoming_fringe_edges).getID())
    #         flow.set("to", random.choice(outgoing_fringe_edges).getID())
    #         flow.set("begin", str(self.begin))
    #         flow.set("end", str(self.end))
    #         flow.set("probability", str(random.uniform(0.1, 0.5)))
    #         flow.set("departSpeed", "max")
    #         flow.set("departPos", "base")
    #         flow.set("departLane", "best")
    #     # Save the generated routes to the file
    #     tree = ET.ElementTree(root)
    #     tree.write(self.routesfile_dir, xml_declaration=True, encoding="utf-8")
    #     # Print the generated XML content
    #     ET.dump(tree)
    #     self._validate_flows()

    def _validate_flows(self):
        # Run duarouter to validate and fix the routes in-place
        validate = "duarouter --repair "
        validate += f"--net-file {self.netfile_dir} "
        validate += f"--route-files {self.routesfile_dir} "
        validate += f"--output-file {self.routesfile_dir} "
        validate += f"--begin {self.begin} "
        validate += f"--end {self.end} "
        validate += "--remove-loops "
        validate += "--repair.from "
        validate += "--repair.to "
        print(validate)
        os.system(validate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        help="scenario name",
        type=str,
        default="4x4_regular_grid",
    )
    args = parser.parse_args()

    trip_generator = TripGenerator(args.scenario)
    trip_generator.generate_flows()

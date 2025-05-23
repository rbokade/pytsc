import math
import os
import sys
from functools import lru_cache

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    from sumolib.net import readNet
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np

from pytsc.common.network_parser import BaseNetworkParser
from pytsc.common.utils import flatten_list, sort_alphanumeric_ids


class NetworkParser(BaseNetworkParser):
    """
    Network parser for SUMO simulator.

    Args:
        config (Config): Configuration object containing simulation parameters.
    """
    def __init__(self, config):
        super().__init__(config)
        self._load_network()
        self._initialize_traffic_signals()

    def _load_network(self):
        """
        Load the SUMO network using the provided configuration.
        """
        self.net = readNet(self.config.net_dir, withPrograms=True)

    def _initialize_traffic_signals(self):
        """ 
        Initialize traffic signals in the network.
        """
        (
            inc_lane_map,
            out_lane_map,
            inc_to_out_lane_map,
        ) = self._get_lane_mappings()
        self.ts_to_nodes, self.nodes_to_ts = self._map_ts_id_and_node_id()
        (
            phases,
            phases_min_max_times,
            phase_indices,
            green_phase_indices,
            yellow_phase_indices,
        ) = self._get_traffic_light_phases()
        self.traffic_signals = {}
        for ts in self.net.getTrafficLights():
            ts_id = ts.getID()
            self.traffic_signals[ts_id] = {
                "coordinates": self.ts_coordinates[ts_id],
                "norm_coordinates": self.ts_norm_coordinates[ts_id],
                "incoming_lanes": inc_lane_map[ts_id],
                "outgoing_lanes": out_lane_map[ts_id],
                "inc_to_out_lanes": inc_to_out_lane_map[ts_id],
                "phase_to_inc_out_lanes": self.ts_phase_to_inc_out_lanes[ts_id],
                "phases": phase_indices[ts_id],
                "n_phases": len(phase_indices[ts_id]),
                "phases_min_max_times": phases_min_max_times[ts_id],
                "phase_indices": phase_indices[ts_id],
                "green_phase_indices": green_phase_indices[ts_id],
                "yellow_phase_indices": yellow_phase_indices[ts_id],
            }
            self.traffic_signals[ts_id].update(self.config.signal)

    @property
    @lru_cache(maxsize=None)
    def adjacency_matrix(self):
        """
        The adjacency matrix is a square matrix used to represent a finite graph.

        Returns:
            np.ndarray: The adjacency matrix representing the connections between traffic signals.
        """
        if "neighbors" not in self.config.network.keys():
            nodes_with_tl = [
                node
                for node in self.net.getNodes()
                if node.getType() == "traffic_light"
            ]
            n_traffic_signals = len(nodes_with_tl)
            node_to_index = {node: idx for idx, node in enumerate(nodes_with_tl)}
            adjacency_matrix = np.zeros((n_traffic_signals, n_traffic_signals))
            for node in nodes_with_tl:
                i = node_to_index[node]
                for edge in node.getOutgoing():
                    dest_node = edge.getToNode()
                    if dest_node in nodes_with_tl:
                        j = node_to_index[dest_node]
                        adjacency_matrix[i, j] = 1
            return adjacency_matrix
        else:
            return super(NetworkParser, self)._get_adjacency_matrix()

    @property
    @lru_cache(maxsize=None)
    def network_boundary(self):
        """
        The network boundary is the minimum and maximum coordinates of the network.

        Returns:
            tuple: A tuple containing the minimum and maximum coordinates of the network.
        """
        try:
            x_min, y_min, x_max, y_max = self.net.getBoundary()
        except Exception:
            [x_min, y_min], [x_max, y_max] = self.net.getBBoxXY()
        return (x_min, y_min), (x_max, y_max)

    @property
    @lru_cache(maxsize=None)
    def norm_network_boundary(self):
        """
        The normalized network boundary is the difference between the maximum and minimum coordinates of the network.

        Returns:
            list: A list containing the normalized width and height of the network.
        """
        (x_min, y_min), (x_max, y_max) = self.network_boundary
        return [x_max - x_min, y_max - y_min]

    @property
    @lru_cache(maxsize=None)
    def traffic_signal_ids(self):
        """
        The traffic signal IDs are the unique identifiers for each traffic signal in the network.

        Returns:
            list: A list of traffic signal IDs.
        """
        traffic_signals = self.net.getTrafficLights()
        ts_ids = [traffic_signals[i].getID() for i in range(len(traffic_signals))]
        return sort_alphanumeric_ids(ts_ids)

    @property
    @lru_cache(maxsize=None)
    def lanes(self):
        """
        The lanes are the unique identifiers for each lane in the network.

        Returns:
            list: A list of lane IDs.
        """
        lanes = [edge.getLanes() for edge in self.net.getEdges()]
        lanes = list(set(flatten_list(lanes)))
        lane_ids = [lane.getID() for lane in lanes]
        return sort_alphanumeric_ids(lane_ids)

    @property
    @lru_cache(maxsize=None)
    def lane_lengths(self):
        """
        The lane lengths are the lengths of each lane in the network.

        Returns:
            dict: A dictionary mapping lane IDs to their lengths.
        """
        lane_lengths = {}
        for edge in self.net.getEdges():
            lanes = edge.getLanes()
            for lane in lanes:
                lane_id = lane.getID()
                lane_lengths[lane_id] = lane.getLength()
        return lane_lengths

    @property
    @lru_cache(maxsize=None)
    def lane_max_speeds(self):
        """
        The lane maximum speeds are the maximum speeds of each lane in the network.

        Returns:
            dict: A dictionary mapping lane IDs to their maximum speeds.
        """
        lane_max_speeds = {}
        for edge in self.net.getEdges():
            for i, lane in enumerate(edge.getLanes()):
                lane_id = lane.getID()
                lane_max_speeds[lane_id] = lane.getSpeed()
        return lane_max_speeds

    @property
    @lru_cache(maxsize=None)
    def lane_indices(self):
        """ 
        The lane indices are the indices of each lane in the network.

        Returns:
            dict: A dictionary mapping lane IDs to their indices.
        """
        lane_indices = {}
        for edge in self.net.getEdges():
            lanes = edge.getLanes()
            for i, lane in enumerate(lanes):
                lane_id = lane.getID()
                lane_indices[lane_id] = i
        return lane_indices

    @property
    @lru_cache(maxsize=None)
    def lane_angles(self):
        """
        The lane angles are the angles of each lane in the network.

        Returns:
            dict: A dictionary mapping lane IDs to their angles in degrees.
        """
        lane_angles = {}
        for edge in self.net.getEdges():
            lanes = edge.getLanes()
            for lane in lanes:
                lane_id = lane.getID()
                shape = lane.getShape()
                if len(shape) >= 2:
                    (x1, y1) = shape[0]
                    (x2, y2) = shape[-1]
                    angle_rad = math.atan2(y2 - y1, x2 - x1)
                    angle_deg = math.degrees(angle_rad)
                    lane_angles[lane_id] = angle_deg
                else:
                    # If shape is not available
                    lane_angles[lane_id] = 0.0
        return lane_angles

    @property
    @lru_cache(maxsize=None)
    def ts_phase_to_inc_out_lanes(self):
        """
        Returns a dictionary of incoming and outgoing lanes for
        each phase for each traffic signal.

        Returns:
            dict: A dictionary mapping traffic signal IDs to their phases and the corresponding incoming and outgoing lanes.
        """
        ts_phase_to_inc_out_lanes = {}
        for ts in self.net.getTrafficLights():
            ts_id = ts.getID()
            ts_phase_to_inc_out_lanes[ts_id] = {}
            controlled_links = ts.getLinks()
            connection_indices = {i: [] for i in range(len(ts.getConnections()))}
            for link_no, links in controlled_links.items():
                connection_indices[link_no] = [conn[1].getID() for conn in links]
            for program in ts.getPrograms().values():
                for phase_idx, phase in enumerate(program.getPhases()):
                    phase_inc_out_lanes = []
                    for i, signal_state in enumerate(phase.state):
                        if signal_state.lower() in ("g", "g1", "y", "y1"):
                            out_lanes = connection_indices[i]
                            for out_lane in out_lanes:
                                inc_lanes = self.net.getLane(out_lane).getIncoming()
                                for inc_lane in inc_lanes:
                                    phase_inc_out_lanes.append(
                                        (inc_lane.getID(), out_lane)
                                    )
                    # ts_phase_to_inc_out_lanes[ts_id][phase_idx] = list(
                    #     set(phase_inc_out_lanes)
                    # )
                    ts_phase_to_inc_out_lanes[ts_id][phase_idx] = {}
                    for inc_lane, out_lane in phase_inc_out_lanes:
                        if (
                            inc_lane
                            not in ts_phase_to_inc_out_lanes[ts_id][phase_idx].keys()
                        ):
                            ts_phase_to_inc_out_lanes[ts_id][phase_idx][inc_lane] = []
                        ts_phase_to_inc_out_lanes[ts_id][phase_idx][inc_lane].append(
                            out_lane
                        )
        return ts_phase_to_inc_out_lanes

    @property
    @lru_cache(maxsize=None)
    def k_hop_neighbors(self):
        """
        Returns a dictionary of k-hop neighbors for each traffic signal.

        Returns:
            dict: A dictionary mapping traffic signal IDs to their k-hop neighbors.
        """
        k_hop_neighbors = {}
        max_hops = self.config.misc["max_hops"]
        for ts_id in self.traffic_signal_ids:
            k_hop_neighbors[ts_id] = {}
            for k in range(1, max_hops + 1):
                k_hop_neighbors[ts_id][k] = self._get_k_hop_neighbors_for_ts(ts_id, k)
        return k_hop_neighbors

    @property
    @lru_cache(maxsize=None)
    def ts_coordinates(self):
        """
        Returns the coordinates of the traffic signals in the network.

        Returns:
            dict: A dictionary mapping traffic signal IDs to their coordinates.
        """
        ts_coordinates = {}
        nodes = self.net.getNodes()
        node_coords = []
        for ts in self.net.getTrafficLights():
            ts_id = ts.getID()
            ts_nodes = self.ts_to_nodes[ts_id]
            for node_id in ts_nodes:
                for node in nodes:
                    if node.getID() == node_id:
                        node_coords.append(node.getCoord())
            ts_coordinates[ts_id] = np.mean(node_coords, axis=0)
        return ts_coordinates

    @property
    @lru_cache(maxsize=None)
    def ts_norm_coordinates(self):
        """
        Returns the normalized coordinates of the traffic signals in the network.

        Returns:
            dict: A dictionary mapping traffic signal IDs to their normalized coordinates.
        """
        ts_norm_coordinates = {}
        for ts_id, ts_coord in self.ts_coordinates.items():
            ts_norm_coordinates[ts_id] = [
                ts_coord[0] / self.norm_network_boundary[0],
                ts_coord[1] / self.norm_network_boundary[1],
            ]
        return ts_norm_coordinates

    def _get_k_hop_neighbors_for_ts(self, ts_id, k):
        """
        K-hop neighbors for a given traffic signal.

        Args:
            ts_id (str): The ID of the traffic signal.
            k (int): The number of hops.
        Returns:
            k_hop_neighbors_ids (list): A list of
        """
        if self.adjacency_matrix.shape[0] == 1:
            return []
        adjacency_matrix_power = np.linalg.matrix_power(self.adjacency_matrix, k)
        ts_index = self.traffic_signal_ids.index(ts_id)
        k_hop_neighbors_indices = np.where(adjacency_matrix_power[ts_index] > 0)[0]
        k_hop_neighbors_ids = [
            self.traffic_signal_ids[index] for index in k_hop_neighbors_indices
        ]
        return k_hop_neighbors_ids

    def _get_lane_mappings(self):
        """
        Returns a dictionary of incoming and outgoing lanes for
        each traffic signal.

        Returns:
            tuple: A tuple containing three dictionaries:
        """
        incoming_lane_map, outgoing_lane_map, inc_to_out_lane_map = {}, {}, {}
        for ts in self.net.getTrafficLights():
            pairs_list = sorted(ts.getConnections(), key=lambda x: x[2])
            ts_inc_to_out_lane_map = {}
            for pair in pairs_list:
                inc_lane = pair[0].getID()
                if inc_lane not in ts_inc_to_out_lane_map.keys():
                    ts_inc_to_out_lane_map[inc_lane] = []
                ts_inc_to_out_lane_map[pair[0].getID()].append(pair[1].getID())
            incoming_lane_map[ts._id] = list(ts_inc_to_out_lane_map.keys())
            outgoing_lane_map[ts._id] = list(
                set(flatten_list(ts_inc_to_out_lane_map.values()))
            )
            inc_to_out_lane_map[ts._id] = ts_inc_to_out_lane_map
        return incoming_lane_map, outgoing_lane_map, inc_to_out_lane_map

    def _map_ts_id_and_node_id(self):
        """
        Maps traffic signal IDs to node IDs and vice versa.

        Returns:
            tuple: A tuple containing two dictionaries:
                - ts_to_nodes: Mapping of traffic signal IDs to node IDs.
                - nodes_to_ts: Mapping of node IDs to traffic signal IDs.
        """
        ts_to_nodes = {}
        nodes_to_ts = {}
        # Map ts to nodes
        for ts in self.net.getTrafficLights():
            incoming_edges = ts.getEdges()
            # A traffic light can have multiple nodes
            # but a node can only have one traffic light
            ts_nodes = set(  # only retain unique nodes
                [ie.getToNode().getID() for ie in incoming_edges]
            )
            ts_to_nodes[ts.getID()] = list(ts_nodes)
        # Map nodes to ts
        for ts, nodes in ts_to_nodes.items():
            for node in nodes:
                nodes_to_ts[node] = ts
        return ts_to_nodes, nodes_to_ts

    def _get_traffic_light_phases(self):
        """
        Returns a dictionary of phases for each traffic signal.

        Returns:
            tuple: A tuple containing:
                - phases: A dictionary mapping traffic signal IDs to their phases.
                - phases_min_max_times: A dictionary mapping traffic signal IDs to their minimum and maximum times for each phase.
                - phase_indices: A dictionary mapping traffic signal IDs to their phase indices.
                - green_phase_indices: A dictionary mapping traffic signal IDs to their green phase indices.
                - yellow_phase_indices: A dictionary mapping traffic signal IDs to their yellow phase indices.  
        """
        phases = {}
        phase_indices = {}
        green_phase_indices = {}
        yellow_phase_indices = {}
        phases_min_max_times = {}
        for ts in self.net.getTrafficLights():
            ts_id = ts.getID()
            # phases[ts_id] = program.getPhases()
            phase_indices[ts_id] = []
            green_phase_indices[ts_id] = []
            yellow_phase_indices[ts_id] = []
            phases_min_max_times[ts_id] = {}
            for program in ts.getPrograms().values():
                for phase_idx, phase in enumerate(program.getPhases()):
                    phase_indices[ts_id].append(phase_idx)
                    if "G" in phase.state:  # green phase
                        green_phase_indices[ts_id].append(phase_idx)
                        min_time = self.config.signal["min_green_time"]
                        max_time = self.config.signal["max_green_time"]
                    elif "y" in phase.state:  # yellow phase
                        yellow_phase_indices[ts_id].append(phase_idx)
                        min_time = self.config.signal["yellow_time"]
                        max_time = self.config.signal["yellow_time"]
                    else:  # all red phase (remove for now)
                        breakpoint()
                    phases_min_max_times[ts_id][phase_idx] = {
                        "min_time": min_time,
                        "max_time": max_time,
                    }
        return (
            phases,
            phases_min_max_times,
            phase_indices,
            green_phase_indices,
            yellow_phase_indices,
        )

    def plot_network(self, figsize=(12, 12)):
        """
        Plots the SUMO network with intersections and traffic signals.

        Args:
            figsize (tuple): Size of the figure for plotting.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()

        # Get all nodes (intersections)
        for node in self.net.getNodes():
            node_id = node.getID()
            coord = node.getCoord()
            G.add_node(node_id, pos=coord, color="gray")

        # Get all edges (roads connecting intersections)
        for edge in self.net.getEdges():
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            G.add_edge(from_node, to_node)

        # Mark traffic signals
        ts_nodes = {
            ts_id: self.ts_coordinates[ts_id] for ts_id in self.traffic_signal_ids
        }

        # Plot network
        pos = nx.get_node_attributes(G, "pos")
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx(
            G,
            pos,
            node_size=20,
            with_labels=False,
            node_color="gray",
            edge_color="black",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=ts_nodes.keys(), node_color="red", node_size=50
        )

        plt.title("SUMO Road Network")
        plt.show()

    # def _map_ts_phase_to_outgoing_lanes(self):
    #     """
    #     Returns a dictionary of outgoing lanes for
    #     each phase for each traffic signal.
    #     NOTE: Needed for SOTL
    #     """
    #     ts_phase_outlanes = {}
    #     for ts in self.net.getTrafficLights():
    #         ts_id = ts.getID()
    #         ts_phase_outlanes[ts_id] = {}
    #         # Get the controlled links and their associated connection index
    #         # for the traffic signal
    #         controlled_links = ts.getLinks()
    #         connection_indices = {
    #             i: [] for i in range(len(ts.getConnections()))
    #         }
    #         for link_no, links in controlled_links.items():
    #             connection_indices[link_no] = [
    #                 conn[1].getID() for conn in links
    #             ]
    #         for program in ts.getPrograms().values():
    #             for phase_idx, phase in enumerate(program.getPhases()):
    #                 phase_outlanes = []
    #                 # Iterate through the signal state of the phase
    #                 for i, signal_state in enumerate(phase.state):
    #                     if signal_state.lower() in ("g", "g1", "y", "y1"):
    #                         phase_outlanes.extend(connection_indices[i])
    #                 ts_phase_outlanes[ts_id][phase_idx] = list(
    #                     set(phase_outlanes)
    #                 )
    #     return ts_phase_outlanes

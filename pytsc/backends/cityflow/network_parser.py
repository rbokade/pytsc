import json
import math
from functools import lru_cache

import networkx as nx
import numpy as np

from pytsc.common.network_parser import BaseNetworkParser


class NetworkParser(BaseNetworkParser):
    """
    Parses the network file and extracts essential information
    and stores it in the config.

    Args:
        config (Config): Configuration object containing the network file path.
    """

    def __init__(self, config):
        super().__init__(config)
        self._load_network()

    def _load_network(self):
        """
        Load the network file and extract essential information.
        """
        with open(self.config.cityflow_roadnet_file, "r") as f:
            self.net = json.load(f)
        self._initialize_traffic_signals()

    def _initialize_traffic_signals(self):
        """
        Initialize traffic signals and their properties.
        """
        (
            inc_lane_map,
            out_lane_map,
            inc_to_out_lane_map,
        ) = self._get_lane_mappings()
        (
            phases,
            phases_min_max_times,
            phase_indices,
            green_phase_indices,
            yellow_phase_indices,
        ) = self._get_traffic_light_phases()
        self.traffic_signals = {}
        for intersection in self.intersections:
            if not intersection["virtual"]:
                ts_id = intersection["id"]
                if "phase_sequence" in self.config.simulator.keys():
                    ts_phases = self.config.simulator["phase_sequence"]
                    ts_n_phases = len(ts_phases)
                    ts_phase_indices = list(range(ts_n_phases))
                    ts_green_phase_indices = ts_phase_indices[0::2]
                    ts_yellow_phase_indices = ts_phase_indices[1::2]
                else:
                    ts_phases = phases[ts_id]
                    ts_n_phases = len(phases[ts_id])
                    ts_phase_indices = phase_indices[ts_id]
                    ts_green_phase_indices = green_phase_indices[ts_id]
                    ts_yellow_phase_indices = yellow_phase_indices[ts_id]
                self.traffic_signals[ts_id] = {
                    "coordinates": self.ts_coordinates[ts_id],
                    "norm_coordinates": self.ts_norm_coordinates[ts_id],
                    "incoming_lanes": inc_lane_map[ts_id],
                    "outgoing_lanes": out_lane_map[ts_id],
                    "inc_to_out_lanes": inc_to_out_lane_map[ts_id],
                    "phase_to_inc_out_lanes": self.ts_phase_to_inc_out_lanes[ts_id],
                    "phases": ts_phases,
                    "n_phases": ts_n_phases,
                    "phases_min_max_times": phases_min_max_times[ts_id],
                    "phase_indices": ts_phase_indices,
                    "green_phase_indices": ts_green_phase_indices,
                    "yellow_phase_indices": ts_yellow_phase_indices,
                }
                self.traffic_signals[ts_id].update(self.config.signal)

    @property
    @lru_cache(maxsize=None)
    def intersections(self):
        """
        Extracts the intersections from the network.

        Returns:
            list: A sorted list of intersection objects.
        """
        return self.net["intersections"]

    @property
    @lru_cache(maxsize=None)
    def roads(self):
        """
        Extracts the roads from the network.

        Returns:
            list: A sorted list of road objects.
        """
        return self.net["roads"]

    @property
    @lru_cache(maxsize=None)
    def lanes(self):
        """
        Extracts the IDs of all lanes in the network.

        Returns:
            list: A sorted list of lane IDs.
        """
        lane_ids = []
        for road in self.roads:
            road_id = road["id"]
            num_lanes = len(road["lanes"])
            for i in range(num_lanes):
                lane_id = f"{road_id}_{i}"
                lane_ids.append(lane_id)
        return sorted(lane_ids)

    @property
    @lru_cache(maxsize=None)
    def traffic_signal_ids(self):
        """
        Extracts the IDs of all traffic signals in the network.

        Returns:
            list: A sorted list of traffic signal IDs.
        """
        traffic_signal_ids = [
            intersection["id"]
            for intersection in self.intersections
            if not intersection["virtual"]
        ]
        return sorted(traffic_signal_ids)

    @property
    @lru_cache(maxsize=None)
    def adjacency_matrix(self):
        """
        Computes the adjacency matrix for the traffic signal network.

        Returns:
            np.ndarray: A 2D array representing the adjacency matrix of the network.
        """
        if "neighbors" not in self.config.network.keys():
            n_traffic_signals = len(self.traffic_signal_ids)
            adjacency_matrix = np.zeros((n_traffic_signals, n_traffic_signals))
            for road in self.roads:
                start_tl = road["startIntersection"]
                end_tl = road["endIntersection"]
                if (
                    start_tl in self.traffic_signal_ids
                    and end_tl in self.traffic_signal_ids
                ):
                    start_index = self.traffic_signal_ids.index(start_tl)
                    end_index = self.traffic_signal_ids.index(end_tl)
                    adjacency_matrix[start_index, end_index] = 1.0
                    adjacency_matrix[end_index, start_index] = (
                        1.0  # assuming undirected graph
                    )
            return adjacency_matrix
        else:
            return super(NetworkParser, self)._get_adjacency_matrix()

    @property
    @lru_cache(maxsize=None)
    def k_hop_neighbors(self):
        """
        Computes the k-hop neighbors for each traffic signal in the network.

        Returns:
            dict: A dictionary where keys are traffic signal IDs and values are dictionaries
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
    def network_boundary(self):
        """
        Returns the network boundary.

        Returns:
            tuple: A tuple containing the minimum and maximum coordinates of the network.
        """
        points = [p["point"] for p in self.net["intersections"]]
        x_values = [p["x"] for p in points]
        y_values = [p["y"] for p in points]
        return (min(x_values), min(y_values)), (max(x_values), max(y_values))

    @property
    @lru_cache(maxsize=None)
    def norm_network_boundary(self):
        """
        Returns the normalized network boundary.

        Returns:
            list: A list containing the width and height of the network.
        """
        points = [p["point"] for p in self.net["intersections"]]
        x_values = [p["x"] for p in points]
        y_values = [p["y"] for p in points]
        return [max(x_values) - min(x_values), max(y_values) - min(y_values)]

    @property
    @lru_cache(maxsize=None)
    def ts_phase_to_inc_out_lanes(self):
        """
        Extracts the mapping of incoming lanes to outgoing lanes for each traffic signal phase.

        Returns:
            dict: A dictionary where keys are traffic signal IDs and values are dictionaries
                  mapping phase indices to dictionaries of incoming lanes and their corresponding outgoing lanes.
        """
        ts_phase_to_inc_out_lanes = {}
        for intersection in self.intersections:
            if not intersection["virtual"] and "trafficLight" in intersection:
                intersection_id = intersection["id"]
                roadlink_to_lanelink = []
                for roadlink in intersection["roadLinks"]:
                    lanelinks = []
                    for lanelink in roadlink["laneLinks"]:
                        startlane = (
                            roadlink["startRoad"]
                            + "_"
                            + str(lanelink["startLaneIndex"])
                        )
                        endlane = (
                            roadlink["endRoad"] + "_" + str(lanelink["endLaneIndex"])
                        )
                        lanelinks.append((startlane, endlane))
                    roadlink_to_lanelink.append(lanelinks)
                phases = intersection["trafficLight"]["lightphases"]
                for i, phase in enumerate(phases):
                    available_road_links = phase["availableRoadLinks"]
                    phase_lanelinks = []
                    for roadlink_index in available_road_links:
                        phase_lanelinks.extend(roadlink_to_lanelink[roadlink_index])
                    if intersection_id not in ts_phase_to_inc_out_lanes:
                        ts_phase_to_inc_out_lanes[intersection_id] = {}
                    if i not in ts_phase_to_inc_out_lanes[intersection_id]:
                        ts_phase_to_inc_out_lanes[intersection_id][i] = {}
                    for inc_lane, out_lane in phase_lanelinks:
                        if (
                            inc_lane
                            not in ts_phase_to_inc_out_lanes[intersection_id][i].keys()
                        ):
                            ts_phase_to_inc_out_lanes[intersection_id][i][inc_lane] = []
                        ts_phase_to_inc_out_lanes[intersection_id][i][inc_lane].append(
                            out_lane
                        )
        return ts_phase_to_inc_out_lanes

    @property
    @lru_cache(maxsize=None)
    def neighbors_lanes(self):
        """
        Extracts a dictionary of neighbors and their connecting lanes for each traffic signal.

        Returns:
            dict: A dictionary where keys are traffic signal IDs and values are dictionaries
                  mapping neighbor traffic signal IDs to lists of connecting lane IDs.
        """
        if "neighbors_lanes" not in self.config.network.keys():
            neighbors_lanes = {}
            for ts_id in self.traffic_signal_ids:
                neighbors_lanes[ts_id] = {}
                ts_index = self.traffic_signal_ids.index(ts_id)
                neighbor_indices = np.where(self.adjacency_matrix[ts_index] > 0)[0]
                neighbor_ids = [
                    self.traffic_signal_ids[index] for index in neighbor_indices
                ]
                for neighbor in neighbor_ids:
                    connecting_lanes = []
                    for road in self.roads:
                        if (
                            road["startIntersection"] == ts_id
                            and road["endIntersection"] == neighbor
                        ):
                            road_id = road["id"]
                            for lane_index, _ in enumerate(road["lanes"]):
                                connecting_lanes.append(f"{road_id}_{lane_index}")
                    neighbors_lanes[ts_id][neighbor] = connecting_lanes
            return neighbors_lanes
        else:
            return self.config.network["neighbors_lanes"]

    @property
    @lru_cache(maxsize=None)
    def neighbors_offsets(self):
        """
        Computes the offsets for each traffic signal based on the travel time
        to its neighbors.

        Returns:
            dict: A dictionary where keys are traffic signal IDs and values are dictionaries
                  mapping neighbor traffic signal IDs to their respective offsets.
        """
        neighbors_offsets = {ts_id: {} for ts_id in self.traffic_signal_ids}
        for i, ts_id in enumerate(self.traffic_signal_ids):
            neighbors_lanes = self.neighbors_lanes[ts_id]
            if not neighbors_lanes:
                continue
            for j, neigh_ts_id in enumerate(self.traffic_signal_ids):
                if neigh_ts_id in neighbors_lanes.keys():
                    travel_time = sum(
                        [
                            self.lane_lengths[lane] / self.lane_max_speeds[lane]
                            for lane in neighbors_lanes[neigh_ts_id]
                        ]
                    )
                    offset_t = travel_time / len(neighbors_lanes[neigh_ts_id])
                    offset_t /= self.config.simulator["delta_time"]
                    offset_t = int(offset_t)
                    neighbors_offsets[ts_id][neigh_ts_id] = offset_t
        return neighbors_offsets

    @property
    @lru_cache(maxsize=None)
    def lane_lengths(self):
        """
        Extracts lengths for each lane in the network.

        Returns:
            dict: A dictionary where keys are lane IDs and values are lengths.
        """
        lane_lengths = {}
        for road in self.roads:
            start_intersection = self._id_to_intersection(road["startIntersection"])
            end_intersection = self._id_to_intersection(road["endIntersection"])
            for i in range(len(road["lanes"])):
                lane_id = f"{road['id']}_{i}"
                lane_lengths[lane_id] = np.linalg.norm(
                    np.array(
                        [
                            start_intersection["point"]["x"],
                            start_intersection["point"]["y"],
                        ]
                    )
                    - np.array(
                        [
                            end_intersection["point"]["x"],
                            end_intersection["point"]["y"],
                        ]
                    )
                )
        return lane_lengths

    @property
    @lru_cache(maxsize=None)
    def lane_max_speeds(self):
        """
        Extracts maximum speeds for each lane in the network.

        Returns:
            dict: A dictionary where keys are lane IDs and values are maximum speeds.
        """
        lane_max_speeds = {}
        for road in self.roads:
            for i, lane in enumerate(road["lanes"]):
                lane_id = f"{road['id']}_{i}"
                lane_max_speeds[lane_id] = lane["maxSpeed"]
        return lane_max_speeds

    @property
    @lru_cache(maxsize=None)
    def lane_indices(self):
        """
        Extracts lane indices for each lane in the network.

        Returns:
            dict: A dictionary where keys are lane IDs and values are lane indices.
        """
        lane_indices = {}
        for road in self.roads:
            num_lanes = len(road["lanes"])
            for i in range(num_lanes):
                lane_id = f"{road['id']}_{i}"
                lane_indices[lane_id] = i
        return lane_indices

    @property
    @lru_cache(maxsize=None)
    def lane_angles(self):
        """
        Extracts angles for each lane in the network.

        Returns:
            dict: A dictionary where keys are lane IDs and values are angles in degrees.
        """
        lane_angles = {}
        for road in self.roads:
            start_intersection = self._id_to_intersection(road["startIntersection"])
            end_intersection = self._id_to_intersection(road["endIntersection"])
            dx = end_intersection["point"]["x"] - start_intersection["point"]["x"]
            dy = end_intersection["point"]["y"] - start_intersection["point"]["y"]
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            num_lanes = len(road["lanes"])
            for i in range(num_lanes):
                lane_id = f"{road['id']}_{i}"
                lane_angles[lane_id] = angle_deg
        return lane_angles

    @property
    @lru_cache(maxsize=None)
    def ts_coordinates(self):
        """
        Extracts coordinates for each traffic signal.

        Returns:
            dict: A dictionary where keys are traffic signal IDs and values are lists of coordinates.
        """
        ts_coordinates = {}
        for intersection in self.intersections:
            if not intersection["virtual"]:
                intersection_id = intersection["id"]
                ts_coordinates[intersection_id] = [
                    intersection["point"]["x"],
                    intersection["point"]["y"],
                ]
        return ts_coordinates

    @property
    @lru_cache(maxsize=None)
    def ts_norm_coordinates(self):
        """
        Extracts normalized coordinates for each traffic signal.

        Returns:
            dict: A dictionary where keys are traffic signal IDs and values are lists of normalized coordinates.
        """
        ts_norm_coordinates = {}
        for intersection in self.intersections:
            if not intersection["virtual"]:
                intersection_id = intersection["id"]
                ts_norm_coordinates[intersection_id] = [
                    intersection["point"]["x"] / self.norm_network_boundary[0],
                    intersection["point"]["y"] / self.norm_network_boundary[1],
                ]
        return ts_norm_coordinates

    @property
    @lru_cache(maxsize=None)
    def directional_lanes(self):
        """
        Extracts a dictionary of directional lanes for each traffic signal.

        Returns:
            dict: A dictionary where keys are traffic signal IDs and values are dictionaries
                  mapping end intersection IDs to lists of lane IDs.
        """
        directional_lanes = {}
        for ts_id in self.traffic_signal_ids:
            directional_lanes[ts_id] = {}
            for intersection in self.intersections:
                if intersection["id"] == ts_id and not intersection["virtual"]:
                    for road_link in intersection["roadLinks"]:
                        start_road_id = road_link["startRoad"]
                        end_road_id = road_link["endRoad"]
                        end_road = next(
                            (road for road in self.roads if road["id"] == end_road_id),
                            None,
                        )
                        if (
                            end_road
                            and end_road["endIntersection"] in self.traffic_signal_ids
                        ):
                            end_intersection_id = end_road["endIntersection"]
                            if end_intersection_id not in directional_lanes[ts_id]:
                                directional_lanes[ts_id][end_intersection_id] = []
                            for lane_link in road_link["laneLinks"]:
                                start_lane_index = lane_link["startLaneIndex"]
                                lane_id = f"{start_road_id}_{start_lane_index}"
                                if (
                                    lane_id
                                    not in directional_lanes[ts_id][end_intersection_id]
                                ):
                                    directional_lanes[ts_id][
                                        end_intersection_id
                                    ].append(lane_id)
        return directional_lanes

    @property
    @lru_cache(maxsize=None)
    def in_degrees(self):
        """
        Computes the in-degrees for each traffic signal in the network.

        Returns:
            np.ndarray: A 1D array representing the in-degrees of traffic signals.
        """
        in_degrees = np.sum(self.adjacency_matrix, axis=0)
        return in_degrees

    @property
    @lru_cache(maxsize=None)
    def out_degrees(self):
        """
        Computes the out-degrees for each traffic signal in the network.

        Returns:
            np.ndarray: A 1D array representing the out-degrees of traffic signals.
        """
        out_degrees = np.sum(self.adjacency_matrix, axis=1)
        return out_degrees

    @property
    @lru_cache(maxsize=None)
    def distance_matrix(self):
        """
        Computes the distance matrix for the traffic signal network.

        Returns:
            np.ndarray: A 2D array representing the distance between traffic signals.
        """
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        n_traffic_signals = len(self.traffic_signal_ids)
        distance_matrix = np.zeros((n_traffic_signals, n_traffic_signals))
        for i in range(n_traffic_signals):
            lengths, _ = nx.single_source_dijkstra(G, i)
            for j in lengths:
                distance_matrix[i, j] = lengths[j]
        return distance_matrix

    @property
    @lru_cache(maxsize=None)
    def edge_features(self):
        """
        Extracts edge features for the traffic signal network.

        Returns:
            np.ndarray: A 3D array of edge features, where each element represents
                        the number of lanes and average lane length between traffic signals.
        """
        n_traffic_signals = len(self.traffic_signal_ids)
        edge_features = np.zeros((n_traffic_signals, n_traffic_signals, 2))
        for road in self.roads:
            start_tl = road["startIntersection"]
            end_tl = road["endIntersection"]
            if (start_tl in self.traffic_signal_ids) and (
                end_tl in self.traffic_signal_ids
            ):
                start_index = self.traffic_signal_ids.index(start_tl)
                end_index = self.traffic_signal_ids.index(end_tl)
                num_lanes = len(road["lanes"])
                lane_lengths = []
                for i, lane in enumerate(road["lanes"]):
                    lane_id = f"{road['id']}_{i}"
                    lane_lengths.append(self.lane_lengths[lane_id])
                avg_lane_length = np.mean(lane_lengths)
                edge_features[start_index, end_index, 0] = num_lanes
                edge_features[start_index, end_index, 1] = avg_lane_length
        # Normalize the matrices
        edge_features[:, :, 0] = edge_features[:, :, 0] / np.max(edge_features[:, :, 0])
        edge_features[:, :, 1] = edge_features[:, :, 1] / np.max(edge_features[:, :, 1])
        # Final feature matrix
        # edge_features_agg = edge_features[:, :, 0] / edge_features[:, :, 1]
        return edge_features

    def _id_to_intersection(self, intersection_id):
        """
        Converts an intersection ID to its corresponding intersection object.

        Args:
            intersection_id (str): The ID of the intersection.
        Returns:
            intersection (dict): The intersection object corresponding to the ID.
        """
        for intersection in self.intersections:
            if intersection["id"] == intersection_id:
                return intersection
        raise ValueError(f"Intersection {intersection_id} not found")

    def _get_k_hop_neighbors_for_ts(self, ts_id, k):
        """
        Computes the k-hop neighbors for a given traffic signal ID.

        Args:
            ts_id (str): The ID of the traffic signal.
            k (int): The number of hops.
        Returns:  
            k_hop_neighbors_ids (list): List of k-hop neighbor traffic signal IDs. 
        """
        adjacency_matrix_power = np.linalg.matrix_power(self.adjacency_matrix, k)
        ts_index = self.traffic_signal_ids.index(ts_id)
        k_hop_neighbors_indices = np.where(adjacency_matrix_power[ts_index] > 0)[0]
        k_hop_neighbors_ids = [
            self.traffic_signal_ids[index] for index in k_hop_neighbors_indices
        ]
        return k_hop_neighbors_ids

    def _get_lane_mappings(self):
        """
        Extracts the incoming and outgoing lane mappings for each intersection.

        Returns:
            tuple: A tuple containing:
                - incoming_lane_map: A dictionary mapping intersection IDs to their incoming lanes.
                - outgoing_lane_map: A dictionary mapping intersection IDs to their outgoing lanes.
                - inc_to_out_lane_map: A dictionary mapping intersection IDs to a mapping of incoming lanes to outgoing lanes.
        """
        incoming_lane_map, outgoing_lane_map, inc_to_out_lane_map = {}, {}, {}
        for intersection in self.intersections:
            intersection_id = intersection["id"]
            if not intersection["virtual"]:
                incoming_lanes, outgoing_lanes, mapping = [], [], {}
                for roadLink in intersection["roadLinks"]:
                    start_road = roadLink["startRoad"]
                    end_road = roadLink["endRoad"]
                    for laneLink in roadLink["laneLinks"]:
                        start_lane_index = laneLink["startLaneIndex"]
                        end_lane_index = laneLink["endLaneIndex"]
                        inc_lane = f"{start_road}_{start_lane_index}"
                        out_lane = f"{end_road}_{end_lane_index}"
                        incoming_lanes.append(inc_lane)
                        outgoing_lanes.append(out_lane)
                        if inc_lane not in mapping:
                            mapping[inc_lane] = []
                        mapping[inc_lane].append(out_lane)
                incoming_lane_map[intersection_id] = sorted(list(set(incoming_lanes)))
                outgoing_lane_map[intersection_id] = sorted(list(set(outgoing_lanes)))
                inc_to_out_lane_map[intersection_id] = mapping
        return incoming_lane_map, outgoing_lane_map, inc_to_out_lane_map

    def _get_traffic_light_phases(self):
        """
        Extracts the traffic light phases and their timings for each intersection.

        Returns:
            tuple: A tuple containing:
                - phases: A dictionary mapping intersection IDs to their traffic light phases.
                - phases_min_max_times: A dictionary mapping intersection IDs to the minimum and maximum times for each phase.
                - phase_indices: A dictionary mapping intersection IDs to the indices of their phases.
                - green_phase_indices: A dictionary mapping intersection IDs to the indices of their green phases.
                - yellow_phase_indices: A dictionary mapping intersection IDs to the indices of their yellow phases.
        """
        phases = {}
        phase_indices = {}
        green_phase_indices = {}
        yellow_phase_indices = {}
        phases_min_max_times = {}
        for intersection in self.intersections:
            if not intersection["virtual"]:
                ts_id = intersection["id"]
                program = intersection["trafficLight"]["lightphases"]
                green_phases = []
                yellow_phases = []
                phases_min_max_times[ts_id] = {}
                for i, p in enumerate(program):
                    if len(p["availableRoadLinks"]) and p["time"] > 5:
                        green_phases.append(i)
                        phases_min_max_times[ts_id][i] = {
                            "min_time": self.config.signal["min_green_time"],
                            "max_time": self.config.signal["max_green_time"],
                        }
                    else:
                        yellow_phases.append(i)
                        phases_min_max_times[ts_id][i] = {
                            "min_time": self.config.signal["yellow_time"],
                            "max_time": self.config.signal["yellow_time"],
                        }
                if len(yellow_phases) == 1:  # common yellow for all
                    yellow_phase = yellow_phases[0]
                    phases[ts_id] = [
                        item
                        for pair in zip(
                            green_phases,
                            [yellow_phase] * len(green_phases),
                        )
                        for item in pair
                    ]
                else:
                    phases[ts_id] = [
                        item
                        for pair in zip(green_phases, yellow_phases)
                        for item in pair
                    ]
                green_phase_indices[ts_id] = [
                    phases[ts_id].index(g) for g in green_phases
                ]
                yellow_phase_indices[ts_id] = [
                    g + 1 for g in green_phase_indices[ts_id]
                ]
                phase_indices[ts_id] = [
                    item
                    for pair in zip(
                        green_phase_indices[ts_id], yellow_phase_indices[ts_id]
                    )
                    for item in pair
                ]
        return (
            phases,
            phases_min_max_times,
            phase_indices,
            green_phase_indices,
            yellow_phase_indices,
        )

    def _get_networkx_representation(self):
        """
        Converts the network into a NetworkX directed graph representation.

        Returns:
            G (networkx.DiGraph): A directed graph representation of the network.
        """
        G = nx.DiGraph()
        for intersection in self.intersections:
            G.add_node(
                intersection["id"],
                color="red" if not intersection["virtual"] else "gray",
                pos=(intersection["point"]["x"], intersection["point"]["y"]),
            )
        for road in self.roads:
            start_intersection = road["startIntersection"]
            end_intersection = road["endIntersection"]
            for i, lane in enumerate(road["lanes"]):
                lane_id = f"{road['id']}_{i}"
                G.add_edge(start_intersection, end_intersection, label=lane_id)
        return G

    def plot_network(self, figsize=(12, 12)):
        """
        Plots the network using NetworkX and Matplotlib.

        Args:
            figsize (tuple): Size of the figure to plot.
        """
        import matplotlib.pyplot as plt

        G = self._get_networkx_representation()
        node_colors = [node[1]["color"] for node in G.nodes(data=True)]
        pos = nx.get_node_attributes(G, "pos")
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx(
            G,
            pos,
            with_labels=True,
            font_weight="bold",
            node_color=node_colors,
            arrowsize=10,
            ax=ax,
        )
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=8, ax=ax
        )
        plt.show()

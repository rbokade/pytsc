from abc import ABC

import numpy as np


class BaseNetworkParser(ABC):
    """
    Base class for network parsers in traffic signal control.
    This class defines the interface for different network parsers
    and provides common functionality for network management.
    Args:
        config (Config): Configuration object containing simulation parameters.
        traffic_signal_ids (list): List of traffic signal IDs in the network.
    """
    def __init__(self, config):
        self.config = config

    def _get_adjacency_matrix(self):        
        """
        The adjacency matrix is a square matrix used to represent a finite graph.
        The elements of the matrix indicate whether pairs of vertices are adjacent or not in the graph.
        In the context of traffic signal control, the adjacency matrix can be used to represent the connections
        between different traffic signals in the network.
        """
        if "neighbors" in self.config.network.keys():
            n_traffic_signals = len(self.traffic_signal_ids)
            adjacency_matrix = np.zeros((n_traffic_signals, n_traffic_signals))
            for tl, tl_id in enumerate(self.traffic_signal_ids):
                for n_tl, n_tl_id in enumerate(self.traffic_signal_ids):
                    ids = self.config.network["neighbors"]
                    neighbors = self.config.network["neighbors"][tl_id]
                    if tl_id in ids and n_tl_id in neighbors:
                        adjacency_matrix[tl, n_tl] = 1.0
            return adjacency_matrix

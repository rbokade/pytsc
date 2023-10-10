from abc import ABC

import numpy as np


class BaseNetworkParser(ABC):
    def __init__(self, config):
        self.config = config

    def _get_adjacency_matrix(self):
        """
        Returns adjacency matrix based on input neighbors
        """
        if "neighbors" in self.config.network_config.keys():
            n_traffic_signals = len(self.traffic_signal_ids)
            adjacency_matrix = np.zeros((n_traffic_signals, n_traffic_signals))
            for tl, tl_id in enumerate(self.traffic_signal_ids):
                for n_tl, n_tl_id in enumerate(self.traffic_signal_ids):
                    ids = self.config.network_config["neighbors"]
                    neighbors = self.config.network_config["neighbors"][tl_id]
                    if tl_id in ids and n_tl_id in neighbors:
                        adjacency_matrix[tl, n_tl] = 1.0
            return adjacency_matrix

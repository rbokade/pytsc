from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    def __init__(self, simulator):
        self.simulator = simulator
        self.parsed_network = simulator.parsed_network
        self.config = simulator.config

    @abstractmethod
    def retrieve_lane_measurements(self):
        raise NotImplementedError

    @abstractmethod
    def retrieve_sim_measurements(self):
        raise NotImplementedError

    @abstractmethod
    def retrieve_ts_measurements(self):
        raise NotImplementedError

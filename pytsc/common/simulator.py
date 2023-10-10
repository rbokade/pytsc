from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    def __init__(self, parsed_network):
        self.parsed_network = parsed_network
        self.config = parsed_network.config

    @abstractmethod
    def retrieve_step_measurements(self):
        return NotImplementedError

    @abstractmethod
    def start_simulator(self):
        raise NotImplementedError

    @abstractmethod
    def simulator_step(self):
        raise NotImplementedError

    @abstractmethod
    def close_simulator(self):
        raise NotImplementedError

    @abstractmethod
    def is_terminated(self):
        raise NotImplementedError

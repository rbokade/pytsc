from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """
    Base class for data retrieval in traffic signal control.
    This class defines the interface for different data retrieval methods
    and provides common functionality for data management.
    
    Args:
        simulator (Simulator): Simulator object containing simulation parameters and network information.
    """
    def __init__(self, simulator):
        self.simulator = simulator
        self.parsed_network = simulator.parsed_network
        self.config = simulator.config

    @abstractmethod
    def retrieve_lane_measurements(self):
        """
        Retrieve lane measurements from the simulator.
        
        Returns:
            dict: Dictionary containing lane measurements.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_sim_measurements(self):
        """
        Retrieve simulation measurements from the simulator.
        
        Returns:
            dict: Dictionary containing simulation measurements.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_ts_measurements(self):
        """
        Retrieve traffic signal measurements from the simulator.
        
        Returns:
            dict: Dictionary containing traffic signal measurements.
        """
        raise NotImplementedError

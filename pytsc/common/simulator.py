from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    """
    Base class for traffic signal control simulators.
    This class defines the interface for different simulators
    and provides common functionality for simulation management.
    Args:
        parsed_network (ParsedNetwork): Parsed network object containing network information.
    """
    def __init__(self, parsed_network):
        self.parsed_network = parsed_network
        self.config = parsed_network.config

    @abstractmethod
    def retrieve_step_measurements(self):
        """
        Retrieve measurements for the current simulation step.
        Returns:
            dict: Dictionary containing measurements for the current step.
        """
        return NotImplementedError

    @abstractmethod
    def start_simulator(self):
        """
        Start the simulator.
        Returns:
            bool: True if the simulator started successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def simulator_step(self):
        """
        Perform a simulation step.
        Returns:
            bool: True if the simulation step was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def close_simulator(self):
        """
        Close the simulator.
        Returns:
            bool: True if the simulator closed successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def is_terminated(self):
        """
        Check if the simulation has terminated.
        Returns:
            bool: True if the simulation has terminated, False otherwise.
        """
        raise NotImplementedError

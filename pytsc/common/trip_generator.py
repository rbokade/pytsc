from abc import ABC


class TripGenerator(ABC):
    """
    Base class for trip generators in traffic signal control.
    This class defines the interface for different trip generators
    and provides common functionality for trip generation.
    """

    def generate_flows(self):
        """
        Generate flows for the trip generator.
        """
        raise NotImplementedError

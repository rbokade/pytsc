from abc import ABC, abstractmethod


class BaseMetricsParser(ABC):
    """
    Base class for metrics parsers in traffic signal control.
    This class defines the interface for different metrics parsers
    and provides common functionality for metrics management.
    
    Args:
        parsed_network (ParsedNetwork): Parsed network object containing network information.
    """

    @abstractmethod
    def get_step_stats(self):
        """
        Get the statistics for the current simulation step.
        
        Returns:
            dict: Dictionary containing the statistics for the current step.
        """
        raise NotImplementedError

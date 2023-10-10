from abc import ABC, abstractmethod


class BaseMetricsParser(ABC):
    @abstractmethod
    def get_step_stats(self):
        raise NotImplementedError

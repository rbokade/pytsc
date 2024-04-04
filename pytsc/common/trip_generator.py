from abc import ABC


class TripGenerator(ABC):

    def generate_flows(self):
        raise NotImplementedError

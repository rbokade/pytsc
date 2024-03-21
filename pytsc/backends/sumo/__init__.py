from pytsc.backends.sumo.config import Config as SUMOConfig
from pytsc.backends.sumo.metrics import MetricsParser as SUMOMetricsParser
from pytsc.backends.sumo.network_parser import (
    NetworkParser as SUMONetworkParser,
)
from pytsc.backends.sumo.retriever import Retriever as SUMORetriever
from pytsc.backends.sumo.simulator import Simulator as SUMOSimulator
from pytsc.backends.sumo.traffic_signal import (
    TrafficSignal as SUMOTrafficSignal,
)

SUMO_MODULES = {
    "config": SUMOConfig,
    "metrics_parser": SUMOMetricsParser,
    "network_parser": SUMONetworkParser,
    "retriever": SUMORetriever,
    "simulator": SUMOSimulator,
    "traffic_signal": SUMOTrafficSignal,
}

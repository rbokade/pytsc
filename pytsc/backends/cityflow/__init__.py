from pytsc.backends.cityflow.config import Config as CityFlowConfig
from pytsc.backends.cityflow.metrics import (
    MetricsParser as CityFlowMetricsParser,
)
from pytsc.backends.cityflow.network_parser import (
    NetworkParser as CityFlowNetworkParser,
)
from pytsc.backends.cityflow.retriever import Retriever as CityFlowRetriever
from pytsc.backends.cityflow.simulator import Simulator as CityFlowSimulator
from pytsc.backends.cityflow.traffic_signal import (
    TrafficSignal as CityFlowTrafficSignal,
)


CITYFLOW_MODULES = {
    "config": CityFlowConfig,
    "metrics_parser": CityFlowMetricsParser,
    "network_parser": CityFlowNetworkParser,
    "retriever": CityFlowRetriever,
    "simulator": CityFlowSimulator,
    "traffic_signal": CityFlowTrafficSignal,
}

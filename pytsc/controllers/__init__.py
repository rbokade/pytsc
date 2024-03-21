from pytsc.controllers.traditional_algorithms import (
    FixedTimePhaseSelector,
    GreedyPhaseSelector,
    MaxPressurePhaseSelector,
    SOTLPhaseSelector,
)


TRADITIONAL_CONTROLLERS = {
    "fixed_time": FixedTimePhaseSelector,
    "greedy": GreedyPhaseSelector,
    "max_pressure": MaxPressurePhaseSelector,
    "sotl": SOTLPhaseSelector,
}

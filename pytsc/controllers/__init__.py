from pytsc.controllers.controllers import (
    FixedTimeController,
    GreedyController,
    MaxPressureController,
    SOTLController,
    RandomController,
)
from pytsc.controllers.rl_controller import (
    RLController,
    MixedRLController,
    SpecializedMARLController,
    MultiGeneralizedAgentRLController,
    MultiGeneralizedGraphAgentRLController,
    SingleGeneralizedAgentRLController,
)


CONTROLLERS = {
    "fixed_time": FixedTimeController,
    "greedy": GreedyController,
    "max_pressure": MaxPressureController,
    "sotl": SOTLController,
    "random": RandomController,
    "rl": RLController,
    "mixed_rl": MixedRLController,
    "specialized_marl": SpecializedMARLController,
    "multi_generalized_agent": MultiGeneralizedAgentRLController,
    "multi_generalized_graph_agent": MultiGeneralizedGraphAgentRLController,
    "single_generalized_agent": SingleGeneralizedAgentRLController,
}

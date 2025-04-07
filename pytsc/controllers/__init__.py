from pytsc.controllers.controllers import (
    FixedTimeController,
    GreedyController,
    MaxPressureController,
    RandomController,
    SOTLController,
)
from pytsc.controllers.rl_controller import (
    MixedRLController,
    MultiGeneralizedAgentRLController,
    MultiGeneralizedGraphAgentRLController,
    MultiGeneralizedVarRobustAgentRLController,
    MultiGeneralizedVarRobustGraphAgentRLController,
    RLController,
    SingleGeneralizedAgentRLController,
    SpecializedMARLController,
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
    "multi_generalized_var_robust_agent": MultiGeneralizedVarRobustAgentRLController,
    "multi_generalized_graph_agent": MultiGeneralizedGraphAgentRLController,
    "multi_generalized_var_robust_graph_agent": MultiGeneralizedVarRobustGraphAgentRLController,
    "single_generalized_agent": SingleGeneralizedAgentRLController,
}

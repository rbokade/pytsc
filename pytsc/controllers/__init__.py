from pytsc.controllers.controllers import (
    FixedTimeController,
    GreedyController,
    MaxPressureController,
    SOTLController,
    RandomController,
)
from pytsc.controllers.rl_controller import RLController

CONTROLLERS = {
    "fixed_time": FixedTimeController,
    "greedy": GreedyController,
    "max_pressure": MaxPressureController,
    "sotl": SOTLController,
    "random": RandomController,
    "rl": RLController,
}

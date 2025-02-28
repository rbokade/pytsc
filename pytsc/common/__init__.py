from pytsc.common.actions import (
    PhaseSwitchActionSpace,
    PhaseSelectionActionSpace,
)
from pytsc.common.observations import PositionMatrix
from pytsc.common.reward import QueueLength, MaxPressure

ACTION_SPACES = {
    "phase_switch": PhaseSwitchActionSpace,
    "phase_selection": PhaseSelectionActionSpace,
}

OBSERVATION_SPACES = {
    "position_matrix": PositionMatrix,
}

REWARD_FUNCTIONS = {
    "queue_length": QueueLength,
    "max_pressure": MaxPressure,
}

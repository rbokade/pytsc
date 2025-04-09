from pytsc.common.actions import PhaseSelectionActionSpace, PhaseSwitchActionSpace
from pytsc.common.observations import LaneFeatures, PositionMatrix
from pytsc.common.reward import MaxPressure, QueueLength

ACTION_SPACES = {
    "phase_switch": PhaseSwitchActionSpace,
    "phase_selection": PhaseSelectionActionSpace,
}

OBSERVATION_SPACES = {
    "position_matrix": PositionMatrix,
    "lane_features": LaneFeatures,
}

REWARD_FUNCTIONS = {
    "queue_length": QueueLength,
    "max_pressure": MaxPressure,
}

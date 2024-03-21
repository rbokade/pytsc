from pytsc.common.actions import (
    PhaseSwitchActionSpace,
    PhaseSelectionActionSpace,
)
from pytsc.common.observations import PositionMatrix, LaneFeatures


ACTION_SPACES = {
    "phase_switch": PhaseSwitchActionSpace,
    "phase_selection": PhaseSelectionActionSpace,
}

OBSERVATION_SPACES = {
    "position_matrix": PositionMatrix,
    "lane_features": LaneFeatures,
}

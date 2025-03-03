from pytsc.controllers.controllers import BaseController

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class LaneAttentionAggregator(nn.Module):
    """
    Aggregates lane features (from max_n_controlled_lanes) using self-attention.
    Input shape: (bs, n_agents, max_n_lanes, lane_feature_dim)
    Output shape: (bs, n_agents, hidden_dim)
    """

    def __init__(self, input_shape, hidden_dim):
        super(LaneAttentionAggregator, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_proj = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lane_features):
        """
        lane_features: (bs * n_agents, max_n_lanes, lane_feature_dim)
        """
        proj = self.fc_proj(lane_features)
        scores = self.attn(proj)
        attn_weights = F.softmax(scores, dim=-2)
        aggregated = torch.sum(attn_weights * proj, dim=-2)
        return aggregated


class TSCAgent(nn.Module):
    def __init__(self, hidden_dim, n_actions):
        super(TSCAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.lane_obs_encoder = LaneAttentionAggregator(17, hidden_dim)
        self.phase_encoder = nn.Linear(10, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.h = self.init_hidden()

    def init_hidden(self):
        return self.phase_encoder.weight.new(1, self.hidden_dim).zero_()

    def forward(self, lane_features, phase_ids):
        lane_feats = self.lane_obs_encoder(lane_features)
        phase_feats = self.phase_encoder(phase_ids)
        x = lane_feats + phase_feats
        h_in = self.h.reshape(-1, self.hidden_dim)
        self.h = self.rnn(x, h_in)
        q = self.fc2(self.h)
        return q

    def select_action(self, obs, act_mask):
        lane_features, phase_ids = self._prepare_inputs(obs)
        q = self.forward(lane_features, phase_ids)
        q[act_mask] = -float("inf")
        action = torch.argmax(q.squeeze(0)).item()
        return action

    def _prepare_inputs(self, obs):
        lane_features = obs[:, :-10]
        phase_ids = obs[:, -10:]
        lane_features = lane_features.reshape(1, -1, 17)
        phase_ids = phase_ids.reshape(1, 10)
        return lane_features, phase_ids


class RLController(BaseController):
    hidden_dim = 128
    n_actions = 2

    def __init__(self, traffic_signal, model_path="pytsc/controllers/agent.th"):
        super(RLController, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller
        self.agent = TSCAgent(self.hidden_dim, self.n_actions)
        self._load_model(model_path=model_path)
        self.prev_action = None

    def _load_model(self, model_path):
        self.agent.load_state_dict(
            torch.load(
                model_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
        )
        self.agent.eval()

    def _get_action_mask(self):
        allowable_switches = self.controller.get_allowable_phase_switches()
        n_phases = self.controller.n_phases
        current_phase_index = self.controller.program.current_phase_index
        next_phase_index = (current_phase_index + 1) % n_phases
        mask = [0, 0]
        if allowable_switches[current_phase_index]:
            mask[0] = 1
        if allowable_switches[next_phase_index]:
            mask[1] = 1
        return torch.tensor(mask).unsqueeze(0).bool()

    def get_action(self, inp):
        obs = inp["observation"]
        act_mask = self._get_action_mask()
        obs = torch.tensor(obs).float().unsqueeze(0)
        action = self.agent.select_action(obs, act_mask)
        return action

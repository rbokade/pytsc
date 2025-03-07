import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx import adjacency_data
from torch.distributions import Categorical

from pytsc.controllers.controllers import BaseController


class LaneAttentionAggregator(nn.Module):
    """
    Aggregates lane features (from max_n_controlled_lanes)
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
        self.lane_obs_encoder = LaneAttentionAggregator(27, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.h = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)

    def forward(self, inputs):
        x = self.lane_obs_encoder(inputs)
        h_in = self.h.reshape(-1, self.hidden_dim)
        self.h = self.rnn(x, h_in)
        q = self.fc2(self.h)
        return q

    # def select_action(self, obs, act_mask):
    #     inputs = self._prepare_inputs(obs)
    #     logits = self.forward(inputs)
    #     logits[~act_mask] = -999
    #     pi = Categorical(logits=logits)
    #     action = pi.sample().item()
    #     return action

    # def _prepare_inputs(self, obs):
    #     lane_features = obs[:, :-10]
    #     phase_ids = obs[:, -10:]
    #     lane_features = lane_features.reshape(1, -1, 17)
    #     phase_ids = phase_ids.reshape(1, 1, 10).repeat(1, lane_features.shape[1], 1)
    #     inputs = torch.cat([lane_features, phase_ids], dim=-1)
    #     return inputs


class TSCGraphAgent(nn.Module):
    def __init__(self, hidden_dim, n_actions, adjacency_matrix):
        super(TSCGraphAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.lane_obs_encoder = LaneAttentionAggregator(27, hidden_dim)
        self.comm = GraphAttentionLayer(
            input_shape=hidden_dim,
            output_shape=hidden_dim,
            hidden_dim=hidden_dim,
        )
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.h = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)

    def forward(self, input, hidden_state):
        x = self.lane_obs_encoder(input)
        x = self.communicate(x)
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        self.h = self.rnn(x, h_in)
        q = self.fc2(self.h)
        return q

    def communicate(self, input):
        bs, _ = input.shape
        x = input.view(bs, 1, -1)
        adj = self.adjacency_matrix.unsqueeze(0)
        adj = adj.expand(bs, -1, -1)
        h = self.comm(x, adj)
        return h.view(bs, -1)


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dim):
        super(GraphAttentionLayer, self).__init__()
        self.fc = nn.Linear(input_shape, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fc_out = nn.Linear(hidden_dim, output_shape)

    def forward(self, x, adj):
        """
        x: (bs, n_agents, hidden_dim)
        adj: (bs, n_agents, n_agents)
        Use adjacency matrix
        """
        bs, n_agents, _ = x.shape
        h = self.fc(x)
        h_i = h.unsqueeze(2).expand(-1, -1, n_agents, -1)
        h_j = h.unsqueeze(1).expand(-1, n_agents, -1, -1)
        h_cat = torch.cat((h_i, h_j), dim=-1)
        e = self.attn(h_cat).squeeze(-1)
        e = e.masked_fill(adj == 0, -1e9)
        attn_weights = F.softmax(e, dim=-1)
        h_prime = torch.bmm(attn_weights, h)
        out = self.fc_out(h_prime)
        return out


class TSCAgentEnsemble(nn.Module):
    temperature = 1.0

    def __init__(
        self, hidden_dim, n_actions, graph=False, adjacency_matrix=None, num_models=3
    ):
        super(TSCAgentEnsemble, self).__init__()
        if graph:
            self.agents = nn.ModuleList(
                [
                    TSCGraphAgent(hidden_dim, n_actions, adjacency_matrix)
                    for _ in range(num_models)
                ]
            )
        else:
            self.agents = nn.ModuleList(
                [TSCAgent(hidden_dim, n_actions) for _ in range(num_models)]
            )

    def forward(self, inputs):
        outputs = [agent(inputs) for agent in self.agents]
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        return ensemble_output

    def _prepare_inputs(self, obs):
        lane_features = obs[:, :-10]
        phase_ids = obs[:, -10:]
        lane_features = lane_features.reshape(1, -1, 17)
        phase_ids = phase_ids.reshape(1, 1, 10).repeat(1, lane_features.shape[1], 1)
        inputs = torch.cat([lane_features, phase_ids], dim=-1)
        return inputs

    def select_action(self, obs, act_mask):
        inputs = self._prepare_inputs(obs)
        logits = self.forward(inputs)
        logits[~act_mask] = float("-inf")
        pi = Categorical(logits=logits)
        action = pi.sample().item()
        return action

    # def select_action(self, obs, act_mask):
    #     inputs = self._prepare_inputs(obs)
    #     probs_list = []
    #     for agent in self.agents:
    #         logits = agent(inputs)
    #         logits[~act_mask] = float("-inf")
    #         # Optionally, apply temperature scaling to each agent's logits
    #         scaled_logits = logits / self.temperature
    #         probs = F.softmax(scaled_logits, dim=-1)
    #         probs_list.append(probs)
    #     # Average the probabilities across all agents
    #     avg_probs = torch.mean(torch.stack(probs_list), dim=0)
    #     pi = Categorical(probs=avg_probs)
    #     action = pi.sample().item()
    #     return action


class RLController(BaseController):
    n_actions = 2
    hidden_dim = 128
    model_paths = [
        "pytsc/controllers/demand_burst_controllers/agent.th",
    ]

    def __init__(self, traffic_signal):
        super(RLController, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller
        self.agent = TSCAgentEnsemble(
            self.hidden_dim, self.n_actions, num_models=len(self.model_paths)
        )
        self._load_models()

    def _load_models(self):
        for agent, path in zip(self.agent.agents, self.model_paths):
            agent.load_state_dict(
                torch.load(
                    path,
                    map_location=lambda storage, loc: storage,
                )
            )
            agent.eval()

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
        return torch.tensor(mask).unsqueeze(0).bool(), mask

    def get_action(self, inp):
        obs = inp["observation"]
        act_mask, mask = self._get_action_mask()
        obs = torch.tensor(obs).float().unsqueeze(0)
        action = self.agent.select_action(obs, act_mask)
        return action


class SingleGeneralizedAgentRLController(RLController):
    model_paths = [
        "pytsc/controllers/demand_burst_controllers/single_generalized_agent_1.th",
        "pytsc/controllers/demand_burst_controllers/single_generalized_agent_2.th",
        "pytsc/controllers/demand_burst_controllers/single_generalized_agent_3.th",
    ]


class MultiGeneralizedAgentRLController(RLController):
    model_paths = [
        "pytsc/controllers/demand_burst_controllers/multi_generalized_agent_1.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_agent_2.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_agent_3.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_agent_4.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_agent_5.th",
    ]


class MultiGeneralizedGraphAgentRLController(RLController):
    model_paths = [
        "pytsc/controllers/demand_burst_controllers/multi_generalized_graph_agent_1.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_graph_agent_2.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_graph_agent_3.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_graph_agent_4.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_graph_agent_5.th",
    ]

    def __init__(self, traffic_signal):
        super(MultiGeneralizedGraphAgentRLController, self).__init__(traffic_signal)
        sim = traffic_signal.controller.simulator
        adj = sim.parsed_network.adjacency_matrix
        self.agent = TSCAgentEnsemble(
            self.hidden_dim,
            self.n_actions,
            graph=True,
            adjacency_matrix=adj,
            num_models=len(self.model_paths),
        )
        self._load_models()


class SpecializedMARLController(RLController):
    model_paths = [
        "pytsc/controllers/demand_burst_controllers/specialized_marl_agent_1.th",
    ]


class MixedRLController(RLController):
    model_paths = [
        "pytsc/controllers/demand_burst_controllers/single_generalized_agent_1.th",
        "pytsc/controllers/demand_burst_controllers/single_generalized_agent_2.th",
        "pytsc/controllers/demand_burst_controllers/single_generalized_agent_3.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_agent_1.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_agent_2.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_agent_3.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_agent_4.th",
        "pytsc/controllers/demand_burst_controllers/multi_generalized_agent_5.th",
    ]

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

    def __init__(self, input_shape, hidden_dim, device="cpu"):
        super(LaneAttentionAggregator, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn = nn.Linear(input_shape + hidden_dim, 1)

    def init_hidden(self, bs, n_agents, max_n_lanes):
        return torch.zeros(bs * n_agents * max_n_lanes, self.hidden_dim).to(self.device)

    def forward(self, lane_features, hidden_state):
        """
        lane_features: (bs * n_agents, max_n_lanes, lane_feature_dim)
        # hidden_state: (bs * n_agents * max_n_lanes, hidden_dim)
        """
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        max_n_lanes = lane_features.shape[-2]
        x = F.relu(self.fc1(lane_features))
        x = x.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        h_out = h.reshape(-1, max_n_lanes, self.hidden_dim)
        attn_inp = torch.cat((lane_features, h_out), dim=-1)
        scores = self.attn(attn_inp)
        attn_weights = F.softmax(scores, dim=-2)
        aggregated = torch.sum(attn_weights * h_out, dim=-2)
        return aggregated, h


class TSCAgent(nn.Module):
    def __init__(self, n_agents, n_actions, hidden_dim, max_n_lanes):
        super(TSCAgent, self).__init__()
        self.lane_obs_encoder = LaneAttentionAggregator(27, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.h = self.init_hidden(1, n_agents, max_n_lanes)

    def init_hidden(self, bs, n_agents, max_n_lanes):
        return self.lane_obs_encoder.init_hidden(bs, n_agents, max_n_lanes)

    def forward(self, input):
        x, self.h = self.lane_obs_encoder(input, self.h)
        q = self.fc2(x)
        return q


class TSCGraphAgent(nn.Module):
    def __init__(
        self,
        n_agents,
        n_actions,
        hidden_dim,
        max_n_lanes,
        adjacency_matrix,
        dropout=0.25,
    ):
        super(TSCGraphAgent, self).__init__()
        self.n_agents = n_agents
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.lane_obs_encoder = LaneAttentionAggregator(27, hidden_dim)
        self.comm = GraphAttentionLayer(
            input_shape=hidden_dim,
            output_shape=hidden_dim,
            hidden_dim=hidden_dim,
        )
        self.fc2 = nn.Linear(2 * hidden_dim, n_actions)
        self.comm_dropout = nn.Dropout(p=self.dropout)
        self.h = self.init_hidden(1, n_agents, max_n_lanes)

    def init_hidden(self, bs, n_agents, max_n_lanes):
        return self.lane_obs_encoder.init_hidden(bs, n_agents, max_n_lanes)

    def forward(self, input):
        x, self.h = self.lane_obs_encoder(input, self.h)
        x_dropped = self.comm_dropout(x)
        c = self.communicate(x_dropped)
        x = torch.cat((x, c), dim=-1)
        q = self.fc2(x)
        return q

    def communicate(self, input):
        bs_n, _ = input.shape
        bs = bs_n // self.n_agents
        x = input.view(bs, self.n_agents, -1)
        adj = self.adjacency_matrix.unsqueeze(0)
        adj = adj.expand(bs, -1, -1)
        h = self.comm(x, adj)
        return h.view(bs_n, -1)


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
    def __init__(
        self, n_agents, n_actions, hidden_dim, max_n_lanes, num_models=3, **kwargs
    ):
        super(TSCAgentEnsemble, self).__init__()
        self.n_agents = n_agents
        graph = kwargs.pop("graph", False)
        adjacency_matrix = kwargs.pop("adjacency_matrix", None)
        if graph:
            self.agents = nn.ModuleList(
                [
                    TSCGraphAgent(
                        n_agents,
                        n_actions,
                        hidden_dim,
                        max_n_lanes,
                        adjacency_matrix,
                    )
                    for _ in range(num_models)
                ]
            )
        else:
            self.agents = nn.ModuleList(
                [
                    TSCAgent(
                        n_agents,
                        n_actions,
                        hidden_dim,
                        max_n_lanes,
                    )
                    for _ in range(num_models)
                ]
            )

    def forward(self, inputs):
        outputs = [agent(inputs) for agent in self.agents]
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        return ensemble_output

    def _prepare_inputs(self, obs):
        lane_features = obs[:, :, :-10]
        phase_ids = obs[:, :, -10:]
        lane_features = lane_features.reshape(self.n_agents, -1, 17)
        phase_ids = phase_ids.reshape(self.n_agents, 1, 10)
        phase_ids = phase_ids.repeat(1, lane_features.shape[-2], 1)
        inputs = torch.cat([lane_features, phase_ids], dim=-1)
        return inputs

    def select_action(self, obs, avail_actions):
        inputs = self._prepare_inputs(obs)
        logits = self.forward(inputs)
        avail_actions = avail_actions.squeeze(0)
        logits[avail_actions == 0] = -1e10
        logits = F.softmax(logits, dim=-1)
        pi = Categorical(logits)
        action = pi.sample().long()
        return action


class RLController:
    graph = False
    n_actions = 2
    hidden_dim = 128
    model_paths = [
        "pytsc/controllers/agent.th",
    ]

    def __init__(self, tsc_env, **kwargs):
        self.tsc_env = tsc_env
        self.n_actions = tsc_env.get_action_size()
        self.n_agents = tsc_env.n_agents
        self.max_n_lanes = tsc_env.observation_space.max_n_controlled_lanes
        self.agent = TSCAgentEnsemble(
            self.n_agents,
            self.n_actions,
            self.hidden_dim,
            self.max_n_lanes,
            num_models=len(self.model_paths),
            graph=self.graph,
            adjacency_matrix=tsc_env.parsed_network.adjacency_matrix,
        )
        self._load_models()

    def _load_models(self):
        for agent, path in zip(self.agent.agents, self.model_paths):
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            # print(f"Checkpoint from {path}:")
            # for key, value in checkpoint.items():
            #     print(f"  {key}: {value.shape}")
            # model_state = agent.state_dict()
            # print("Model state_dict:")
            # for key, value in model_state.items():
            #     print(f"  {key}: {value.shape}")
            agent.load_state_dict(checkpoint)
            agent.eval()

    def get_action(self):
        obs = self.tsc_env.get_observations()
        avail_actions = self.tsc_env.get_action_mask()
        obs = torch.tensor(obs).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions).unsqueeze(0)
        action = self.agent.select_action(obs, avail_actions)
        return action


class SingleGeneralizedAgentRLController(RLController):
    model_paths = [
        "pytsc/controllers/single_generalized_agent_1.th",
        "pytsc/controllers/single_generalized_agent_2.th",
        "pytsc/controllers/single_generalized_agent_3.th",
    ]


class MultiGeneralizedAgentRLController(RLController):
    model_paths = [
        "pytsc/controllers/multi_generalized_agent_1.th",
        "pytsc/controllers/multi_generalized_agent_2.th",
        "pytsc/controllers/multi_generalized_agent_3.th",
        "pytsc/controllers/multi_generalized_agent_4.th",
        "pytsc/controllers/multi_generalized_agent_5.th",
    ]


class MultiGeneralizedGraphAgentRLController(RLController):
    graph = True
    model_paths = [
        "pytsc/controllers/multi_generalized_graph_agent_1.th",
        "pytsc/controllers/multi_generalized_graph_agent_2.th",
        "pytsc/controllers/multi_generalized_graph_agent_3.th",
        "pytsc/controllers/multi_generalized_graph_agent_4.th",
        "pytsc/controllers/multi_generalized_graph_agent_5.th",
    ]


class SpecializedMARLController(RLController):
    model_paths = [
        "pytsc/controllers/specialized_marl_agent_1.th",
        "pytsc/controllers/specialized_marl_agent_2.th",
    ]


class MixedRLController(RLController):
    model_paths = [
        "pytsc/controllers/single_generalized_agent_1.th",
        "pytsc/controllers/single_generalized_agent_2.th",
        "pytsc/controllers/single_generalized_agent_3.th",
        "pytsc/controllers/multi_generalized_agent_1.th",
        "pytsc/controllers/multi_generalized_agent_2.th",
        "pytsc/controllers/multi_generalized_agent_3.th",
        "pytsc/controllers/multi_generalized_agent_4.th",
        "pytsc/controllers/multi_generalized_agent_5.th",
    ]

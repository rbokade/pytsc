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

    def __init__(
        self,
        static_feat_dim=7,
        pos_mat_dim=10,
        phase_id_dim=10,
        hidden_dim=128,
        n_heads=4,
        device="cpu",
    ):
        super(LaneAttentionAggregator, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.static_feat_encoder = nn.Linear(static_feat_dim, hidden_dim)
        self.pos_mat_encoder = nn.Linear(pos_mat_dim, hidden_dim // 2)
        self.phase_id_encoder = nn.Linear(phase_id_dim, hidden_dim // 2)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.kv = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads)

    def init_hidden(self, bs, n_agents, max_n_lanes):
        return torch.zeros(bs * n_agents * max_n_lanes, self.hidden_dim).to(self.device)

    def forward(self, lane_features, hidden_state):
        """
        lane_features: (bs * n_agents, max_n_lanes, lane_feature_dim)
        # hidden_state: (bs * n_agents * max_n_lanes, hidden_dim)
        """
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        bs_n, max_n_lanes, _ = lane_features.shape
        static_feats, pos_mat, phase_ids = self._prepare_inputs(lane_features)
        # (bs_n, max_n_lanes, hidden_dim)
        static_feats_embed = self.static_feat_encoder(static_feats)
        static_feats_embed = self.relu(static_feats_embed)
        # (bs_n, max_n_lanes, hidden_dim // 2)
        pos_mat_embed = self.pos_mat_encoder(pos_mat)
        pos_mat_embed = self.relu(pos_mat_embed)
        # (bs_n, max_n_lanes, hidden_dim // 2)
        phase_ids_embed = self.phase_id_encoder(phase_ids)
        phase_ids_embed = self.relu(phase_ids_embed)
        # (bs_n, max_n_lanes, hidden_dim)
        dynamic_feats = torch.cat((pos_mat_embed, phase_ids_embed), dim=-1)
        h = self.rnn(dynamic_feats.reshape(-1, self.hidden_dim), h_in)
        qs = self.q(static_feats_embed.mean(dim=1))  # (bs_n, hidden_dim)
        qs = qs.unsqueeze(0)  # (1, bs_n, hidden_dim)
        kv = self.kv(h)  # (bs_n * max_n_lanes, hidden_dim)
        kv = kv.reshape(bs_n, max_n_lanes, self.hidden_dim)
        kv_t = kv.transpose(0, 1)  # (max_n_lanes, bs_n, hidden_dim)
        attn_output, _ = self.attn(qs, kv_t, kv_t)
        aggregated = attn_output.squeeze(0)
        return aggregated, h

    def _prepare_inputs(self, lane_features):
        inp_shape = lane_features.shape[-1]
        assert inp_shape == 27, "Expected lane feature dimension to be 27"
        static_feats = lane_features[:, :, :7]  # (bs_n, max_n_lanes, 7)
        pos_mat = lane_features[:, :, 7:17]  # (bs_n, max_n_lanes, 10)
        phase_ids = lane_features[:, :, 17:]  # (bs_n, max_n_lanes, 10)
        return static_feats, pos_mat, phase_ids


class TSCAgent(nn.Module):
    def __init__(self, n_agents, n_actions, hidden_dim, max_n_lanes):
        super(TSCAgent, self).__init__()
        self.n_agents = n_agents
        self.max_n_lanes = max_n_lanes
        self.hidden_dim = hidden_dim
        self.lane_obs_encoder = LaneAttentionAggregator(hidden_dim=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        return self.lane_obs_encoder.init_hidden(1, self.n_agents, self.max_n_lanes)

    def forward(self, input, hidden_state):
        x, next_hidden_state = self.lane_obs_encoder(input, hidden_state)
        q = self.fc2(x)
        return q, next_hidden_state


class TSCGraphAgent(nn.Module):
    def __init__(
        self,
        n_agents,
        n_actions,
        hidden_dim,
        max_n_lanes,
        adjacency_matrix,
        dropout=0.0,
    ):
        super(TSCGraphAgent, self).__init__()
        self.n_agents = n_agents
        self.dropout = dropout
        self.max_n_lanes = max_n_lanes
        self.hidden_dim = hidden_dim
        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.lane_obs_encoder = LaneAttentionAggregator(hidden_dim=hidden_dim)
        self.comm = GraphAttentionLayer(hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, n_actions)
        self.comm_dropout = nn.Dropout(p=self.dropout)

    def init_hidden(self):
        return self.lane_obs_encoder.init_hidden(1, self.n_agents, self.max_n_lanes)

    def forward(self, input, hidden_state):
        x, next_hidden_state = self.lane_obs_encoder(input, hidden_state)
        x_dropped = self.comm_dropout(x)
        c = self.communicate(x_dropped)
        x = torch.cat((x, c), dim=-1)
        q = self.fc2(x)
        return q, next_hidden_state

    def communicate(self, input):
        bs_n, _ = input.shape
        bs = bs_n // self.n_agents
        x = input.view(bs, self.n_agents, -1)
        adj = self.adjacency_matrix.unsqueeze(0)
        adj = adj.expand(bs, -1, -1)
        m = self.comm(x, adj)
        return m.reshape(bs_n, -1)


class GraphAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads=4):
        super(GraphAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )

    def forward(self, x, adj):
        """
        x: (bs, n_agents, hidden_dim)
        adj: (bs, n_agents, n_agents)
        Use adjacency matrix
        """
        mask = (adj[0] == 0).float() * -1e9
        attn_output = self.attn(x, x, x, attn_mask=mask)[0]
        return attn_output


class TSCAgentEnsemble(nn.Module):
    def __init__(
        self, n_agents, n_actions, hidden_dim, max_n_lanes, num_models=3, **kwargs
    ):
        super(TSCAgentEnsemble, self).__init__()
        self.n_agents = n_agents
        self.max_n_lanes = max_n_lanes
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

    def init_hidden(self):
        hidden_states = []
        for agent in self.agents:
            hidden_states.append(agent.init_hidden())
        return hidden_states

    def forward(self, inputs, hidden_states):
        outputs = []
        next_hidden_states = []
        for agent, hidden_state in zip(self.agents, hidden_states):
            agent_out, hidden_state = agent(inputs, hidden_state)
            outputs.append(agent_out)
            next_hidden_states.append(hidden_state)
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        return ensemble_output, next_hidden_states

    def _prepare_inputs(self, obs):
        lane_features = obs[:, :, :-10]
        phase_ids = obs[:, :, -10:]
        lane_features = lane_features.reshape(self.n_agents, self.max_n_lanes, 17)
        phase_ids = phase_ids.reshape(self.n_agents, 1, 10)
        phase_ids = phase_ids.repeat(1, lane_features.shape[-2], 1)
        inputs = torch.cat([lane_features, phase_ids], dim=-1)
        return inputs

    def select_action(self, obs, hidden_states, avail_actions):
        inputs = self._prepare_inputs(obs)
        logits, next_hidden_states = self.forward(inputs, hidden_states)
        avail_actions = avail_actions.squeeze(0)
        logits[avail_actions == 0] = -1e10
        logits = F.softmax(logits, dim=-1)
        pi = Categorical(logits)
        action = pi.sample().long()
        return action, next_hidden_states


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

    def init_hidden(self):
        return self.agent.init_hidden()

    def _load_models(self):
        for agent, path in zip(self.agent.agents, self.model_paths):
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            print(f"Checkpoint from {path}:")
            for key, value in checkpoint.items():
                print(f"  {key}: {value.shape}")
            model_state = agent.state_dict()
            print("Model state_dict:")
            for key, value in model_state.items():
                print(f"  {key}: {value.shape}")
            agent.load_state_dict(checkpoint)
            agent.eval()

    def get_action(self, hidden_states):
        obs = self.tsc_env.get_observations()
        avail_actions = self.tsc_env.get_action_mask()
        obs = torch.tensor(obs).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions).unsqueeze(0)
        action, next_hidden_states = self.agent.select_action(
            obs, hidden_states, avail_actions
        )
        return action, next_hidden_states


class SingleGeneralizedAgentRLController(RLController):
    model_paths = [
        "pytsc/controllers/single_generalized_agent_1.th",
        "pytsc/controllers/single_generalized_agent_2.th",
        "pytsc/controllers/single_generalized_agent_3.th",
    ]


class MultiGeneralizedAgentRLController(RLController):
    model_paths = [
        "pytsc/controllers/multi_generalized_agent_1.th",
        # "pytsc/controllers/multi_generalized_agent_2.th",
        # "pytsc/controllers/multi_generalized_agent_3.th",
        # "pytsc/controllers/multi_generalized_agent_4.th",
        # "pytsc/controllers/multi_generalized_agent_5.th",
    ]


class MultiGeneralizedGraphAgentRLController(RLController):
    graph = True
    model_paths = [
        "pytsc/controllers/multi_generalized_graph_agent_1.th",
        # "pytsc/controllers/multi_generalized_graph_agent_2.th",
        # "pytsc/controllers/multi_generalized_graph_agent_3.th",
        # "pytsc/controllers/multi_generalized_graph_agent_4.th",
        # "pytsc/controllers/multi_generalized_graph_agent_5.th",
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

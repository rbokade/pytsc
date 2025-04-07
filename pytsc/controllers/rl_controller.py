import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class LaneAttentionAggregator(nn.Module):
    def __init__(
        self,
        static_feat_dim=9,
        pos_mat_dim=10,
        phase_id_dim=20,
        hidden_dim=128,
        n_heads=4,
        device="cpu",
    ):
        super(LaneAttentionAggregator, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        # self.kqv = nn.Sequential(
        #     nn.Linear(static_feat_dim + pos_mat_dim + phase_id_dim, hidden_dim),
        #     nn.LeakyReLU(),
        # )
        # self.lane_aggregator = nn.MultiheadAttention(
        #     embed_dim=hidden_dim, num_heads=n_heads, batch_first=True
        # )
        self.lane_mlp = nn.Sequential(
            nn.Linear(static_feat_dim + pos_mat_dim + phase_id_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )

    def forward(self, static_feats, pos_mats, phase_ids):
        """
        static_feats: (bs * n_agents, max_n_lanes, lane_feature_dim)
        pos_mats: (bs * n_agents, max_n_lanes, pos_mat_dim)
        phase_ids: (bs * n_agents, 1, phase_id_dim)
        # hidden_state: (bs * n_agents, hidden_dim)
        """
        bs_n, max_n_lanes, inp_shape = pos_mats.shape
        phase_ids = phase_ids.expand(-1, max_n_lanes, -1)
        x = torch.cat((static_feats, pos_mats, phase_ids), dim=-1)
        # x = self.kqv(x)
        # x, _ = self.lane_aggregator(x, x, x, need_weights=False)
        # x = x.squeeze(1)
        x = self.lane_mlp(x)
        x = x.view(bs_n, max_n_lanes, -1).mean(dim=1).view(bs_n, -1)
        return x


class GraphAttentionComm(nn.Module):
    def __init__(self, adjacency_matrix, hidden_dim=128, n_heads=4):
        super(GraphAttentionComm, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.adjacency_matrix = adjacency_matrix  # (n_agents, n_agents)
        self.comm = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

    def forward(self, x, dropout=0.0):
        if dropout > 0:
            x_dropped = F.dropout(x, p=dropout, training=True)
        else:
            x_dropped = x
        attn_mask = (1 - self.adjacency_matrix).clone()
        num_agents = attn_mask.shape[0]
        for i in range(num_agents):
            attn_mask[i, i] = 0
        attn_mask = attn_mask.to(torch.bool)
        attn_out, _ = self.comm(
            x_dropped,
            x_dropped,
            x_dropped,
            attn_mask=attn_mask,
            need_weights=False,
        )
        out = x + attn_out
        return out


class TSCAgent(nn.Module):
    def __init__(self, n_agents, n_actions, hidden_dim, max_n_lanes):
        super(TSCAgent, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.max_n_lanes = max_n_lanes
        self.lane_obs_encoder = LaneAttentionAggregator(hidden_dim=hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        return torch.zeros(self.n_agents, self.hidden_dim)

    def forward(
        self,
        static_lane_feats,
        pos_mats,
        phase_ids,
        hidden_state,
        **kwargs,
    ):
        x = self.lane_obs_encoder(static_lane_feats, pos_mats, phase_ids)
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc_out(h)
        return q, h


class TSCGraphAgent(nn.Module):
    def __init__(
        self,
        n_agents,
        n_actions,
        hidden_dim,
        max_n_lanes,
        adjacency_matrix,
        graph,
        n_heads=4,
    ):
        super(TSCGraphAgent, self).__init__()
        self.graph = graph
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.max_n_lanes = max_n_lanes
        self.lane_obs_encoder = LaneAttentionAggregator(hidden_dim=hidden_dim)
        adj = self._get_adjcency_matrix(adjacency_matrix)
        self.communicate = GraphAttentionComm(adj, hidden_dim, n_heads)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_actions)

    def _get_adjcency_matrix(self, adjacency_matrix):
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        # add self loops if not present
        if not torch.all(adjacency_matrix.diagonal() == 1):
            adjacency_matrix = adjacency_matrix + torch.eye(adjacency_matrix.shape[0])
            adjacency_matrix = torch.clamp(adjacency_matrix, 0, 1)
        return adjacency_matrix

    def init_hidden(self):
        return torch.zeros(self.n_agents, self.hidden_dim)

    def forward(
        self, static_lane_feats, pos_mats, phase_ids, hidden_state, dropout=0.0
    ):
        x = self.lane_obs_encoder(static_lane_feats, pos_mats, phase_ids)
        bs = x.shape[0] // self.n_agents
        x = x.view(bs, self.n_agents, -1)
        x = self.communicate(x, dropout=dropout)
        x = x.view(bs * self.n_agents, -1)
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc_out(x)
        return q, h


class TSCAgentEnsemble(nn.Module):
    max_phases = 20
    max_n_controlled_lanes = 16

    def __init__(
        self,
        n_agents,
        n_actions,
        hidden_dim,
        max_n_lanes,
        n_models,
        adjacency_matrix,
        static_lane_feats,
        graph,
        action_selection_strategy="mean",
    ):
        super(TSCAgentEnsemble, self).__init__()
        self.n_agents = n_agents
        self.max_n_lanes = max_n_lanes
        self.action_selection_strategy = action_selection_strategy
        if graph:
            self.agents = nn.ModuleList(
                [
                    TSCGraphAgent(
                        n_agents,
                        n_actions,
                        hidden_dim,
                        max_n_lanes,
                        adjacency_matrix,
                        graph,
                    )
                    for _ in range(n_models)
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
                    for _ in range(n_models)
                ]
            )
        self.static_lane_feats = torch.tensor(
            static_lane_feats,
            dtype=torch.float32,
        )

    def _prepare_inputs(self, obs):
        lane_feats = obs[:, :, : -self.max_phases].view(
            self.n_agents, self.max_n_lanes, -1
        )
        pos_mats = lane_feats[:, :, -10:].to(torch.float32)
        static_lane_feats = lane_feats[:, :, :-10].to(torch.float32)
        phase_ids = (
            obs[:, :, -self.max_phases :].to(torch.float32).view(self.n_agents, 1, -1)
        )
        return static_lane_feats, pos_mats, phase_ids

    def init_hidden(self):
        hidden_states = []
        for agent in self.agents:
            hidden_states.append(agent.init_hidden())
        return hidden_states

    def _get_mean_action(self, obs, avail_actions, hidden_states):
        inputs = self._prepare_inputs(obs)
        outputs = []
        next_hidden_states = []
        for agent, hidden_state in zip(self.agents, hidden_states):
            agent_out, hidden_state = agent(*inputs, hidden_state)
            outputs.append(agent_out)
            next_hidden_states.append(hidden_state)
        logits = torch.mean(torch.stack(outputs), dim=0)
        avail_actions = avail_actions.squeeze(0)
        logits[avail_actions == 0] = -1e10
        logits = F.softmax(logits, dim=-1)
        pi = Categorical(logits)
        action = pi.sample().long()
        return action, next_hidden_states

    def _get_max_vote_action(self, obs, avail_actions, hidden_states):
        inputs = self._prepare_inputs(obs)
        votes = []
        next_hidden_states = []
        for agent, hidden_state in zip(self.agents, hidden_states):
            logits, hidden_state = agent(*inputs, hidden_state)
            avail_actions = avail_actions.squeeze(0)
            logits[avail_actions == 0] = -1e10
            logits = F.softmax(logits, dim=-1)
            pi = Categorical(logits)
            action = pi.sample().long()
            votes.append(action)
            next_hidden_states.append(hidden_state)
        votes = torch.stack(votes, dim=0)
        action = torch.mode(votes, dim=0).values.long()
        return action, next_hidden_states

    def select_action(self, obs, hidden_states, avail_actions):
        if self.action_selection_strategy == "mean":
            return self._get_mean_action(obs, avail_actions, hidden_states)
        elif self.action_selection_strategy == "max_vote":
            return self._get_max_vote_action(obs, avail_actions, hidden_states)
        else:
            raise ValueError(
                "Unknown action selection"
                + f"strategy: {self.action_selection_strategy}"
            )


class RLController:
    graph = False
    n_actions = 2
    hidden_dim = 64
    model_paths = ["pytsc/controllers/agent.th"]

    def __init__(self, tsc_env, **kwargs):
        self.tsc_env = tsc_env
        self.n_actions = tsc_env.get_action_size()
        self.n_agents = tsc_env.n_agents
        self.max_n_lanes = tsc_env.observation_space.max_n_controlled_lanes
        adjacency_matrix = tsc_env.parsed_network.adjacency_matrix
        self.agent = TSCAgentEnsemble(
            self.n_agents,
            self.n_actions,
            self.hidden_dim,
            self.max_n_lanes,
            n_models=len(self.model_paths),
            adjacency_matrix=adjacency_matrix,
            static_lane_feats=tsc_env.observation_space.get_per_agent_lane_features(),
            graph=self.graph,
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
    model_paths = ["pytsc/controllers/single_generalized_agent_1.th"]


class MultiGeneralizedAgentRLController(RLController):
    model_paths = [
        # "pytsc/controllers/single_generalized_agent_1.th",
        "pytsc/controllers/multi_generalized_agent_1.th",
        # "pytsc/controllers/multi_generalized_agent_2.th",
        # "pytsc/controllers/multi_generalized_agent_3.th",
    ]


class MultiGeneralizedGraphAgentRLController(RLController):
    graph = True
    model_paths = [
        # "pytsc/controllers/single_generalized_agent_1.th",
        "pytsc/controllers/multi_generalized_graph_agent_1.th",
        # "pytsc/controllers/multi_generalized_graph_agent_2.th",
        # "pytsc/controllers/multi_generalized_graph_agent_3.th",
    ]


class MultiGeneralizedVarRobustAgentRLController(RLController):
    model_paths = [
        # "pytsc/controllers/single_generalized_agent_1.th",
        "pytsc/controllers/multi_generalized_var_robust_agent_1.th",
        # "pytsc/controllers/multi_generalized_agent_2.th",
        # "pytsc/controllers/multi_generalized_agent_3.th",
    ]


class MultiGeneralizedVarRobustGraphAgentRLController(RLController):
    graph = True
    model_paths = [
        # "pytsc/controllers/single_generalized_agent_1.th",
        "pytsc/controllers/multi_generalized_var_robust_graph_agent_1.th",
        # "pytsc/controllers/multi_generalized_graph_agent_2.th",
        # "pytsc/controllers/multi_generalized_graph_agent_3.th",
    ]


class SpecializedMARLController(RLController):
    model_paths = ["pytsc/controllers/specialized_marl_agent_1.th"]


class MixedRLController(RLController):
    model_paths = [
        "pytsc/controllers/single_generalized_agent_1.th"
        "pytsc/controllers/multi_generalized_agent_1.th",
    ]

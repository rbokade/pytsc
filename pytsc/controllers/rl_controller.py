from pytsc.controllers.controllers import BaseController

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RNNAgent(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions):
        super(RNNAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        out = self.fc2(h)
        return out

    def select_action(self, agent_inputs, hidden_state, mask):
        out = self.forward(agent_inputs, hidden_state)
        out[mask == 0] = -1e10
        soft_out = F.softmax(out, dim=1)
        m = Categorical(soft_out)
        picked_actions = m.sample().long()
        return picked_actions


class RLController(BaseController):
    input_shape = 86
    hidden_dim = 128
    n_actions = 2

    def __init__(self, traffic_signal, model_path="pytsc/controllers/agent.th"):
        super(RLController, self).__init__(traffic_signal)
        self.controller = traffic_signal.controller
        self.agent = RNNAgent(self.input_shape, self.hidden_dim, self.n_actions)
        self._load_model(model_path=model_path)
        self.hidden_state = self.agent.init_hidden()
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
        if self.prev_action is None:
            prev_action = torch.tensor([1.0, 0.0]).unsqueeze(0)
        else:
            prev_action = self.prev_action
        obs = torch.tensor(obs).float().unsqueeze(0)
        inputs = torch.cat((obs, prev_action), dim=1)
        action = self.agent.select_action(inputs, self.hidden_state, act_mask)
        action_onehot = F.one_hot(action, self.n_actions).float()
        self.prev_action = action_onehot
        return action.item()

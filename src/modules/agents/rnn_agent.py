# code adapted from https://github.com/wendelinboehmer/dcg

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.use_action_repr = args.use_action_repr

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        if self.use_action_repr:
            self.fc2 = nn.Linear(args.hidden_dim, args.action_latent_dim)
        else:
            self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, action_repr=None):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        if self.use_action_repr:
            key = self.fc2(h).unsqueeze(-1)  # [bs*n, al, 1]
            action_repr_reshaped = action_repr.unsqueeze(0).repeat(key.shape[0], 1, 1)
            q = th.bmm(action_repr_reshaped, key).squeeze()
        else:
            q = self.fc2(h)
        return q, h

class CGRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CGRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        return h

class PairRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(PairRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.pair_rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.pair_rnn_hidden_dim, args.pair_rnn_hidden_dim)
        self.fc2 = nn.Linear(args.pair_rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.pair_rnn_hidden_dim).zero_()

    def h_forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.pair_rnn_hidden_dim)
        h = self.rnn(x, h_in)
        return h

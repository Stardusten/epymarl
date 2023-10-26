from typing import Tuple
from torch.distributions import Categorical

from components.epsilon_schedules import DecayThenFlatSchedule
from utils.dcop import dcop_solve_batch, make_chordal_cg

from .basic_controller import BasicMAC
from modules.agents import REGISTRY as agent_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
import torch as th
import torch.nn as nn
import numpy as np
import itertools
import copy


# This multi-agent controller shares parameters between agents
class ScpcgMAC(BasicMAC):

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        self.input_shape = self.args.rnn_hidden_dim

        # action representation
        self.use_action_repr = args.use_action_repr
        if self.use_action_repr:
            self.action_encoder = action_encoder_REGISTRY[args.action_encoder](args)
            self.action_repr = th.ones(self.n_actions, self.args.action_latent_dim).to(args.device)
            input_i = self.action_repr.unsqueeze(1).repeat(1, self.n_actions, 1)
            input_j = self.action_repr.unsqueeze(0).repeat(self.n_actions, 1, 1)
            self.p_action_repr = th.cat([input_i, input_j], dim=-1).view(self.n_actions * self.n_actions,
                                                                         -1).t().unsqueeze(0)

        # single q
        self.single_q = self._mlp(self.input_shape, args.single_q_hidden_dim, self.n_actions)

        # pairwise q
        if self.use_action_repr:
            self.pairwise_q = self._mlp(2 * self.input_shape, args.pairwise_q_hidden_dim,
                                        2 * self.args.action_latent_dim)
        else:
            self.pairwise_q = self._mlp(2 * self.input_shape, args.pairwise_q_hidden_dim, self.n_actions ** 2)

        # privileged bias
        self.privileged_bias = args.privileged_bias
        if self.privileged_bias:
            self.state_value = self._mlp(int(np.prod(args.state_shape)), [args.state_embed_dim], 1)

        # constraint network
        self.cg = make_chordal_cg(self.n_agents, device=self.args.device)

        # e-greedy scheduler
        self.schedule = DecayThenFlatSchedule(
            args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False) -> th.Tensor:
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        f, g, best_actions = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # exploration vs. exploitation
        self.epsilon = self.schedule.eval(t_env)  # update epsilon
        random_actions = Categorical(avail_actions.float()).sample().long()
        random_numbers = th.rand_like(f[:, :, 0])
        chosen_random = (random_numbers < self.epsilon).long()
        chosen_actions = chosen_random * random_actions + (1 - chosen_random) * best_actions
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        self.hidden_states = (self.agent(agent_inputs, self.hidden_states)
                              .view(-1, self.n_agents, self.args.rnn_hidden_dim))

        g_input = self.hidden_states
        f = self.single_q(g_input.view(-1, self.input_shape))
        f = f.view(-1, self.n_agents, self.n_actions)

        gi_input = g_input.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        gj_input = g_input.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
        g_input = th.cat([gi_input, gj_input], dim=-1).view(-1, 2 * self.input_shape)
        if self.use_action_repr:
            key = self.pairwise_q(g_input).view(-1, self.n_agents * self.n_agents, 2 * self.args.action_latent_dim)
            g = th.bmm(key, self.p_action_repr.repeat(f.shape[0], 1, 1)) / self.args.action_latent_dim / 2
        else:
            g = self.pairwise_q(g_input)
        g = g.view(-1, self.n_agents, self.n_agents, self.n_actions, self.n_actions)
        g = (g + g.permute(0, 2, 1, 4, 3)) / 2.

        if self.args.privileged_bias and test_mode == False:
            f[:, 0, :] += self.state_value(ep_batch['state'][:, t])

        masked_f, masked_g = self._mask_f_g(f, g, avail_actions)

        bs, n_agents, n_actions = f.shape

        best_actions = dcop_solve_batch(bs, n_agents, n_actions, masked_f.clone().detach().cpu(), masked_g.clone().detach().cpu(), *self.cg)\
            .detach().to(th.int64).to(self.args.device)

        return masked_f, masked_g, best_actions

    def _mask_f_g(self, f, g, avail_actions):
        """Masked out unavailable actions in f and g (set their Q-values to -9999999)"""
        n_agents, n_actions = f.shape[1], f.shape[2]
        if not th.is_tensor(avail_actions):
            avail_actions = th.tensor(avail_actions)
        f[avail_actions == 0] = -9999999
        g[avail_actions.unsqueeze(1).unsqueeze(-2).repeat(1, n_agents, 1, n_actions, 1) == 0] = -9999999
        g[avail_actions.unsqueeze(2).unsqueeze(-1).repeat(1, 1, n_agents, 1, n_actions) == 0] = -9999999
        return f, g

    def _select_f_g(self, f, g, actions):
        """Select f and g corresponding to given actions"""
        n_agents, n_actions = f.shape[1], f.shape[2]
        if len(actions.shape) == 2:  # dirty
            actions = actions.unsqueeze(-1)
        f = th.gather(f, dim=-1, index=actions).squeeze(-1)
        g = (th.gather(g, dim=-1, index=actions.unsqueeze(1).unsqueeze(-2).repeat(1, n_agents, 1, n_actions, 1))
             .squeeze(-1))
        g = th.gather(g, dim=-1, index=actions.unsqueeze(2).repeat(1, 1, n_agents, 1)).squeeze(-1)
        return f, g

    def q_values(self, f, g, actions):
        """Compute Q-values for given utilities f, payoffs g, coordination graphs and actions"""
        f, g = self._select_f_g(f, g, actions)
        if self.args.individual_q:
            values = f.sum(dim=-1) + (g * self.cg[0]).sum(dim=-1).sum(dim=-1) / 2  # /2 since each edge is computed twice
        else:
            isolated_nodes = th.max(1 - self.cg[0].sum(dim=-1), th.zeros_like(self.cg[0].sum(dim=-1)))
            values = (f * isolated_nodes).sum(dim=-1) + (g * self.cg[0]).sum(dim=-1).sum(dim=-1) / 2  # /2 since each edge is computed twice
        return values

    def update_action_repr(self):
        action_repr = self.action_encoder()

        self.action_repr = action_repr.detach().clone()

        # Pairwise Q (|A|, al) -> (|A|, |A|, 2*al)
        input_i = self.action_repr.unsqueeze(1).repeat(1, self.n_actions, 1)
        input_j = self.action_repr.unsqueeze(0).repeat(self.n_actions, 1, 1)
        self.p_action_repr = (th.cat([input_i, input_j], dim=-1)
            .view(self.n_actions * self.n_actions, -1).t().unsqueeze(0))

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return itertools.chain(self.agent.parameters(), self.single_q.parameters(), self.pairwise_q.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.single_q.load_state_dict(other_mac.single_q.state_dict())
        self.pairwise_q.load_state_dict(other_mac.pairwise_q.state_dict())
        if self.args.use_action_repr:
            self.action_repr = copy.deepcopy(other_mac.action_repr)
            self.p_action_repr = copy.deepcopy(other_mac.p_action_repr)

    def cuda(self):
        self.agent.cuda()
        self.single_q.cuda()
        self.pairwise_q.cuda()
        if self.privileged_bias:
            self.state_value.cuda()
        if self.use_action_repr:
            self.action_encoder.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.single_q.state_dict(), "{}/single_q.th".format(path))
        th.save(self.pairwise_q.state_dict(), "{}/pairwise_q.th".format(path))
        if self.args.privileged_bias:
            th.save(self.state_value.state_dict(), "{}/agent.th".format(path))
        if self.args.use_action_repr:
            th.save(self.action_repr, "{}/action_repr.pt".format(path))
            th.save(self.p_action_repr, "{}/p_action_repr.pt".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.single_q.load_state_dict(th.load("{}/single_q.th".format(path), map_location=lambda storage, loc: storage))
        self.pairwise_q.load_state_dict(
            th.load("{}/pairwise_q.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.privileged_bias:
            self.state_value.load_state_dict(
                th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.use_action_repr:
            self.action_repr = th.load("{}/action_repr.pt".format(path),
                                       map_location=lambda storage, loc: storage).to(self.args.device)
            self.p_action_repr = th.load("{}/p_action_repr.pt".format(path),
                                         map_location=lambda storage, loc: storage).to(self.args.device)

    def action_encoder_params(self):
        return list(self.action_encoder.parameters())

    def action_repr_forward(self, ep_batch, t):
        return self.action_encoder.predict(ep_batch["obs"][:, t], ep_batch["actions_onehot"][:, t])

    # ========================= Private methods =========================

    @staticmethod
    def _mlp(input, hidden_dims, output):
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d
        layers.append(nn.Linear(dim, output))
        return nn.Sequential(*layers)

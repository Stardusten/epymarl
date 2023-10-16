from __future__ import annotations

import time
from typing import List, Optional, Tuple, Union
import torch as th
from array import array


class Util:

    # agent name must from small to large
    def __init__(self, agent_names: List[int], num_actions: int, body: Optional[th.Tensor] = None):
        self.agent_names = agent_names
        self.agent_names.sort()
        self.num_actions = num_actions
        self.max_idx = None
        if body is None:
            self.body = th.zeros((num_actions,) * len(agent_names))
        else:
            self.body = body

    def clone(self) -> Util:
        return Util([*self.agent_names], self.num_actions, self.body.clone())

    def get_value(self, joint_action: Tuple[int, ...]) -> float:
        return self.body[joint_action].item()

    def set_value(self, joint_action: Tuple[int, ...], value: float):
        self.body[joint_action] = value

    @staticmethod
    def select(agent_names: List[int], joint_action: Tuple[int, ...], select_names: List[int]) -> th.Tensor:
        result = []
        for name in select_names:
            idx = agent_names.index(name)
            if idx != -1:
                result.append(joint_action[idx])
        return th.tensor(result)

    @staticmethod
    def join(u1: Optional[Util], u2: Optional[Util]) -> Util:
        if u1 is None:
            return u2.clone()
        if u2 is None:
            return u1.clone()
        # assume q1.num_actions = q2.num_actions
        agent_names = list({*u1.agent_names, *u2.agent_names})  # remove duplicates
        agent_names.sort()
        num_actions = u1.num_actions
        u1_body = u1.body.clone()
        u2_body = u2.body.clone()
        for i in range(len(agent_names)):
            name = agent_names[i]
            if name not in u1.agent_names:
                s = u1_body.shape
                new_s = (*s[:i], num_actions, *s[i:])
                u1_body = u1_body.unsqueeze(i).expand(new_s)
            if name not in u2.agent_names:
                s = u2_body.shape
                new_s = (*s[:i], num_actions, *s[i:])
                u2_body = u2_body.unsqueeze(i).expand(new_s)
        return Util(agent_names, num_actions, u1_body + u2_body)

    def collapse_(self, agent_name: int) -> Util:
        idx = self.agent_names.index(agent_name)
        if idx != -1:
            self.agent_names.pop(idx)
            self.body, self.max_idx = self.body.max(dim=idx)
            return self


class DcopSolver:

    def __init__(self, num_agents: int, num_actions: int, pq: th.Tensor, sq: th.Tensor, graph: th.Tensor):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.sq = sq  # single q
        self.pq = pq  # pair q
        self.G = graph
        self.P = [-1] * num_agents
        self.C = [[] for _ in range(num_agents)]
        self.PP = [[] for _ in range(num_agents)]
        self.PC = [[] for _ in range(num_agents)]
        self._stat = [0] * num_agents
        # self.leaves = []
        self.waited_children = [[] for _ in range(num_agents)]
        self.utils = [self.get_single_util(u) for u in range(num_agents)]  # init with single utils
        self.opt_joint_action = [-1] * num_agents

        # build pseudo-tree
        for u in range(self.num_agents):
            if self._stat[u] != 0:
                continue
            self.dfs(u)

    def has_edge(self, u, v) -> bool:
        if u > v:
            return self.G[u, v].item() > 0.5
        else:
            return self.G[v, u].item() > 0.5

    def get_pair_q(self, u, v, i, j) -> float:
        if u > v:
            return self.pq[u, v, i, j].item()
        else:
            return self.pq[v, u, j, i].item()

    def get_single_q(self, u, i) -> float:
        return self.sq[u, i].item()

    def dfs(self, u: int):
        # is_leaf = True
        self._stat[u] = 1
        for v in range(self.num_agents):
            if not self.has_edge(u, v):
                continue
            if self._stat[v] == 0:  # tree edge found
                is_leaf = False
                self.P[v] = u
                self.C[u].append(v)
                self.dfs(v)
            elif self._stat[v] == 2:  # back edge found
                self.PP[v].append(u)
                self.PC[u].append(v)
        self._stat[u] = 2
        # if is_leaf:
        #     self.leaves.append(u)

    def get_edge_util(self, u: int, v: int) -> Util:
        if u > v:
            u, v = v, u
        util = Util([u, v], self.num_actions)
        for i in range(self.num_actions):
            for j in range(self.num_actions):
                q = self.get_pair_q(u, v, i, j)
                util.set_value((i, j), q)
        return util

    def get_single_util(self, u: int) -> Util:
        util = Util([u], self.num_actions)
        for i in range(self.num_actions):
            q = self.get_single_q(u, i)
            util.set_value((i,), q)
        return util

    def get_util_msg(self, u: int) -> Util:
        deps = [self.P[u], *self.PP[u]]
        joint_util = self.utils[u]
        for dep in deps:
            edge_util = self.get_edge_util(u, dep)
            joint_util = Util.join(joint_util, edge_util)
        joint_util.collapse_(u)
        self.utils[u] = joint_util
        return joint_util

    def solve(self):
        for u in range(self.num_agents):
            self.waited_children[u] += self.C[u]
        util_sent = [False] * self.num_agents
        fin = False
        while not fin:
            fin = True
            for u in range(self.num_agents):
                if len(self.waited_children[u]) == 0 and not util_sent[u]:
                    p = self.P[u]
                    if p == -1:  # I can determine my optimal action now!
                        util = self.utils[u].collapse_(u)
                        opt_action = util.max_idx.item()
                        self.opt_joint_action[u] = opt_action
                        # print(f'{u}\'s opt action is {opt_action}')
                    else:  # send util msg to u's parent
                        fin = False
                        util = self.get_util_msg(u)
                        # print(f'{u} send util to {p}: {util.body}')
                        self.utils[p] = Util.join(self.utils[p], util)
                        self.waited_children[p].remove(u)
                        util_sent[u] = True

        fin = False
        while not fin:
            fin = True
            for u in range(self.num_agents):
                if self.opt_joint_action[u] != -1:
                    for v in self.C[u]:
                        if self.opt_joint_action[v] == -1:
                            fin = False
                            util = self.utils[v]
                            dep_actions = [self.opt_joint_action[i] for i in util.agent_names]
                            opt_action = util.max_idx[tuple(dep_actions)].item()
                            self.opt_joint_action[v] = opt_action


def dcop_batch_solve(f, g, graph, bs, n, m):
    best_actions = th.zeros((bs, n))
    for i in range(bs):
        solver = DcopSolver(num_agents=n, num_actions=m, sq=f[i], pq=g[i], graph=graph[i])
        solver.solve()
        best_actions[i] = th.tensor(solver.opt_joint_action)
    return best_actions

def test():
    # sq = th.zeros((7, 2))
    # sq[0, 0] = 3
    # sq[0, 1] = 3
    # sq[1, 0] = 2
    # sq[1, 1] = 3
    # sq[2, 0] = 1
    # sq[2, 1] = 5
    # sq[3, 0] = 4
    # sq[3, 1] = 5
    # sq[4, 0] = 3
    # sq[4, 1] = 4
    # sq[5, 0] = 6
    # sq[5, 1] = 3
    # sq[6, 0] = 4
    # sq[6, 1] = 2
    # pq = th.zeros((7, 7, 2, 2))
    # pq[1, 0, 0, 0] = 4
    # pq[1, 0, 0, 1] = 1
    # pq[1, 0, 1, 0] = 2
    # pq[1, 0, 1, 1] = 2
    # pq[2, 0, 0, 0] = 2
    # pq[2, 0, 0, 1] = 4
    # pq[2, 0, 1, 0] = 1
    # pq[2, 0, 1, 1] = 3
    # pq[6, 0, 0, 0] = 7
    # pq[6, 0, 0, 1] = 4
    # pq[6, 0, 1, 0] = 2
    # pq[6, 0, 1, 1] = 2
    # pq[2, 1, 0, 0] = 2
    # pq[2, 1, 0, 1] = 2
    # pq[2, 1, 1, 0] = 3
    # pq[2, 1, 1, 1] = 3
    # pq[6, 2, 0, 0] = 3
    # pq[6, 2, 0, 1] = 2
    # pq[6, 2, 1, 0] = 2
    # pq[6, 2, 1, 1] = 3
    # pq[3, 1, 0, 0] = 4
    # pq[3, 1, 0, 1] = 2
    # pq[3, 1, 1, 0] = 3
    # pq[3, 1, 1, 1] = 1
    # pq[4, 1, 0, 0] = 2
    # pq[4, 1, 0, 1] = 4
    # pq[4, 1, 1, 0] = 3
    # pq[4, 1, 1, 1] = 1
    # pq[5, 4, 0, 0] = 5
    # pq[5, 4, 0, 1] = 2
    # pq[5, 4, 1, 0] = 1
    # pq[5, 4, 1, 1] = 1
    # pq[5, 2, 0, 0] = 6
    # pq[5, 2, 0, 1] = 4
    # pq[5, 2, 1, 0] = 2
    # pq[5, 2, 1, 1] = 3
    # solver = DcopSolver(
    #     7,
    #     2,
    #     pq,
    #     sq,
    #     th.tensor([
    #         [1, 1, 1, 0, 0, 0, 1],
    #         [1, 1, 1, 1, 1, 0, 0],
    #         [1, 1, 1, 0, 0, 1, 1],
    #         [0, 1, 0, 1, 0, 0, 0],
    #         [0, 1, 0, 0, 1, 0, 0],
    #         [0, 0, 1, 0, 1, 1, 0],
    #         [1, 0, 1, 0, 0, 0, 1]
    #     ]))
    n_agents = 10
    n_actions = 5
    sq = th.zeros((n_agents, n_actions))
    pq = th.zeros((n_agents, n_agents, n_actions, n_actions))
    graph = th.tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
         [1., 0., 1., 1., 1., 0., 0., 0., 0., 1.],
         [0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    solver = DcopSolver(
            n_agents,
            n_actions,
            pq,
            sq,
            graph)
    solver.solve()
    print(solver.opt_joint_action)


if __name__ == '__main__':
    # start = time.time()
    # for i in range(10000):
        test()
    # end = time.time()
    # print(f'time: {(end - start) / 10000}')

from __future__ import annotations
import torch as th
import dcop_cpp


def dcop_solve_batch(bs, n_agents, n_actions, f, g, G, P, C, PP, PC):
    return dcop_cpp.dcop_solve_batch(bs, n_agents, n_actions, f, g, G, P, C, PP, PC)  # delegate

def make_chordal_cg(n_agents: int, device=None):
    G = th.zeros((n_agents, n_agents))
    P = [-1 for _ in range(n_agents)]
    C = [[] for _ in range(n_agents)]
    PP = [[] for _ in range(n_agents)]
    PC = [[] for _ in range(n_agents)]
    for i in range(1, n_agents):
        P[i] = i - 1
        C[i - 1].append(i)
        G[i, i - 1] = 1
        G[i - 1, i] = 1
    for i in range(n_agents - 1, 1, -1):
        PP[i].append(0)
        PC[0].append(i)
        G[i, 0] = 1
        G[0, i] = 1
    if device:
        G = G.to(device)
    return (G, P, C, PP, PC, )

# from setuptools import setup
# from torch.utils import cpp_extension
#
# setup(name='dcop_cpp',
#       ext_modules=[cpp_extension.CppExtension('dcop_cpp', ['dcop.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})
#include <torch/extension.h>
#include <iostream>
#include <vector>

#define MAX_N_AGENTS 50
#define MAX_N_ACTIONS 50
#define MAX_BATCH_SIZE 35

#define vcontains(v, e) (find((v).begin(), (v).end(), e) != (v).end())
#define vconcat(v1, v2) (v1.insert(v1.end(), v2.begin(), v2.end()))
#define vsort(v) (sort(v.begin(), v.end()))
#define vremove_duplicated(v)                         \
    do {                                              \
        sort(v.begin(), v.end());                     \
        v.erase(unique(v.begin(), v.end()), v.end()); \
    } while (0)
#define vremove_ith(v, i) (v.erase(v.begin() + i))
#define vappend(v, e) (v.push_back(e))
#define vremove(vec, elem)                                 \
    do {                                                   \
        auto it = std::find(vec.begin(), vec.end(), elem); \
        if (it != vec.end())                               \
            vec.erase(it);                                 \
    } while (0)

using namespace std;

torch::Tensor unsqueeze_expand(torch::Tensor t, int dim, int size) {
    vector<long> sizes = vector<long>(t.sizes().vec());
    sizes.insert(sizes.begin() + dim, size);
    return t.unsqueeze(dim).expand(sizes);
}

struct Util {
    vector<int> agent_names;
    int n_actions;
    optional<torch::Tensor> max_idx;
    torch::Tensor body;

    Util clone() {
        Util ret;
        ret.agent_names = vector<int>(this->agent_names);
        ret.n_actions = this->n_actions;
        if (this->max_idx.has_value())
            ret.max_idx = this->max_idx.value().clone();
        else ret.max_idx = nullopt;
        ret.body = this->body.clone();
        return ret;
    }

    void collapse_(int agent_name) {
        int idx = -1;
        for (int i = 0; i < this->agent_names.size(); i++)
            if (this->agent_names[i] == agent_name) {
                idx = i;
                break;
            }
        if (idx != -1) {
            vremove_ith(this->agent_names, idx);
            auto _t = this->body.max(idx);
            this->body = get<0>(_t);
            this->max_idx = make_optional(get<1>(_t));
        }
    }
};

Util make_util(vector<int> agent_names, int n_actions, torch::Tensor body) {
    Util util;
    util.agent_names = agent_names;
    util.n_actions = n_actions;
    util.body = body;
    return util;
}

static Util join(Util& u1, Util& u2) {
    // merge agent names
    Util ret;
    vconcat(ret.agent_names, u1.agent_names);
    vconcat(ret.agent_names, u2.agent_names);
    vremove_duplicated(ret.agent_names);
    vsort(ret.agent_names);
    ret.n_actions = u1.n_actions;
    torch::Tensor u1_body = u1.body.clone();
    torch::Tensor u2_body = u2.body.clone();
    for (int i = 0; i < ret.agent_names.size(); i++) {
        int name = ret.agent_names[i];
        if (!vcontains(u1.agent_names, name))
            u1_body = unsqueeze_expand(u1_body, i, ret.n_actions);
        if (!vcontains(u2.agent_names, name))
            u2_body = unsqueeze_expand(u2_body, i, ret.n_actions);
    }
    ret.body = u1_body + u2_body;
    return ret;
}

struct DcopSolver {
    int n_agents;
    int n_actions;
    torch::Tensor sq;
    torch::Tensor pq;
    vector<int> P;
    vector<vector<int> > C;
    vector<vector<int> > PP;
    vector<vector<int> > PC;
    vector<int> waited_children[MAX_N_AGENTS];
    vector<Util> utils;
    vector<int> opt_joint_action;

    Util get_edge_util(int u, int v) {
        if (u > v) swap(u, v);
        torch::Tensor body = this->pq[u][v].clone();
        Util util = make_util({ u, v }, this->n_actions, body);
        return util;
    }

    Util get_single_util(int u) {
        torch::Tensor body = this->sq[u].clone();
        Util util = make_util({ u }, this->n_actions, body);
        return util;
    }

    Util& get_util_msg(int u) {
        vector<int> deps;
        vappend(deps, this->P[u]);
        vconcat(deps, this->PP[u]);
        Util& util_u = this->utils[u];
        for (int i = 0; i < deps.size(); i++) {
            Util edge_util = this->get_edge_util(u, deps[i]);
            util_u = join(edge_util, util_u);
        }
        util_u.collapse_(u);
        return util_u;
    }

    void solve() {
        vector<bool> util_sent;
        for (int u = 0; u < this->n_agents; u++) {
            vconcat(this->waited_children[u], this->C[u]);
            vappend(this->opt_joint_action, -1);
            vappend(this->utils, this->get_single_util(u));
            vappend(util_sent, false);
        }
        bool fin = false;
        while (!fin) {
            fin = true;
            for (int u = 0; u < this->n_agents; u++) {
                if (this->waited_children[u].size() == 0 && !util_sent[u]) {
                    int p = this->P[u];
                    if (p == -1) { // I can determine my optimal action now!
                        Util& util = this->utils[u];
                        util.collapse_(u);
                        int opt_action = util.max_idx.value().item().toInt();
                        this->opt_joint_action[u] = opt_action;
                    } else { // send util msg to u's parent
                        fin = false;
                        Util& util_u = this->get_util_msg(u);
                        Util& util_p = this->utils[p];
                        util_p = join(util_p, util_u);
                        vremove(this->waited_children[p], u);
                        util_sent[u] = true;
                    }
                }
            }
        }

        fin = false;
        while (!fin) {
            fin = true;
            for (int u = 0; u < this->n_agents; u++) {
                if (this->opt_joint_action[u] != -1) {
                    const vector<int>& children = this->C[u];
                    for (const int v : children) {
                        if (this->opt_joint_action[v] == -1) {
                            fin = false;
                            Util& util_v = this->utils[v];
                            long n_depth = util_v.agent_names.size();
                            torch::Tensor& _t = util_v.max_idx.value();
                            for (const int i : util_v.agent_names)
                                _t = _t.index({ this->opt_joint_action[i] });
                            int opt_action_v = _t.item().toInt();
                            this->opt_joint_action[v] = opt_action_v;
                        }
                    }
                }
            }
        }
    }
};

torch::Tensor dcop_solve_batch(
    int bs,
    int n_agents,
    int n_actions,
    torch::Tensor f,
    torch::Tensor g,
    torch::Tensor G,
    vector<int> P,
    vector<vector<int> > C,
    vector<vector<int> > PP,
    vector<vector<int> > PC
) {
    DcopSolver dcop_solvers[bs];
    vector<int> result;
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < bs; i++) {
        DcopSolver& dcop_solver = dcop_solvers[i];
        dcop_solver.n_agents = n_agents;
        dcop_solver.n_actions = n_actions;
        dcop_solver.sq = f[i];
        dcop_solver.pq = g[i];
        dcop_solver.P = P;
        dcop_solver.C = C;
        dcop_solver.PP = PP;
        dcop_solver.PC = PC;
        dcop_solver.solve();
        vconcat(result, dcop_solver.opt_joint_action);
    }
    return torch::from_blob(
        result.data(),
        { bs, n_agents },
        torch::TensorOptions().dtype(torch::kInt32)
    ).clone();
}

torch::Tensor dcop_solve(
    int n_agents,
    int n_actions,
    torch::Tensor f,
    torch::Tensor g,
    torch::Tensor G,
    vector<int> P,
    vector<vector<int> > C,
    vector<vector<int> > PP,
    vector<vector<int> > PC
) {
    DcopSolver dcop_solver;
    dcop_solver.n_agents = n_agents;
    dcop_solver.n_actions = n_actions;
    dcop_solver.sq = f;
    dcop_solver.pq = g;
    dcop_solver.P = P;
    dcop_solver.C = C;
    dcop_solver.PP = PP;
    dcop_solver.PC = PC;
    dcop_solver.solve();
    return torch::from_blob(
        dcop_solver.opt_joint_action.data(),
        { n_agents },
        torch::TensorOptions().dtype(torch::kInt32)
    ).clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unsqueeze_expand", &unsqueeze_expand);
    py::class_<Util>(m, "Util")
        .def_readwrite("agent_names", &Util::agent_names)
        .def_readwrite("n_actions", &Util::n_actions)
        .def_readwrite("max_idx", &Util::max_idx)
        .def_readwrite("body", &Util::body)
        .def("clone", &Util::clone)
        .def("collapse_", &Util::collapse_);
    m.def("make_util", &make_util);
    m.def("join", &join);
    m.def("dcop_solve", &dcop_solve);
    m.def("dcop_solve_batch", &dcop_solve_batch);
}
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import remove_self_loops, to_undirected

from .base_explainer import ExplainerBase


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n + 1))
        self.size = [0] * (n + 1)
        self.made = [False] * (n + 1)
        self.Q = [0.0] * (n + 1)

    def make_if_needed(self, v: int):
        if not self.made[v]:
            self.made[v] = True
            self.parent[v] = v
            self.size[v] = 1
            self.Q[v] = 0.0

    def find(self, v: int) -> int:
        if not self.made[v]:
            return v
        while self.parent[v] != v:
            self.parent[v] = self.parent[self.parent[v]]
            v = self.parent[v]
        return v

    def union(self, a: int, b: int) -> int:
        self.make_if_needed(a)
        self.make_if_needed(b)
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.Q[ra] += self.Q[rb]
        return ra


def _rmc_single_cluster_from_adj(adj1: List[List[int]], epsilon: float) -> Tuple[Set[int], float]:
    import heapq

    n = len(adj1) - 1
    deg0 = [0] * (n + 1)
    for i in range(1, n + 1):
        deg0[i] = len(adj1[i])
    pq = [(deg0[i], i) for i in range(1, n + 1)]
    heapq.heapify(pq)
    peel_stack: List[int] = []

    while pq:
        d, u = heapq.heappop(pq)
        if d != deg0[u]:
            continue
        peel_stack.append(u)
        for v in adj1[u]:
            if deg0[v] > 0:
                deg0[v] -= 1
                heapq.heappush(pq, (deg0[v], v))
        deg0[u] = 0

    add_order = [0] * n
    idx = [0] * (n + 1)
    for t in range(n):
        u = peel_stack.pop()
        add_order[t] = u
        idx[u] = t

    succ = [[] for _ in range(n + 1)]
    pred = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        for v in adj1[u]:
            if u < v:
                if idx[u] < idx[v]:
                    succ[u].append(v)
                    pred[v].append(u)
                else:
                    succ[v].append(u)
                    pred[u].append(v)
    for v in range(1, n + 1):
        if len(succ[v]) > 1:
            succ[v].sort(key=lambda w: idx[w])

    dsu = _DSU(n)
    deg = [0] * (n + 1)
    pred_sum = [0] * (n + 1)

    bestSL = 0.0
    bestComponent: Set[int] = set()

    def sum_succ_until(v: int, T: int) -> int:
        s = 0
        for w in succ[v]:
            if idx[w] >= T:
                break
            s += deg[w]
        return s

    def snapshot_component(root: int) -> Set[int]:
        comp = set()
        for i in range(1, n + 1):
            if dsu.made[i] and dsu.find(i) == root:
                comp.add(i)
        return comp

    for u in add_order:
        dsu.make_if_needed(u)

        ru = dsu.find(u)
        sL = dsu.size[ru] / (dsu.Q[ru] + epsilon)
        if sL > bestSL:
            bestSL = sL
            bestComponent = snapshot_component(ru)

        Su = 0
        Tu = idx[u]

        for v in pred[u]:
            a = deg[u]
            b = deg[v]

            Sv = pred_sum[v] + sum_succ_until(v, Tu)

            dQu = 2 * a * a - 2 * Su + a
            dQv = 2 * b * b - 2 * Sv + b
            edgeTerm = (a - b) * (a - b)

            ru = dsu.find(u)
            rv = dsu.find(v)

            dsu.Q[ru] += float(dQu)
            dsu.Q[rv] += float(dQv)

            if ru != rv:
                r = dsu.union(ru, rv)
                dsu.Q[r] += float(edgeTerm)
            else:
                r = ru
                dsu.Q[r] += float(edgeTerm)

            sL = dsu.size[r] / (dsu.Q[r] + epsilon)
            if sL > bestSL:
                bestSL = sL
                bestComponent = snapshot_component(r)

            deg[u] += 1
            deg[v] += 1

            for y in succ[u]:
                pred_sum[y] += 1
            for y in succ[v]:
                pred_sum[y] += 1

            Su += deg[v]

    return bestComponent, float(bestSL)


def _generate_lacore_cluster(edges: List[Tuple[int, int]], epsilon: float) -> Dict:
    try:
        eps = float(epsilon)
    except Exception:
        eps = float(str(epsilon))

    if not edges:
        return {"seed_nodes": [], "score": 0.0}

    max_node = -1
    for u, v in edges:
        if u == v:
            continue
        if u > max_node:
            max_node = u
        if v > max_node:
            max_node = v
    n = max_node + 1

    adj1: List[List[int]] = [[] for _ in range(n + 1)]
    seen = set()
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        aa, bb = a + 1, b + 1
        adj1[aa].append(bb)
        adj1[bb].append(aa)

    best_comp_1b, bestSL = _rmc_single_cluster_from_adj(adj1, eps)
    seed_nodes_0b = sorted([u - 1 for u in best_comp_1b])
    return {"seed_nodes": seed_nodes_0b, "score": bestSL}


def _edge_list_from_edge_index(edge_index: Tensor, num_nodes: int) -> List[Tuple[int, int]]:
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edges = set()
    row, col = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(row, col):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    return list(edges)


class LaCoreExplainer(ExplainerBase):
    r"""Use a single LaCore cluster as an explainer mask for graph classification."""

    def __init__(self,
                 model,
                 epsilon: float = 0.1,
                 explain_graph: bool = True,
                 molecule: bool = False):
        super().__init__(model=model, explain_graph=explain_graph, molecule=molecule)
        self.epsilon = epsilon
        self.last_cluster_nodes: Optional[List[int]] = None
        self.last_cluster_score: Optional[float] = None

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs):
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()

        num_classes = kwargs.get('num_classes')
        if num_classes is None:
            raise ValueError("LaCoreExplainer requires num_classes in kwargs.")

        epsilon = kwargs.get('epsilon', self.epsilon)
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        edges = _edge_list_from_edge_index(edge_index, x.size(0))
        res = _generate_lacore_cluster(edges, epsilon=epsilon)
        cluster_nodes = res.get("seed_nodes", [])
        self.last_cluster_nodes = cluster_nodes
        self.last_cluster_score = res.get("score", 0.0)

        edge_mask = torch.zeros(edge_index.size(1), device=self.device)
        if cluster_nodes:
            in_cluster = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
            in_cluster[torch.tensor(cluster_nodes, device=self.device, dtype=torch.long)] = True
            edge_mask = (in_cluster[edge_index[0]] & in_cluster[edge_index[1]]).float()

        edge_masks = [edge_mask for _ in range(num_classes)]
        hard_edge_masks = edge_masks

        self.__set_masks__(x, edge_index)
        with torch.no_grad():
            related_preds = self.eval_related_pred(x, edge_index, edge_masks, **kwargs)
        self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds


__all__ = ["LaCoreExplainer"]

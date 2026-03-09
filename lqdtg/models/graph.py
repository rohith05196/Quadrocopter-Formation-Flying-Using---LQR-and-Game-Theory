"""Communication graph for multi-agent formation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class Graph:
    n_agents: int
    edges: List[Tuple[int, int]]
    adj: np.ndarray = field(init=False, repr=False)
    inc: np.ndarray = field(init=False, repr=False)
    lap: np.ndarray = field(init=False, repr=False)
    nbrs: Dict[int, List[int]] = field(init=False, repr=False)

    def __post_init__(self):
        N, M = self.n_agents, len(self.edges)
        self.adj = np.zeros((N, N))
        self.inc = np.zeros((N, M))
        for idx, (i, j) in enumerate(self.edges):
            self.adj[i, j] = self.adj[j, i] = 1.0
            self.inc[i, idx] = 1.0
            self.inc[j, idx] = -1.0
        self.lap = np.diag(self.adj.sum(axis=1)) - self.adj
        self.nbrs = {v: [] for v in range(N)}
        for i, j in self.edges:
            self.nbrs[i].append(j)
            self.nbrs[j].append(i)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def graph3() -> Graph:
    """Triangle: 0-1, 1-2, 0-2"""
    return Graph(3, [(0, 1), (1, 2), (0, 2)])


def graph4() -> Graph:
    """Square + diagonal: 0-1, 1-2, 2-3, 3-0, 0-2"""
    return Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

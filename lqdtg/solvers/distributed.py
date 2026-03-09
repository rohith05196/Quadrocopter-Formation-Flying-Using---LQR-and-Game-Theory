"""
Distributed LQDTG — edge-based 2-player Nash via coupled Riccati.

Each graph edge is an independent 2-player game on the relative state
z = x_i − x_j, enabling distributed computation.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.linalg import inv
from lqdtg.models.graph import Graph


def solve_edge_nash(Ad, Bd, Q, R, Qf, N_horizon):
    B1, B2 = Bd, -Bd
    P1, P2 = Qf.copy(), Qf.copy()
    K1s, K2s = [], []
    for _ in range(N_horizon - 1, -1, -1):
        S1 = R + B1.T @ P1 @ B1
        S2 = R + B2.T @ P2 @ B2
        K1 = inv(S1) @ B1.T @ P1 @ Ad
        K2 = inv(S2) @ B2.T @ P2 @ Ad
        K1s.insert(0, K1); K2s.insert(0, K2)
        Acl = Ad - B1 @ K1 - B2 @ K2
        P1 = Q + K1.T @ R @ K1 + Acl.T @ P1 @ Acl
        P2 = Q + K2.T @ R @ K2 + Acl.T @ P2 @ Acl
        P1 = 0.5 * (P1 + P1.T)
        P2 = 0.5 * (P2 + P2.T)
    return K1s, K2s


def solve_edges(graph, Ad, Bd, Q_edge, R_edge, Qf_edge, N_horizon):
    return [solve_edge_nash(Ad, Bd, Q_edge, R_edge, Qf_edge, N_horizon)
            for _ in graph.edges]

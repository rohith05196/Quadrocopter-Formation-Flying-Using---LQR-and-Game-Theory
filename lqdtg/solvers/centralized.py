"""Centralized LQDTG — augmented N-player Nash (benchmark)."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from numpy.linalg import inv
from scipy.linalg import block_diag
from lqdtg.models.graph import Graph


def solve_centralized(graph, Ad, Bd, Q, R, Qf, N_horizon):
    n, m, Na = Ad.shape[0], Bd.shape[1], graph.n_agents
    Aa = block_diag(*([Ad] * Na))
    Bs = []
    for i in range(Na):
        Bi = np.zeros((Na * n, m))
        Bi[i * n:(i + 1) * n, :] = Bd
        Bs.append(Bi)

    Qs, Qfs, Rs = [], [], []
    for i in range(Na):
        Qi = np.zeros((Na * n, Na * n))
        Qi[i*n:(i+1)*n, i*n:(i+1)*n] = Q
        Q_form = Q * 0.3
        for j in graph.nbrs[i]:
            Qi[i*n:(i+1)*n, i*n:(i+1)*n] += Q_form
            Qi[j*n:(j+1)*n, j*n:(j+1)*n] += Q_form
            Qi[i*n:(i+1)*n, j*n:(j+1)*n] -= Q_form
            Qi[j*n:(j+1)*n, i*n:(i+1)*n] -= Q_form
        Qs.append(0.5 * (Qi + Qi.T))
        Qfi = np.zeros((Na * n, Na * n))
        Qfi[i*n:(i+1)*n, i*n:(i+1)*n] = Qf
        Qfs.append(0.5 * (Qfi + Qfi.T))
        Rs.append(R.copy())

    P = [Qfs[i].copy() for i in range(Na)]
    Kg = [[] for _ in range(Na)]
    for _ in range(N_horizon - 1, -1, -1):
        Ks = [inv(Rs[i] + Bs[i].T @ P[i] @ Bs[i]) @ (Bs[i].T @ P[i] @ Aa)
              for i in range(Na)]
        for i in range(Na):
            Kg[i].insert(0, Ks[i])
        Acl = Aa.copy()
        for i in range(Na):
            Acl -= Bs[i] @ Ks[i]
        P = [0.5 * ((Qs[i] + Ks[i].T @ Rs[i] @ Ks[i] + Acl.T @ P[i] @ Acl) +
                     (Qs[i] + Ks[i].T @ Rs[i] @ Ks[i] + Acl.T @ P[i] @ Acl).T)
             for i in range(Na)]
    return Kg, Bs, Aa

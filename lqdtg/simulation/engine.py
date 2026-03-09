"""Closed-loop simulation of LQDTG formation control."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.linalg import norm

from lqdtg.config import CostConfig
from lqdtg.models.graph import Graph
from lqdtg.models.quadrotor import QuadParams, linearized_model, discretize
from lqdtg.solvers.centralized import solve_centralized
from lqdtg.solvers.distributed import solve_edges
from lqdtg.solvers.lqr import lqr_gain
from lqdtg.trajectories.generators import formation_offsets


@dataclass
class SimResult:
    states: np.ndarray      # (N, Na, 12)
    controls: np.ndarray    # (N, Na, 4)
    references: np.ndarray  # (N, Na, 12)
    time: np.ndarray        # (N,)
    form_err: np.ndarray    # (N, n_edges)
    track_err: np.ndarray   # (N, Na)


def simulate(
    graph: Graph, qp: QuadParams,
    traj_func: Callable[[int, float], np.ndarray],
    N_steps: int = 300, method: str = "distributed",
    seed: int = 42, cost: CostConfig | None = None,
    rh_window: int = 40, pos_spread: float = 3.0,
    formation_dist: float = 1.5, verbose: bool = True,
) -> SimResult:

    if cost is None:
        cost = CostConfig()
    dt, Na = qp.dt, graph.n_agents

    Ac, Bc = linearized_model(qp)
    Ad, Bd = discretize(Ac, Bc, dt)
    K_lqr = lqr_gain(Ad, Bd, cost.Q, cost.R)

    ref = traj_func(N_steps, dt)
    off = formation_offsets(Na, dist=formation_dist)
    aref = np.zeros((N_steps, Na, 12))
    for k in range(N_steps):
        for i in range(Na):
            aref[k, i] = ref[k] + off[i]

    rng = np.random.default_rng(seed)
    x = np.zeros((Na, 12))
    for i in range(Na):
        x[i, 0] = rng.uniform(-pos_spread, pos_spread)
        x[i, 1] = rng.uniform(-pos_spread, pos_spread)
        x[i, 2] = rng.uniform(0, pos_spread * 0.5)

    Nh = rh_window if method == "receding_horizon" else N_steps
    if verbose:
        print(f"  Solving {method} LQDTG (Nh={Nh}) ...")
    if method == "centralized":
        cK, cB, cA = solve_centralized(graph, Ad, Bd, cost.Q, cost.R, cost.Qf, Nh)
    else:
        eK = solve_edges(graph, Ad, Bd, cost.Q_edge, cost.R_edge, cost.Qf_edge, Nh)

    st = np.zeros((N_steps, Na, 12))
    ct = np.zeros((N_steps, Na, 4))
    fe = np.zeros((N_steps, graph.n_edges))
    te = np.zeros((N_steps, Na))
    st[0] = x.copy()

    if verbose:
        print("  Simulating ...")

    alpha, u_max = cost.formation_alpha, cost.u_max
    for k in range(N_steps - 1):
        u = np.zeros((Na, 4))
        for i in range(Na):
            u[i] = -K_lqr @ (x[i] - aref[k, i])

        if method == "centralized":
            xa = x.flatten() - aref[k].flatten()
            ki = min(k, Nh - 1)
            for i in range(Na):
                uc = -cK[i][ki] @ xa
                u[i] = 0.5 * u[i] + 0.5 * uc
        else:
            if method == "receding_horizon" and k > 0 and k % rh_window == 0:
                eK = solve_edges(graph, Ad, Bd, cost.Q_edge, cost.R_edge,
                                 cost.Qf_edge, min(rh_window, N_steps - k))
            ki = min(k if method == "distributed" else k % rh_window,
                     len(eK[0][0]) - 1)
            for eidx, (ei, ej) in enumerate(graph.edges):
                ze = (x[ei] - x[ej]) - (off[ei] - off[ej])
                u[ei] -= alpha * eK[eidx][0][ki] @ ze
                u[ej] += alpha * eK[eidx][1][ki] @ ze

        for i in range(Na):
            u[i] = np.clip(u[i], -u_max, u_max)
        ct[k] = u
        for i in range(Na):
            x[i] = Ad @ x[i] + Bd @ u[i]
        st[k + 1] = x.copy()

        for eidx, (ei, ej) in enumerate(graph.edges):
            fe[k+1, eidx] = abs(norm(x[ei, :3] - x[ej, :3])
                                - norm(off[ei, :3] - off[ej, :3]))
        for i in range(Na):
            te[k+1, i] = norm(x[i, :3] - aref[k+1, i, :3])

    return SimResult(st, ct, aref, np.arange(N_steps) * dt, fe, te)

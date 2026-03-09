"""Reference trajectory generators and formation offsets."""

from __future__ import annotations
import numpy as np


def helical_traj(N: int, dt: float, radius: float = 2.0,
                 omega: float = 0.3, vz: float = 0.15) -> np.ndarray:
    traj = np.zeros((N, 12))
    t = np.arange(N) * dt
    traj[:, 0] = radius * np.cos(omega * t)
    traj[:, 1] = radius * np.sin(omega * t)
    traj[:, 2] = vz * t
    traj[:, 3] = -radius * omega * np.sin(omega * t)
    traj[:, 4] = radius * omega * np.cos(omega * t)
    traj[:, 5] = vz
    return traj


def infinity_traj(N: int, dt: float, a: float = 3.0,
                  omega: float = 0.2, vz: float = 0.08) -> np.ndarray:
    traj = np.zeros((N, 12))
    t = np.arange(N) * dt
    s, c = np.sin(omega * t), np.cos(omega * t)
    d = 1.0 + s**2
    traj[:, 0] = a * c / d
    traj[:, 1] = a * s * c / d
    traj[:, 2] = vz * t
    ds, dc = omega * c, -omega * s
    dd = 2.0 * s * ds
    traj[:, 3] = (dc * d - a * c * dd) / d**2
    traj[:, 4] = ((ds * c + s * dc) * d - a * s * c * dd) / d**2
    traj[:, 5] = vz
    return traj


def formation_offsets(n_agents: int, dist: float = 1.5) -> np.ndarray:
    off = np.zeros((n_agents, 12))
    angles = 2.0 * np.pi * np.arange(n_agents) / n_agents
    off[:, 0] = dist * np.cos(angles)
    off[:, 1] = dist * np.sin(angles)
    return off

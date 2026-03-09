"""
Linearized quadrocopter dynamics.

State (12): [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
Input  (4): [delta_T, tau_phi, tau_theta, tau_psi]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class QuadParams:
    """Physical parameters of a single quadrocopter."""
    mass: float = 0.468
    g: float = 9.81
    Ixx: float = 4.856e-3
    Iyy: float = 4.856e-3
    Izz: float = 8.801e-3
    l: float = 0.225
    dt: float = 0.05
    n: int = 12
    m: int = 4


def linearized_model(p: QuadParams) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous-time (Ac, Bc) linearized about hover."""
    Ac = np.zeros((12, 12))
    Bc = np.zeros((12, 4))
    Ac[0, 3] = 1.0;  Ac[1, 4] = 1.0;  Ac[2, 5] = 1.0
    Ac[3, 7] = p.g;  Ac[4, 6] = -p.g
    Ac[6, 9] = 1.0;  Ac[7, 10] = 1.0; Ac[8, 11] = 1.0
    Bc[5, 0] = 1.0 / p.mass
    Bc[9, 1] = 1.0 / p.Ixx
    Bc[10, 2] = 1.0 / p.Iyy
    Bc[11, 3] = 1.0 / p.Izz
    return Ac, Bc


def discretize(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """ZOH discretization (2nd-order Taylor)."""
    n = Ac.shape[0]
    Ad = np.eye(n) + Ac * dt + 0.5 * (Ac @ Ac) * dt**2
    Bd = Bc * dt + 0.5 * (Ac @ Bc) * dt**2
    return Ad, Bd

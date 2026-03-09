"""Infinite-horizon discrete LQR (per-agent tracking baseline)."""

from __future__ import annotations
import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are


def lqr_gain(Ad: np.ndarray, Bd: np.ndarray,
             Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_discrete_are(Ad, Bd, Q, R)
    return inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

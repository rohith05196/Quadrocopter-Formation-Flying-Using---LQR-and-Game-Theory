"""Shared cost-tuning configuration."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CostConfig:
    q_pos: float = 50.0;  q_vel: float = 10.0
    q_ang: float = 5.0;   q_rate: float = 1.0
    r_input: float = 1.0; qf_scale: float = 5.0

    qe_pos: float = 30.0; qe_vel: float = 5.0
    qe_ang: float = 2.0;  qe_rate: float = 0.5
    re_input: float = 2.0; qfe_scale: float = 3.0

    formation_alpha: float = 0.3
    u_max: np.ndarray = field(default_factory=lambda: np.array([8.0, 1.0, 1.0, 0.5]))

    @property
    def Q(self):
        return np.diag([self.q_pos]*3 + [self.q_vel]*3 + [self.q_ang]*3 + [self.q_rate]*3)
    @property
    def R(self):
        return self.r_input * np.eye(4)
    @property
    def Qf(self):
        return self.qf_scale * self.Q
    @property
    def Q_edge(self):
        return np.diag([self.qe_pos]*3 + [self.qe_vel]*3 + [self.qe_ang]*3 + [self.qe_rate]*3)
    @property
    def R_edge(self):
        return self.re_input * np.eye(4)
    @property
    def Qf_edge(self):
        return self.qfe_scale * self.Q_edge

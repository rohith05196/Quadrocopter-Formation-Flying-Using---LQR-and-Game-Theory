"""
LQDTG — Linear Quadratic Discrete-Time Games for Quadrocopter Formation Flying
"""

__version__ = "2.0.0"

from lqdtg.models.quadrotor import QuadParams, linearized_model, discretize
from lqdtg.models.graph import Graph, graph3, graph4
from lqdtg.trajectories.generators import helical_traj, infinity_traj, formation_offsets
from lqdtg.solvers.lqr import lqr_gain
from lqdtg.solvers.distributed import solve_edge_nash, solve_edges
from lqdtg.solvers.centralized import solve_centralized
from lqdtg.config import CostConfig
from lqdtg.simulation.engine import SimResult, simulate
from lqdtg.visualization.plots import (
    plot3d, plot_errors, plot_controls, plot_comparison,
    plot_summary, plot_topdown_strip,
)
from lqdtg.visualization.animations import animate_3d, animate_topdown
from lqdtg.visualization._drone import draw_drone, COLORS

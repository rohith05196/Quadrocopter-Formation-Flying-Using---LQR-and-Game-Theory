"""Shared drone-drawing primitives and theme constants."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Theme ───────────────────────────────────────────────────────────────
COLORS      = ["#42A5F5", "#EF5350", "#66BB6A", "#FFA726", "#AB47BC"]
COLORS_DARK = ["#1565C0", "#C62828", "#2E7D32", "#E65100", "#6A1B9A"]
BG          = "#0D1117"
GRID        = "#21262D"
TXT         = "#C9D1D9"
ACCENT      = "#58A6FF"


# ── Rotation ────────────────────────────────────────────────────────────

def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX Euler rotation."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cps*cth, cps*sth*sph - sps*cph, cps*sth*cph + sps*sph],
        [sps*cth, sps*sth*sph + cps*cph, sps*sth*cph - cps*sph],
        [-sth,    cth*sph,                cth*cph               ],
    ])


# ── 3-D Drone ──────────────────────────────────────────────────────────

def draw_drone(ax, pos, angles, color, arm=0.35, alpha=0.9):
    """Draw a quadrotor: 4 arms, rotor discs, diamond body."""
    R = rotation_matrix(*angles)
    tips_body = arm * np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]]) / np.sqrt(2)
    tips = (R @ tips_body.T).T + pos

    # Arms
    for tip in tips:
        ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]],
                color=color, lw=2.2, alpha=alpha, solid_capstyle="round")

    # Rotor discs
    th = np.linspace(0, 2*np.pi, 16)
    circ = (arm * 0.38) * np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
    for tip in tips:
        disc = (R @ circ.T).T + tip
        verts = [list(zip(disc[:,0], disc[:,1], disc[:,2]))]
        ax.add_collection3d(Poly3DCollection(
            verts, alpha=0.25, facecolor=color, edgecolor=color, linewidths=0.6))

    # Diamond body
    bh = arm * 0.18
    bp = arm * 0.25 * np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]])
    bt = (R @ bp.T).T + pos + R @ [0, 0, bh]
    bb = (R @ bp.T).T + pos + R @ [0, 0, -bh]
    ax.add_collection3d(Poly3DCollection(
        [list(zip(bt[:,0], bt[:,1], bt[:,2]))],
        alpha=0.7, facecolor=color, edgecolor="white", linewidths=0.4))
    ax.add_collection3d(Poly3DCollection(
        [list(zip(bb[:,0], bb[:,1], bb[:,2]))],
        alpha=0.5, facecolor=color, edgecolor="white", linewidths=0.3))
    for k in range(4):
        k2 = (k + 1) % 4
        side = np.array([bt[k], bt[k2], bb[k2], bb[k]])
        ax.add_collection3d(Poly3DCollection(
            [list(zip(side[:,0], side[:,1], side[:,2]))],
            alpha=0.4, facecolor=color, edgecolor=color, linewidths=0.2))


def draw_drone_light(ax, pos, angles, color, arm=0.35, alpha=0.9):
    """Lightweight drone for animation (arms + rotor discs + dot)."""
    R = rotation_matrix(*angles)
    tips_body = arm * np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]]) / np.sqrt(2)
    tips = (R @ tips_body.T).T + pos
    for tip in tips:
        ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]],
                color=color, lw=2.5, alpha=alpha, solid_capstyle="round")
    th = np.linspace(0, 2*np.pi, 12)
    circ = (arm * 0.36) * np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
    for tip in tips:
        disc = (R @ circ.T).T + tip
        verts = [list(zip(disc[:,0], disc[:,1], disc[:,2]))]
        ax.add_collection3d(Poly3DCollection(
            verts, alpha=0.3, facecolor=color, edgecolor=color, linewidths=0.4))
    ax.scatter(*pos, c=color, s=30, edgecolors="white", linewidths=0.4, zorder=5)


# ── Axis styling ────────────────────────────────────────────────────────

def style_3d(ax, title=""):
    ax.set_facecolor(BG)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor(GRID)
    ax.tick_params(colors=TXT, labelsize=7)
    for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
        a.label.set_color(TXT)
        for gl in a.get_gridlines():
            gl.set_color(GRID); gl.set_alpha(0.4)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_zlabel("Z (m)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=TXT, pad=12)


def style_2d(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TXT, labelsize=8)
    for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        lbl.set_color(TXT)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.4, linewidth=0.5)


def legend(ax, **kw):
    return ax.legend(framealpha=0.3, facecolor=BG, edgecolor=GRID, labelcolor=TXT, **kw)

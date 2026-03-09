"""Enhanced static plots: 3-D with drones, dark theme, gradient trails."""

from __future__ import annotations
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa

from lqdtg.models.graph import Graph
from lqdtg.simulation.engine import SimResult
from lqdtg.visualization._drone import (
    COLORS, COLORS_DARK, BG, GRID, TXT, ACCENT,
    draw_drone, style_3d, style_2d, legend,
)


def _finish(fig, path):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
    return fig


# ── 3-D trajectories with drones ───────────────────────────────────────

def plot3d(result, graph, title="Formation — 3D", save_path=None,
           n_snapshots=8, drone_scale=0.35):
    fig = plt.figure(figsize=(14, 10)); fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d"); style_3d(ax, title)
    Nt = result.states.shape[0]
    snaps = np.linspace(0, Nt - 1, n_snapshots, dtype=int)

    for i in range(graph.n_agents):
        c, cd = COLORS[i % 5], COLORS_DARK[i % 5]
        seg = 6
        for s in range(0, Nt - seg, seg):
            f = s / Nt; sl = slice(s, s + seg + 1)
            ax.plot(result.states[sl, i, 0], result.states[sl, i, 1],
                    result.states[sl, i, 2], color=c,
                    lw=1.0 + 1.5*f, alpha=0.15 + 0.65*f, solid_capstyle="round")
        ax.plot(result.references[:, i, 0], result.references[:, i, 1],
                result.references[:, i, 2], color=c, ls=":", alpha=0.20, lw=0.8)
        ax.scatter(*result.states[0, i, :3], c=cd, marker="o", s=60,
                   edgecolors="white", linewidths=0.6, zorder=5)
        ax.scatter(*result.states[-1, i, :3], c=c, marker="D", s=70,
                   edgecolors="white", linewidths=0.6, zorder=5)
        for k in snaps:
            draw_drone(ax, result.states[k, i, :3], result.states[k, i, 6:9],
                       c, arm=drone_scale, alpha=0.35 + 0.55*(k/Nt))

    for k in snaps:
        for ei, ej in graph.edges:
            ax.plot([result.states[k, ei, 0], result.states[k, ej, 0]],
                    [result.states[k, ei, 1], result.states[k, ej, 1]],
                    [result.states[k, ei, 2], result.states[k, ej, 2]],
                    color=ACCENT, alpha=0.10 + 0.14*(k/Nt), lw=0.6, ls="--")

    h = [plt.Line2D([0],[0], color=COLORS[i], lw=2.5, label=f"Agent {i+1}")
         for i in range(graph.n_agents)]
    legend(ax, handles=h, loc="upper left", fontsize=9)
    plt.tight_layout(); return _finish(fig, save_path)


# ── Errors ──────────────────────────────────────────────────────────────

def plot_errors(result, graph, title="Errors", save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    fig.patch.set_facecolor(BG)
    for a in axes: style_2d(a)

    for eidx, (ei, ej) in enumerate(graph.edges):
        c = COLORS[(eidx+1) % 5]
        axes[0].plot(result.time, result.form_err[:, eidx], lw=1.6, color=c,
                     label=f"Edge ({ei+1},{ej+1})")
        axes[0].fill_between(result.time, 0, result.form_err[:, eidx], color=c, alpha=0.08)
    axes[0].set_ylabel("Formation Error (m)"); axes[0].set_title("Inter-Agent Distance Error", color=TXT)
    legend(axes[0], fontsize=8)

    for i in range(graph.n_agents):
        c = COLORS[i % 5]
        axes[1].plot(result.time, result.track_err[:, i], lw=1.6, color=c, label=f"Agent {i+1}")
        axes[1].fill_between(result.time, 0, result.track_err[:, i], color=c, alpha=0.06)
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Tracking Error (m)")
    axes[1].set_title("Trajectory Tracking Error", color=TXT)
    legend(axes[1], fontsize=8)
    plt.suptitle(title, fontsize=14, fontweight="bold", color=TXT)
    plt.tight_layout(); return _finish(fig, save_path)


# ── Controls ────────────────────────────────────────────────────────────

_ULBL = ["δT (N)", "τ_φ (Nm)", "τ_θ (Nm)", "τ_ψ (Nm)"]

def plot_controls(result, graph, title="Controls", save_path=None):
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.patch.set_facecolor(BG)
    for ui, ax in enumerate(axes):
        style_2d(ax)
        for i in range(graph.n_agents):
            ax.plot(result.time, result.controls[:, i, ui], color=COLORS[i%5], lw=1, alpha=0.85)
        ax.set_ylabel(_ULBL[ui], fontsize=9, color=TXT)
    h = [plt.Line2D([0],[0], color=COLORS[i], lw=2, label=f"Agent {i+1}") for i in range(graph.n_agents)]
    legend(axes[0], handles=h, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title, fontsize=14, fontweight="bold", color=TXT)
    plt.tight_layout(); return _finish(fig, save_path)


# ── Centralized vs Distributed ──────────────────────────────────────────

def plot_comparison(rc, rd, graph, title="Centralized vs Distributed", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor(BG)
    for a in axes: style_2d(a)
    cc, dc = "#EF5350", "#42A5F5"
    for i in range(graph.n_agents):
        axes[0].plot(rc.time, rc.track_err[:, i], color=cc, alpha=0.45, lw=1.5)
        axes[0].plot(rd.time, rd.track_err[:, i], color=dc, alpha=0.45, lw=1.5, ls="--")
    axes[0].plot([],[], color=cc, lw=2.5, label="Centralized")
    axes[0].plot([],[], color=dc, lw=2.5, ls="--", label="Distributed")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Tracking Error (m)")
    axes[0].set_title("Tracking Error", color=TXT); legend(axes[0])

    ce = np.sum(rc.controls**2, axis=(1,2)); de = np.sum(rd.controls**2, axis=(1,2))
    axes[1].plot(rc.time, ce, color=cc, lw=1.5, label="Centralized")
    axes[1].fill_between(rc.time, 0, ce, color=cc, alpha=0.10)
    axes[1].plot(rd.time, de, color=dc, lw=1.5, ls="--", label="Distributed")
    axes[1].fill_between(rd.time, 0, de, color=dc, alpha=0.10)
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("||u||²")
    axes[1].set_title("Control Effort", color=TXT); legend(axes[1])
    plt.suptitle(title, fontsize=14, fontweight="bold", color=TXT)
    plt.tight_layout(); return _finish(fig, save_path)


# ── Summary grid ────────────────────────────────────────────────────────

def plot_summary(scenarios, title="All Scenarios", save_path=None):
    n = len(scenarios); cols = min(n, 3); rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(7*cols, 6*rows)); fig.patch.set_facecolor(BG)
    for idx, (res, gr, label) in enumerate(scenarios):
        ax = fig.add_subplot(rows, cols, idx+1, projection="3d"); style_3d(ax, label)
        for i in range(gr.n_agents):
            c = COLORS[i % 5]
            ax.plot(res.states[:, i, 0], res.states[:, i, 1], res.states[:, i, 2],
                    color=c, lw=1.5, alpha=0.8)
            ax.scatter(*res.states[0, i, :3], c=c, marker="o", s=35, edgecolors="white", linewidths=0.4)
            draw_drone(ax, res.states[-1, i, :3], res.states[-1, i, 6:9], c, arm=0.30, alpha=0.85)
    plt.suptitle(title, fontsize=16, fontweight="bold", color=TXT)
    plt.tight_layout(); return _finish(fig, save_path)


# ── Top-down filmstrip ──────────────────────────────────────────────────

def plot_topdown_strip(result, graph, title="Top-Down", save_path=None, n_frames=6):
    Nt = result.states.shape[0]
    snaps = np.linspace(0, Nt - 1, n_frames, dtype=int)
    fig, axes = plt.subplots(1, n_frames, figsize=(3.2*n_frames, 3.5))
    fig.patch.set_facecolor(BG)
    for panel, k in enumerate(snaps):
        ax = axes[panel]; ax.set_facecolor(BG)
        ax.tick_params(colors=TXT, labelsize=6)
        for sp in ax.spines.values(): sp.set_color(GRID)
        for ei, ej in graph.edges:
            ax.plot([result.states[k, ei, 0], result.states[k, ej, 0]],
                    [result.states[k, ei, 1], result.states[k, ej, 1]],
                    color=ACCENT, alpha=0.5, lw=1.0, ls="--")
        for i in range(graph.n_agents):
            c = COLORS[i % 5]; tr = max(0, k - 30)
            ax.plot(result.states[tr:k+1, i, 0], result.states[tr:k+1, i, 1],
                    color=c, lw=1.0, alpha=0.4)
            ax.plot(result.states[k, i, 0], result.states[k, i, 1],
                    "o", color=c, ms=8, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_title(f"t = {result.time[k]:.1f}s", fontsize=9, color=TXT)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID, alpha=0.3, linewidth=0.4)
    plt.suptitle(title, fontsize=13, fontweight="bold", color=TXT)
    plt.tight_layout(); return _finish(fig, save_path)

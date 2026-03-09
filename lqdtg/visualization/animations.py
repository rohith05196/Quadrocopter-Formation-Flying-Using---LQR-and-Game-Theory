"""Animated GIFs: 3-D rotating flight and top-down 2-D view."""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from lqdtg.models.graph import Graph
from lqdtg.simulation.engine import SimResult
from lqdtg.visualization._drone import (
    COLORS, BG, GRID, TXT, ACCENT,
    draw_drone_light, style_3d,
)


# ═══════════════════════════════════════════════════════════════════════
#  3-D animated GIF
# ═══════════════════════════════════════════════════════════════════════

def animate_3d(
    result: SimResult, graph: Graph, save_path: str,
    title: str = "Formation Flight", fps: int = 18, step: int = 3,
    trail_len: int = 40, drone_scale: float = 0.30,
    rotate: bool = True, dpi: int = 100,
) -> None:
    Nt = result.states.shape[0]
    frames = list(range(0, Nt, step))
    nf = len(frames)

    pos = result.states[:, :, :3].reshape(-1, 3)
    mg = 1.5
    xl = [pos[:,0].min()-mg, pos[:,0].max()+mg]
    yl = [pos[:,1].min()-mg, pos[:,1].max()+mg]
    zl = [pos[:,2].min()-0.5, pos[:,2].max()+mg]

    fig = plt.figure(figsize=(10, 8)); fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    print(f"  Rendering {nf} 3-D frames ...")

    def _upd(fn):
        ax.cla(); style_3d(ax)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
        k = frames[fn]; t = result.time[k]
        ax.set_title(f"{title}   t = {t:.2f}s", fontsize=12,
                     fontweight="bold", color=TXT, pad=10)
        if rotate:
            ax.view_init(elev=25 + 10*np.sin(2*np.pi*fn/nf),
                         azim=-60 + 120*(fn/nf))

        for i in range(graph.n_agents):
            c = COLORS[i % 5]
            ax.plot(result.references[:, i, 0], result.references[:, i, 1],
                    result.references[:, i, 2], color=c, ls=":", alpha=0.12, lw=0.6)
            s0 = max(0, k - trail_len)
            trail = result.states[s0:k+1, i, :]
            ns = len(trail) - 1
            for s in range(ns):
                f = s / max(ns, 1)
                ax.plot(trail[s:s+2, 0], trail[s:s+2, 1], trail[s:s+2, 2],
                        color=c, lw=0.8 + 1.5*f, alpha=0.1 + 0.7*f, solid_capstyle="round")
            draw_drone_light(ax, result.states[k, i, :3], result.states[k, i, 6:9],
                             c, arm=drone_scale, alpha=0.9)

        for ei, ej in graph.edges:
            ax.plot([result.states[k,ei,0], result.states[k,ej,0]],
                    [result.states[k,ei,1], result.states[k,ej,1]],
                    [result.states[k,ei,2], result.states[k,ej,2]],
                    color=ACCENT, alpha=0.35, lw=0.8, ls="--")
        return []

    anim = animation.FuncAnimation(fig, _upd, frames=nf, blit=False)
    anim.save(save_path, writer="pillow", fps=fps, dpi=dpi,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"  Saved: {save_path}  ({nf} frames, {nf/fps:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════
#  Top-down 2-D animated GIF
# ═══════════════════════════════════════════════════════════════════════

def animate_topdown(
    result: SimResult, graph: Graph, save_path: str,
    title: str = "Top Down", fps: int = 18, step: int = 3,
    trail_len: int = 50, dpi: int = 100,
) -> None:
    Nt = result.states.shape[0]
    frames = list(range(0, Nt, step))
    nf = len(frames)

    pos = result.states[:, :, :2].reshape(-1, 2); mg = 2.0
    xl = [pos[:,0].min()-mg, pos[:,0].max()+mg]
    yl = [pos[:,1].min()-mg, pos[:,1].max()+mg]

    fig, ax = plt.subplots(figsize=(8, 8)); fig.patch.set_facecolor(BG)
    print(f"  Rendering {nf} top-down frames ...")

    def _upd(fn):
        ax.cla(); ax.set_facecolor(BG)
        ax.tick_params(colors=TXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.grid(True, color=GRID, alpha=0.3, linewidth=0.4)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
        ax.set_xlabel("X (m)", fontsize=10, color=TXT)
        ax.set_ylabel("Y (m)", fontsize=10, color=TXT)
        k = frames[fn]; t = result.time[k]
        ax.set_title(f"{title}   t = {t:.2f}s", fontsize=13, fontweight="bold", color=TXT)

        for i in range(graph.n_agents):
            c = COLORS[i % 5]
            ax.plot(result.references[:, i, 0], result.references[:, i, 1],
                    color=c, ls=":", alpha=0.15, lw=0.7)

        for ei, ej in graph.edges:
            ax.plot([result.states[k,ei,0], result.states[k,ej,0]],
                    [result.states[k,ei,1], result.states[k,ej,1]],
                    color=ACCENT, alpha=0.4, lw=1.2, ls="--")

        for i in range(graph.n_agents):
            c = COLORS[i % 5]
            s0 = max(0, k - trail_len)
            trail = result.states[s0:k+1, i, :]
            ns = len(trail) - 1
            for s in range(max(0, ns - trail_len), ns):
                f = (s - max(0, ns - trail_len)) / max(trail_len, 1)
                ax.plot(trail[s:s+2, 0], trail[s:s+2, 1],
                        color=c, lw=1.0 + 2.0*f, alpha=0.08 + 0.55*f)
            px, py = result.states[k, i, 0], result.states[k, i, 1]
            psi = result.states[k, i, 8]
            arm = 0.35
            for aoff in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
                dx, dy = arm*np.cos(psi+aoff), arm*np.sin(psi+aoff)
                ax.plot([px, px+dx], [py, py+dy], color=c, lw=2.5, alpha=0.8, solid_capstyle="round")
                ax.add_patch(plt.Circle((px+dx, py+dy), arm*0.25, fill=True,
                                        facecolor=c, edgecolor=c, alpha=0.25, lw=0.5))
            ax.plot(px, py, "o", color=c, ms=6, markeredgecolor="white", markeredgewidth=0.8, zorder=10)
        return []

    anim = animation.FuncAnimation(fig, _upd, frames=nf, blit=False)
    anim.save(save_path, writer="pillow", fps=fps, dpi=dpi,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"  Saved: {save_path}  ({nf} frames, {nf/fps:.1f}s)")

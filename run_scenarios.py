#!/usr/bin/env python3
"""
run_scenarios.py — Run all LQDTG formation-flying scenarios.

Usage:
    python run_scenarios.py                        # full run with animations
    python run_scenarios.py --output results/      # custom output directory
    python run_scenarios.py --steps 500            # longer simulation
    python run_scenarios.py --no-animations        # static plots only (faster)
"""

import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from lqdtg import (
    QuadParams, graph3, graph4, helical_traj, infinity_traj,
    simulate, plot3d, plot_errors, plot_controls, plot_comparison,
    plot_summary, plot_topdown_strip, animate_3d, animate_topdown,
)


def run(name, na, ttype, method="distributed", N=300, out="output", anim=True):
    print(f"\n{'═'*60}\n  {name}\n{'═'*60}")
    qp = QuadParams()
    g  = graph3() if na == 3 else graph4()
    tf = helical_traj if ttype == "helical" else infinity_traj
    print(f"  {na} agents, edges={g.edges}, traj={ttype}, method={method}")

    r = simulate(g, qp, tf, N_steps=N, method=method)
    tag = name.lower().replace(" ", "_").replace("-", "_")

    print("  Generating plots ...")
    plot3d(r, g, f"{name} — 3D",         f"{out}/{tag}_3d.png")
    plot_errors(r, g, name,               f"{out}/{tag}_err.png")
    plot_controls(r, g, f"{name} — Ctrl", f"{out}/{tag}_ctrl.png")
    plot_topdown_strip(r, g, f"{name}",   f"{out}/{tag}_topdown.png")

    if anim:
        print("  Generating 3-D animation ...")
        animate_3d(r, g, f"{out}/{tag}_3d.gif", title=name,
                   fps=18, step=3, trail_len=40, drone_scale=0.30)
        print("  Generating top-down animation ...")
        animate_topdown(r, g, f"{out}/{tag}_topdown.gif", title=name,
                        fps=18, step=3, trail_len=50)

    print(f"  Track err:  {np.mean(r.track_err[-1]):.4f} m")
    print(f"  Form err:   {np.mean(r.form_err[-1]):.4f} m")
    print(f"  Ctrl effort: {np.sum(r.controls**2):.1f}")
    return r, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="output")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--no-animations", action="store_true")
    args = ap.parse_args()
    out, N, anim = args.output, args.steps, not args.no_animations
    os.makedirs(out, exist_ok=True)

    r1,g1 = run("3 Agents Helical",  3, "helical",  "distributed",      N, out, anim)
    r2,g2 = run("3 Agents Infinity", 3, "infinity", "distributed",      N, out, anim)
    r3,g3 = run("4 Agents Helical",  4, "helical",  "distributed",      N, out, anim)
    r4,g4 = run("4 Agents Infinity", 4, "infinity", "distributed",      N, out, anim)
    r5,g5 = run("3 Agents Receding Horizon", 3, "helical", "receding_horizon", N, out, anim)
    r6,g6 = run("3 Agents Centralized",      3, "helical", "centralized",      N, out, anim)

    print("\n  Comparison plot ...")
    plot_comparison(r6, r1, g1, save_path=f"{out}/centralized_vs_distributed.png")

    print("  Summary panel ...")
    plot_summary([
        (r1,g1,"3 Agents — Helical"), (r2,g2,"3 Agents — Infinity"),
        (r3,g3,"4 Agents — Helical"), (r4,g4,"4 Agents — Infinity"),
        (r5,g5,"Receding Horizon"),   (r6,g6,"Centralized"),
    ], save_path=f"{out}/all_scenarios_summary.png")

    print(f"\n{'═'*60}")
    print(f"  ALL COMPLETE — {os.path.abspath(out)}/")
    print(f"{'═'*60}")

if __name__ == "__main__":
    main()

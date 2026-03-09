# LQDTG — Quadrocopter Formation Flying

Multi-agent quadrocopter formation control via **Linear Quadratic Discrete-Time Games**.

## Quick Start

```bash
pip install numpy scipy matplotlib pillow
python run_scenarios.py                    # full run (plots + GIF animations)
python run_scenarios.py --no-animations    # static plots only (faster)
python run_scenarios.py --steps 500        # longer simulation
```

## Structure

```
lqdtg/
├── __init__.py                  # public API
├── config.py                    # CostConfig (all tunable weights)
├── models/
│   ├── quadrotor.py             # QuadParams, linearized_model, discretize
│   └── graph.py                 # Graph, graph3(), graph4()
├── trajectories/
│   └── generators.py            # helical_traj, infinity_traj, formation_offsets
├── solvers/
│   ├── lqr.py                   # infinite-horizon discrete LQR
│   ├── distributed.py           # edge-based 2-player Nash (coupled Riccati)
│   └── centralized.py           # augmented N-player Nash (benchmark)
├── simulation/
│   └── engine.py                # SimResult, simulate()
└── visualization/
    ├── _drone.py                # 3-D drone model, theme constants, styling
    ├── plots.py                 # plot3d, plot_errors, plot_controls, ...
    └── animations.py            # animate_3d (rotating GIF), animate_topdown (2-D GIF)

run_scenarios.py                 # CLI entry-point
```

## Library Usage

```python
from lqdtg import *

r = simulate(graph3(), QuadParams(), helical_traj, N_steps=300)
plot3d(r, graph3(), save_path="traj.png")
animate_3d(r, graph3(), "flight.gif")
```

## Dependencies

Python ≥ 3.9 · NumPy · SciPy · Matplotlib · Pillow

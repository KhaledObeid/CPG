#!/usr/bin/env python3
"""
Plot CPG (Hopf) outputs without Gazebo.

Generates:
  - cpg_time_series_annotated.png  (x & z vs time for all legs, with stats & swing shading)
  - cpg_foot_trajectories.png      (x–z loops for all legs, 2×2)
  - cpg_foot_trajectory_<LEG>.png  (x–z loop for a chosen single leg)
"""

import os
import argparse
import numpy as np

# Headless-safe backend
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Import HopfNetwork
try:
    from quad_cpg.hopf_network import HopfNetwork
except Exception:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from quad_cpg.hopf_network import HopfNetwork


LEG_NAMES = ["FR", "FL", "RR", "RL"]
LEG_MAP = {name: idx for idx, name in enumerate(LEG_NAMES)}


def simulate_cpg(gait: str, total_time: float, dt: float):
    steps = int(np.round(total_time / dt))
    t = np.arange(steps) * dt
    cpg = HopfNetwork(time_step=dt, gait=gait)

    xs_hist = np.zeros((steps, 4))
    zs_hist = np.zeros((steps, 4))
    # [r, theta, rdot, thetadot] per leg
    cpg_history = np.zeros((steps, 4, 4))

    for k in range(steps):
        xs, zs = cpg.update()
        xs_hist[k, :] = xs
        zs_hist[k, :] = zs
        cpg_history[k, :, :] = np.concatenate((cpg.X, cpg.X_dot), axis=0)

    return t, xs_hist, zs_hist, cpg_history


def plot_time_series(t, xs_hist, zs_hist, cpg_history, out_path="cpg_time_series_annotated.png"):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8))
    for i, ax in enumerate(axes):
        x = xs_hist[:, i]
        z = zs_hist[:, i]
        theta = cpg_history[:, 1, i]

        ax.plot(t, x, lw=2, label="x (fore–aft) [m]")
        ax.plot(t, z, lw=2, label="z (height) [m]")

        y_min = min(x.min(), z.min()) - 0.02
        y_max = max(x.max(), z.max()) + 0.02
        ax.set_ylim(y_min, y_max)

        swing_mask = theta < np.pi
        ax.fill_between(t, y_min, y_max, where=swing_mask, step='pre',
                        alpha=0.09, color="tab:green", label="swing")

        settle = int(0.3 * len(t))
        stance_duty = float(np.mean(theta[settle:] >= np.pi))
        ax.set_title(f"CPG outputs (leg {LEG_NAMES[i]}) — stance duty ≈ {stance_duty:.2f}")

        idx = slice(len(t)//2, None)

        def stats(sig):
            s = sig[idx]
            return s.mean(), s.min(), s.max(), (s.max() - s.min())

        mx, mix, maxx, ppx = stats(x)
        mz, miz, maz, ppz = stats(z)
        txt = (f"x: mean={mx:.3f} m, min={mix:.3f}, max={maxx:.3f}, p2p={ppx:.3f}\n"
               f"z: mean={mz:.3f} m, min={miz:.3f}, max={maz:.3f}, p2p={ppz:.3f}")
        ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

        ax.grid(True, which='both', alpha=0.35)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylabel("m")

    axes[0].legend(ncol=3, loc='upper left', framealpha=0.9)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    try:
        plt.show()
    except Exception:
        pass
    print(f"Saved annotated time-series to {out_path}")


def plot_foot_trajectories(xs_hist, zs_hist, out_path="cpg_foot_trajectories.png"):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        ax.plot(xs_hist[:, i], zs_hist[:, i], lw=1.8)
        ax.set_title(f"Foot trajectory (leg {LEG_NAMES[i]})")
        ax.set_xlabel("x (fore–aft) [m]")
        ax.set_ylabel("z (height) [m]")
        ax.grid(True, alpha=0.35)
        ax.set_aspect("equal")
        xi, zi = xs_hist[:, i], zs_hist[:, i]
        ax.set_xlim(xi.min() - 0.01, xi.max() + 0.01)
        ax.set_ylim(zi.min() - 0.01, zi.max() + 0.01)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    try:
        plt.show()
    except Exception:
        pass
    print(f"Saved foot trajectories to {out_path}")


def plot_single_foot_trajectory(xs_hist, zs_hist, leg="FR"):
    """Optional single-leg overlay like your original FR plot."""
    idx = LEG_MAP.get(leg.upper(), 0)
    out_path = f"cpg_foot_trajectory_{LEG_NAMES[idx]}.png"

    plt.figure(figsize=(6, 4.5))
    plt.plot(xs_hist[:, idx], zs_hist[:, idx], lw=2)
    plt.title(f"Foot trajectory from CPG (leg {LEG_NAMES[idx]})")
    plt.xlabel("x (fore–aft) [m]")
    plt.ylabel("z (height) [m]")
    plt.grid(True, alpha=0.45)
    plt.gca().set_aspect("equal")
    xi, zi = xs_hist[:, idx], zs_hist[:, idx]
    plt.xlim(xi.min() - 0.01, xi.max() + 0.01)
    plt.ylim(zi.min() - 0.01, zi.max() + 0.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    try:
        plt.show()
    except Exception:
        pass
    print(f"Saved single foot trajectory to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Hopf CPG outputs without Gazebo.")
    parser.add_argument("--gait", default="PRONK",
                        choices=["WALK", "WALK_DIAG", "AMBLE",
                                 "TROT", "TROT_RUN", "TROT_WALK",
                                 "PACE", "PACE_FLY",
                                 "CANTER_TRANS", "CANTER_ROTA",
                                 "BOUND", "GALLOP_ROTA", "GALLOP_TRANS", "GALLOP",
                                 "PRONK"],
                        help="Gait to set in HopfNetwork.")
    parser.add_argument("--time", type=float, default=1.0, help="Total simulation time [s].")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step [s].")
    parser.add_argument("--single_traj", default="FR",
                        help="Also save a single-leg trajectory image (choices: FR,FL,RR,RL).")
    args = parser.parse_args()

    print(args.gait)
    t, xs_hist, zs_hist, cpg_history = simulate_cpg(args.gait, args.time, args.dt)
    plot_time_series(t, xs_hist, zs_hist, cpg_history, out_path="cpg_time_series_annotated.png")
    plot_foot_trajectories(xs_hist, zs_hist, out_path="cpg_foot_trajectories.png")
    if args.single_traj:
        plot_single_foot_trajectory(xs_hist, zs_hist, leg=args.single_traj)


if __name__ == "__main__":
    main()

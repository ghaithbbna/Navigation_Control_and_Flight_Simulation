"""M4.3: Monte Carlo over random worlds (gust + elevator error).

    python experiments/m4_monte_carlo.py tune    # seeds 100-119: for developing and tuning
    python experiments/m4_monte_carlo.py test    # seeds 0-49: the comparison that counts (default)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from navigation_control_flight_simulation.mission import (
    TOLERANCE_FT, fly, fly_c1, gate_errors, make_world, raw_reference, shaped_reference)
from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.mpc_incremental import IncrementalMPC
from navigation_control_flight_simulation.plant import course_plant

which = sys.argv[1] if len(sys.argv) > 1 else "test"
seeds = {"tune": range(100, 120), "test": range(0, 50)}[which]

plant = course_plant()
c0, c1 = ExplicitMPC(plant), IncrementalMPC(plant)
raw, shaped = raw_reference(), shaped_reference()
configs = {          # name: (flight function, reference, energy speed target)
    "C0 staircase, speed 0": (lambda r, e, w: fly(c0, r, e, world=w), raw, False),
    "C0 staircase": (lambda r, e, w: fly(c0, r, e, world=w), raw, True),
    "C0 shaped": (lambda r, e, w: fly(c0, r, e, world=w), shaped, True),
    "C1 staircase": (lambda r, e, w: fly_c1(c1, r, e, world=w), raw, True),
    "C1 shaped": (lambda r, e, w: fly_c1(c1, r, e, world=w), shaped, True),
}

# per configuration, one value per seed
res = {n: {k: [] for k in ("worst", "rms", "pass", "sat", "move")} for n in configs}
for seed in seeds:
    world = make_world(seed)
    for name, (flight, ref, energy) in configs.items():
        f = flight(ref, energy, world)
        e, el = gate_errors(f), f["elevator_deg"]
        r = res[name]
        r["worst"].append(np.abs(e).max())
        r["rms"].append(np.sqrt(np.mean(e ** 2)))
        r["pass"].append(np.all(np.abs(e) <= TOLERANCE_FT))
        r["sat"].append(100 * np.mean(np.abs(el) >= 24.99))
        r["move"].append(np.abs(np.diff(el)).sum())
res = {n: {k: np.array(v) for k, v in r.items()} for n, r in res.items()}

print(f"{len(seeds)} seeds ({which}); median [25th, 75th percentile]")
print(f"{'configuration':24s} {'worst miss (ft)':>20s} {'RMS miss (ft)':>20s} {'all gates in 50 ft':>19s} {'elevator at limit %':>20s} {'elevator movement':>18s}")
q = lambda v: f"{np.median(v):6.1f} [{np.percentile(v, 25):5.1f},{np.percentile(v, 75):6.1f}]"
for n, r in res.items():
    print(f"{n:24s} {q(r['worst']):>20s} {q(r['rms']):>20s} {100 * r['pass'].mean():17.0f} % {q(r['sat']):>20s} {np.median(r['move']):15.0f} deg")


def compare(label, better, worse, key):
    d = res[worse][key] - res[better][key]            # positive = `better` wins
    ok = np.mean(d > 0) >= 0.8 and np.median(d) > d.std(ddof=1)
    print(f"{label:62s} wins {100 * np.mean(d > 0):4.0f}% of seeds, median gain {np.median(d):6.1f}, spread {d.std(ddof=1):6.1f}  ->  "
          f"{'SUPPORTED' if ok else 'not supported'}")


print("\nH-a  energy speed target beats speed target 0")
compare("  worst miss, C0 with staircase", "C0 staircase", "C0 staircase, speed 0", "worst")
print("H-b  shaped reference beats staircase (less limit time, accuracy not hurt)")
compare("  time at the elevator limit (% of steps), C0", "C0 shaped", "C0 staircase", "sat")
compare("  worst miss, C0", "C0 shaped", "C0 staircase", "worst")
print("H-c  C1 beats C0")
compare("  worst miss, staircase", "C1 staircase", "C0 staircase", "worst")
compare("  worst miss, shaped", "C1 shaped", "C0 shaped", "worst")
compare("  RMS miss, shaped", "C1 shaped", "C0 shaped", "rms")

# ------------------------------------------------------------------ figure: one dot per seed
INK, MUTED, GRID, BLUE, GREEN = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#1baf7a"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "font.size": 10})
show = ["C0 staircase", "C1 staircase", "C0 shaped", "C1 shaped"]
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.25})
for ax, key, title in zip(axs, ("worst", "sat"), ("Worst gate miss per seed (ft)", "Time at the elevator limit per seed (% of steps)")):
    for i, n in enumerate(show):
        v, colour = res[n][key], BLUE if n.startswith("C0") else GREEN
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=22, color=colour, alpha=0.65, lw=0)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=2)
    ax.set_xticks(range(len(show)), show)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)
axs[0].axhline(TOLERANCE_FT, color=MUTED, ls=":", lw=1.2)
axs[0].text(3.45, TOLERANCE_FT + 2, "±50 ft tolerance", ha="right", color=MUTED, fontsize=9)
out = Path(__file__).resolve().parent.parent / "results" / f"m4_monte_carlo_{which}.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.show()
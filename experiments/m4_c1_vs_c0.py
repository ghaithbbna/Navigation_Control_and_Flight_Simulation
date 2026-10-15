"""M4.2: C1 against C0 on the five Munich gates (noise-free, one run each)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from navigation_control_flight_simulation.mission import (
    GATE_DH, GATE_S, TOLERANCE_FT, T, fly, fly_c1, gate_errors, raw_reference, shaped_reference)
from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.mpc_incremental import IncrementalMPC
from navigation_control_flight_simulation.plant import course_plant
from navigation_control_flight_simulation.units import M_PER_FT

plant = course_plant()
c0, c1 = ExplicitMPC(plant), IncrementalMPC(plant)
c1_one = IncrementalMPC(plant, blocks=(0,))
raw, shaped = raw_reference(), shaped_reference()

# label, flight function, reference name, reference, speed target from energy?
cases = [
    ("C0 100 Hz", lambda r, e: fly(c0, r, e), "raw", raw, True),
    ("C1 10 Hz", lambda r, e: fly_c1(c1, r, e), "raw", raw, True),
    ("C0 100 Hz", lambda r, e: fly(c0, r, e), "shaped", shaped, True),
    ("C1 10 Hz", lambda r, e: fly_c1(c1, r, e), "shaped", shaped, True),
    ("C1 10 Hz, speed target 0", lambda r, e: fly_c1(c1, r, e), "shaped", shaped, False),
    ("C1 one move, end value only", lambda r, e: fly_c1(c1_one, r, e, preview=False), "raw", raw, True),
    ("C1 one move, end value only", lambda r, e: fly_c1(c1_one, r, e, preview=False), "shaped", shaped, True),
]

print(f"{'controller':30s} {'reference':>9s} {'worst miss':>11s} {'RMS miss':>9s} {'within 50 ft':>13s} "
      f"{'elevator at limit':>18s} {'elevator movement':>18s}")
runs = {}
for label, flight, ref_name, ref, energy in cases:
    f = flight(ref, energy)
    e = gate_errors(f)
    el = f["elevator_deg"]
    runs[(label, ref_name, energy)] = (f, e)
    print(f"{label:30s} {ref_name:>9s} {np.abs(e).max():9.1f} ft {np.sqrt(np.mean(e ** 2)):7.1f} ft "
          f"{np.sum(np.abs(e) <= TOLERANCE_FT):10d}/5 {100 * np.mean(np.abs(el) >= 24.99):16.1f} % "
          f"{np.abs(np.diff(el)).sum():16.0f} deg")

# ------------------------------------------------------------------ figure (shaped reference)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, GREEN = "#2a78d6", "#1baf7a"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
                     "font.size": 10})
(f0, e0), (f1, e1) = runs[("C0 100 Hz", "shaped", True)], runs[("C1 10 Hz", "shaped", True)]
gate_km = GATE_S * M_PER_FT / 1000

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.32, wspace=0.22)

ax = fig.add_subplot(gs[0, :])
ax.plot(T * 500 * M_PER_FT / 1000, shaped, "--", color=MUTED, lw=1, label="shaped reference")
for f, e, colour, name in [(f0, e0, BLUE, "C0"), (f1, e1, GREEN, "C1")]:
    ax.plot(f["s"] * M_PER_FT / 1000, f["h"], color=colour, lw=2,
            label=f"{name}  (worst miss {np.abs(e).max():.0f} ft)")
ax.errorbar(gate_km, GATE_DH, yerr=TOLERANCE_FT, fmt="D", color=INK, ms=7, capsize=5, lw=1.5,
            label=f"gate  (bar = ±{TOLERANCE_FT:.0f} ft tolerance)", zorder=5)
ax.set_xlabel("distance along the track from Marienplatz, heading east (km)")
ax.set_ylabel("altitude above 10 000 ft (ft)")
ax.set_title("Altitude profile through the five gates (shaped reference, energy speed target)",
             loc="left", fontweight="bold")
ax.set_ylim(-40, 250)
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)

ax = fig.add_subplot(gs[1, 0])
ax.plot(T, f0["elevator_deg"], color=BLUE, lw=1.6, label="C0")
ax.plot(T, f1["elevator_deg"], color=GREEN, lw=1.6, label="C1")
for y in (-25, 25):
    ax.axhline(y, color=MUTED, ls=":", lw=1.2)
ax.text(59, 22.5, "limit +25°", ha="right", color=MUTED, fontsize=9)
ax.text(59, -23.5, "limit −25°", ha="right", color=MUTED, fontsize=9)
ax.set_xlabel("time (s)")
ax.set_ylabel("elevator command (deg)")
ax.set_title("Elevator demand", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.88))

ax = fig.add_subplot(gs[1, 1])
ax.axhspan(-TOLERANCE_FT, TOLERANCE_FT, color="#d9d8d3", alpha=0.5, lw=0)
ax.text(5.45, 41, "tolerance", color=MUTED, fontsize=9, ha="right")
w = 0.36
ax.bar(np.arange(1, 6) - w / 2, e0, width=w - 0.03, color=BLUE, label="C0")
ax.bar(np.arange(1, 6) + w / 2, e1, width=w - 0.03, color=GREEN, label="C1")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(range(1, 6))
ax.set_xlabel("gate")
ax.set_ylabel("altitude error at the gate (ft)")
ax.set_title("Gate errors", loc="left", fontweight="bold")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="lower right")

out = Path(__file__).resolve().parent.parent / "results" / "m4_c1_vs_c0.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.show()
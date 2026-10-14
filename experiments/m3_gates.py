"""M3.3: C0 flies the five Munich altitude gates. Does the speed target / a smooth reference matter?"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from navigation_control_flight_simulation.mission import (
    GATE_DH, GATE_S, TOLERANCE_FT, T, fly, gate_errors, raw_reference, shaped_reference)
from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.plant import course_plant
from navigation_control_flight_simulation.units import M_PER_FT

mpc = ExplicitMPC(course_plant())
raw, shaped = raw_reference(), shaped_reference()

cases = [                      # label, reference, energy speed target, colour (validated palette slots 1-3)
    ("course style: staircase, speed target 0", raw, False, "#2a78d6"),
    ("+ energy speed target", raw, True, "#eb6834"),
    ("+ shaped reference (proposed)", shaped, True, "#1baf7a"),
    ("shaped reference only (no energy target)", shaped, False, None),
]

print(f"{'configuration':44s} {'worst miss (ft)':>15s} {'RMS miss (ft)':>14s} {'within 50 ft':>13s} {'elevator range (deg)':>22s}")
runs = {}
for label, ref, energy, colour in cases:
    f = fly(mpc, ref, energy)
    e = gate_errors(f)
    runs[label] = (f, e, colour)
    el = f["elevator_deg"]
    print(f"{label:44s} {np.abs(e).max():15.1f} {np.sqrt(np.mean(e ** 2)):14.1f} "
          f"{np.sum(np.abs(e) <= TOLERANCE_FT):10d}/5 {el.min():11.1f} .. {el.max():5.1f}")
    
# ------------------------------------------------------------------ figure
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
                     "font.size": 10})
shown = [c for c in cases if c[3] is not None]
gate_km = GATE_S * M_PER_FT / 1000

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.32, wspace=0.22)

ax = fig.add_subplot(gs[0, :])
ax.plot(T * 500 * M_PER_FT / 1000, shaped, "--", color=MUTED, lw=1, label="shaped reference")
for label, _, _, colour in shown:
    f = runs[label][0]
    ax.plot(f["s"] * M_PER_FT / 1000, f["h"], color=colour, lw=2,
            label=f"{label}  (worst miss {np.abs(runs[label][1]).max():.0f} ft)")
ax.errorbar(gate_km, GATE_DH, yerr=TOLERANCE_FT, fmt="D", color=INK, ms=7, capsize=5, lw=1.5,
            label=f"gate  (bar = ±{TOLERANCE_FT:.0f} ft tolerance)", zorder=5)
ax.set_xlabel("distance along the track from Marienplatz, heading east (km)")
ax.set_ylabel("altitude above 10 000 ft (ft)")
ax.set_title("Altitude profile through the five gates", loc="left", fontweight="bold")
ax.set_ylim(-40, 300)
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)

ax = fig.add_subplot(gs[1, 0])
for label, _, _, colour in shown:
    ax.plot(T, runs[label][0]["elevator_deg"], color=colour, lw=1.6)
for y in (-25, 25):
    ax.axhline(y, color=MUTED, ls=":", lw=1.2)
ax.text(59, 22.5, "limit +25°", ha="right", color=MUTED, fontsize=9)
ax.text(59, -23.5, "limit −25°", ha="right", color=MUTED, fontsize=9)
ax.set_xlabel("time (s)")
ax.set_ylabel("elevator command (deg)")
ax.set_title("Elevator demand", loc="left", fontweight="bold")

ax = fig.add_subplot(gs[1, 1])
ax.axhspan(-TOLERANCE_FT, TOLERANCE_FT, color="#d9d8d3", alpha=0.5, lw=0)
ax.text(5.55, 41, "tolerance", color=MUTED, fontsize=9, ha="right")
w = 0.26
for j, (label, _, _, colour) in enumerate(shown):
    ax.bar(np.arange(1, 6) + (j - 1) * w, runs[label][1], width=w - 0.03, color=colour)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(range(1, 6))
ax.set_xlabel("gate")
ax.set_ylabel("altitude error at the gate (ft)")
ax.set_title("Gate errors", loc="left", fontweight="bold")
ax.grid(axis="x", visible=False)

out = Path(__file__).resolve().parent.parent / "results" / "m3_gates.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.show()
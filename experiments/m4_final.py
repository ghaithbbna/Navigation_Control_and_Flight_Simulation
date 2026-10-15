import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from navigation_control_flight_simulation.mission import (
    TOLERANCE_FT, fly, fly_c1, gate_errors, linear_reference, make_world, raw_reference, shaped_reference, shifted)
from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.mpc_incremental import IncrementalMPC
from navigation_control_flight_simulation.plant import course_plant

which = sys.argv[1] if len(sys.argv) > 1 else "test"
seeds, abl_seeds = (range(0, 50), range(0, 20)) if which == "test" else (range(100, 105), range(100, 105))
root = Path(__file__).resolve().parent.parent
tuned = json.loads((root / "results" / "tuned.json").read_text())
plant = course_plant()
refs = {"staircase": raw_reference(), "linear": linear_reference(), "shaped": shaped_reference()}


def setting(ctrl, ref, energy):
    t = tuned[f"{ctrl}|{ref}|{energy}"]
    return np.diag([1.0, 1.0, t["qv"], 1.0, 1.0]), shifted(refs[ref], t["delay"])


def measure(f):
    e, el = gate_errors(f), f["elevator_deg"]
    return dict(worst=np.abs(e).max(), rms=np.sqrt(np.mean(e ** 2)), sat=100 * np.mean(np.abs(el) >= 24.99),
                move=np.abs(np.diff(el)).sum())


def run(flights, seed_list):
    """flights: {name: function(world) -> flight}.  Returns {name: {metric: array over seeds}}."""
    out = {n: [] for n in flights}
    for s in seed_list:
        w = make_world(s)
        for n, fl in flights.items():
            out[n].append(measure(fl(w)))
    return {n: {k: np.array([m[k] for m in v]) for k in v[0]} for n, v in out.items()}


def table(title, res):
    q = lambda v: f"{np.median(v):5.1f} [{np.percentile(v, 25):5.1f},{np.percentile(v, 75):5.1f}]"
    print(f"\n{title}\n{'':44s} {'worst miss (ft)':>20s} {'RMS (ft)':>7s} {'at limit %':>11s} {'movement':>9s}")
    for n, r in res.items():
        print(f"{n:44s} {q(r['worst']):>20s} {np.median(r['rms']):7.1f} {np.median(r['sat']):11.1f} {np.median(r['move']):8.0f}°")


# ------------------------------------------------------------------ main comparison (all tuned)
main = {}
for c in ("C0", "C1"):
    for r in refs:
        for e in (True, False):
            Q, href = setting(c, r, e)
            ctrl = ExplicitMPC(plant, Q=Q) if c == "C0" else IncrementalMPC(plant, Q=Q)
            fl = fly if c == "C0" else fly_c1
            main[f"{c} {r}, speed target {'energy' if e else '0'}"] = (lambda w, fl=fl, ctrl=ctrl, href=href, e=e: fl(ctrl, href, e, world=w))
res = run(main, seeds)
table(f"{len(seeds)} seeds ({which}), tuned settings; median [25th, 75th percentile]", res)
print("all gates within 50 ft:", {n: f"{100 * np.mean(r['worst'] <= TOLERANCE_FT):.0f}%" for n, r in res.items()})


def compare(label, better, worse, key):
    d = res[worse][key] - res[better][key]            # positive = `better` wins
    ok = np.mean(d > 0) >= 0.8 and np.median(d) > d.std(ddof=1)
    print(f"  {label:48s} wins {100 * np.mean(d > 0):4.0f}% of seeds, median gain {np.median(d):6.1f}, spread {d.std(ddof=1):5.1f}  ->  "
          f"{'SUPPORTED' if ok else 'not supported'}")


n = lambda c, r, e=True: f"{c} {r}, speed target {'energy' if e else '0'}"
print("\nH-a  energy speed target beats speed target 0 (worst miss)")
for c in ("C0", "C1"):
    for r in ("staircase", "shaped"):
        compare(f"{c} {r}", n(c, r), n(c, r, False), "worst")
print("H-b  shaped reference beats staircase / linear")
for c in ("C0", "C1"):
    for other in ("staircase", "linear"):
        compare(f"{c} against {other}: worst miss", n(c, "shaped"), n(c, other), "worst")
        compare(f"{c} against {other}: time at the elevator limit", n(c, "shaped"), n(c, other), "sat")
print("H-c  C1 beats C0")
for r in refs:
    compare(f"{r}: worst miss", n("C1", r), n("C0", r), "worst")
    compare(f"{r}: RMS miss", n("C1", r), n("C0", r), "rms")

# ------------------------------------------------------------------ ablations (shaped reference, energy target)
Qc0, href0 = setting("C0", "shaped", True)
Qc1, href1 = setting("C1", "shaped", True)
c0, c1 = ExplicitMPC(plant, Q=Qc0), IncrementalMPC(plant, Q=Qc1)
f0 = lambda ctrl, **kw: (lambda w: fly(ctrl, href0, True, world=w, **kw))
f1 = lambda ctrl, **kw: (lambda w: fly_c1(ctrl, href1, True, world=w, **kw))

table("Ablation 1: what makes C1 better than C0?", run({
    "C0 (100 Hz, penalty on the command)": f0(c0),
    "C1 full (6 moves, whole reference)": f1(c1),
    "C1 one move, whole reference": f1(IncrementalMPC(plant, Q=Qc1, blocks=(0,))),
    "C1 6 moves, end value only": f1(c1, preview=False),
    "C1 one move, end value only": f1(IncrementalMPC(plant, Q=Qc1, blocks=(0,)), preview=False),
}, abl_seeds))


class Clipped(ExplicitMPC):
    """C0 without the QP: clip the unconstrained command to the limits."""

    def control(self, x, h_ref, V_ref):
        r = np.zeros(5)
        r[0], r[2] = h_ref, self.plant.C[2, 2] * V_ref
        E = np.zeros(5)
        E[[0, 2]] = (r - self.F @ x)[[0, 2]]
        return np.clip(self.K @ E, -self.g[2:], self.g[:2])


table("Ablation 2: QP against clipping (C0)", run({"C0 with the QP": f0(c0), "C0 clipped": f0(Clipped(plant, Q=Qc0))}, abl_seeds))

rate = {}
for Ts in (0.2, 0.1, 0.05, 0.02):
    blocks = sorted({round(t / Ts) for t in (0, 0.1, 0.2, 0.4, 0.8, 1.6)})
    ctrl = IncrementalMPC(plant, Ts=Ts, N=round(3.5 / Ts), blocks=blocks, Q=Qc1)
    rate[f"C1 at {1 / Ts:.0f} Hz"] = (lambda w, ctrl=ctrl, Ts=Ts: fly_c1(ctrl, href1, True, Ts=Ts, world=w))
table("Ablation 3: sample rate of C1", run(rate, abl_seeds))

table("Ablation 4: a real elevator cannot move instantly", run({
    "C0 no limit": f0(c0), "C0 40 deg/s": f0(c0, rate_deg_s=40), "C0 20 deg/s": f0(c0, rate_deg_s=20),
    "C1 no limit": f1(c1), "C1 40 deg/s": f1(c1, rate_deg_s=40), "C1 20 deg/s": f1(c1, rate_deg_s=20)}, abl_seeds))

kappa = {}
for k in (0, 1, 10):
    p = plant.with_thrust_scale(k)
    kappa[f"C0, thrust scale {k}"] = (lambda w, p=p: fly(ExplicitMPC(p, Q=Qc0), href0, True, world=w))
    kappa[f"C1, thrust scale {k}"] = (lambda w, p=p: fly_c1(IncrementalMPC(p, Q=Qc1), href1, True, world=w))
table("Ablation 5: thrust authority (settings not re-tuned)", run(kappa, abl_seeds))

# ------------------------------------------------------------------ figure: worst miss per seed
INK, MUTED, GRID, BLUE, GREEN = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#1baf7a"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "font.size": 10})
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, e, title in zip(axs, (True, False), ("Speed target from the energy relation", "Speed target 0")):
    for i, (c, r) in enumerate([(c, r) for r in refs for c in ("C0", "C1")]):
        v = res[n(c, r, e)]["worst"]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=22, color=BLUE if c == "C0" else GREEN, alpha=0.65, lw=0)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=2)
    ax.set_xticks(range(6), [f"{c}\n{r}" for r in refs for c in ("C0", "C1")])
    ax.set_title(title, loc="left", fontweight="bold")
    ax.axhline(TOLERANCE_FT, color=MUTED, ls=":", lw=1.2)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("worst gate miss per seed (ft), tuned settings")
out = root / "results" / f"m4_final_{which}.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.show()
"""M4.4: tune every configuration on the tuning seeds (100-109), never on the test seeds.
"""

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from navigation_control_flight_simulation.mission import (
    fly, fly_c1, gate_errors, linear_reference, make_world, raw_reference, shaped_reference, shifted)
from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.mpc_incremental import IncrementalMPC
from navigation_control_flight_simulation.plant import course_plant

QV, DELAY = [0.01, 0.1, 1, 10, 49], [-1, 0, 1, 2, 3, 4]
worlds = [make_world(s) for s in range(100, 110)]
plant = course_plant()
refs = {"staircase": raw_reference(), "linear": linear_reference(), "shaped": shaped_reference()}
build = {"C0": (ExplicitMPC, fly), "C1": (IncrementalMPC, fly_c1)}


def tune(ctrl_name, ref_name, energy):
    make, flight = build[ctrl_name]
    ctrls = {qv: make(plant, Q=np.diag([1.0, 1.0, qv, 1.0, 1.0])) for qv in QV}

    @lru_cache(maxsize=None)
    def score(qv, d):                                   # median worst miss over the tuning seeds
        ref = shifted(refs[ref_name], d)
        return float(np.median([np.abs(gate_errors(flight(ctrls[qv], ref, energy, world=w))).max() for w in worlds]))

    qv, d = 49, 0                                       # start from the course settings
    start = score(qv, d)
    d = min(DELAY, key=lambda x: score(qv, x))
    qv = min(QV, key=lambda x: score(x, d))
    d = min(DELAY, key=lambda x: score(qv, x))
    return qv, d, start, score(qv, d)


tuned = {}
print(f"{'configuration':32s} {'course settings':>16s} {'tuned':>8s}   qv   delay")
for ctrl_name in ("C0", "C1"):
    for ref_name in refs:
        for energy in (True, False):
            t = time.time()
            qv, d, start, best = tune(ctrl_name, ref_name, energy)
            tuned[f"{ctrl_name}|{ref_name}|{energy}"] = dict(qv=qv, delay=d)
            label = f"{ctrl_name} {ref_name}, speed target {'energy' if energy else '0'}"
            print(f"{label:32s} {start:13.1f} ft {best:6.1f} ft  {qv:5g} {d:4d} s   ({time.time() - t:.0f} s)", flush=True)

out = Path(__file__).resolve().parent.parent / "results" / "tuned.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(tuned, indent=1))
print("written", out)
"""M3.4: fly the two missions and save them for the FlightGear replay."""

from pathlib import Path

import numpy as np

from navigation_control_flight_simulation.mission import (
    DT, GATE_DH, GATE_S, fly, gate_errors, raw_reference, shaped_reference, track_llh)
from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.plant import course_plant

out = Path(__file__).resolve().parent.parent / "results"
out.mkdir(exist_ok=True)
mpc = ExplicitMPC(course_plant())

flights = {"course": (raw_reference(), False),          # staircase, speed target 0
           "proposed": (shaped_reference(), True)}      # shaped reference + energy speed target

for name, (ref, energy) in flights.items():
    f = fly(mpc, ref, energy)
    t = (np.arange(len(f["h"])) + 1) * DT
    lat, lon, alt_m = track_llh(f["s"], f["h"])
    table = np.column_stack([t, lat, lon, alt_m, f["theta"], f["alpha"], f["elevator_deg"]])
    np.savetxt(out / f"flight_{name}.csv", table, delimiter=",", fmt="%.9f",
               header="t_s,lat_deg,lon_deg,alt_m,theta_rad,alpha_rad,elevator_deg", comments="")
    gates = np.column_stack([np.arange(1, 6), np.interp(GATE_S, f["s"], t), gate_errors(f)])
    np.savetxt(out / f"gates_{name}.csv", gates, delimiter=",", fmt="%.6f",
               header="gate,t_cross_s,altitude_error_ft", comments="")
    print(f"{name:9s} gate errors (ft): {np.round(gate_errors(f), 1)}   saved flight_{name}.csv, gates_{name}.csv")
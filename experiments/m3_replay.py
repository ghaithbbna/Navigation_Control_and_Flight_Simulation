import time
from pathlib import Path

import numpy as np
from flightgear_python.fg_if import FDMConnection

FLIGHT = "proposed"                       # "proposed" or "course"
HEADING_RAD = np.radians(90.0)            # due east

# Module level on purpose: on Windows the callback runs in a second process
# that re-imports this file, so the data must be loaded here, not under __main__.
results = Path(__file__).resolve().parent.parent / "results"
flight = np.loadtxt(results / f"flight_{FLIGHT}.csv", delimiter=",", skiprows=1)
gates = np.loadtxt(results / f"gates_{FLIGHT}.csv", delimiter=",", skiprows=1)
T, LAT, LON, ALT, THETA, ALPHA, ELEV = flight.T
t_start, t_last, next_gate = None, 0.0, 0


def pose_at(t):
    """Interpolate the saved flight at time t (s)."""
    return [np.interp(t, T, column) for column in (LAT, LON, ALT, THETA, ALPHA, ELEV)]


def fdm_callback(fdm_data, event_pipe):
    global t_start, t_last, next_gate
    if t_start is None:
        t_start = time.time()
    t = (time.time() - t_start) % T[-1]               # real time, looping
    if t < t_last:
        next_gate = 0                                 # a new lap started
    t_last = t
    while next_gate < len(gates) and t >= gates[next_gate, 1]:
        g = gates[next_gate]
        print(f"gate {int(g[0])}: altitude error {g[2]:+.1f} ft  (tolerance +-50 ft)")
        next_gate += 1

    lat, lon, alt, theta, alpha, elev = pose_at(t)
    fdm_data.lat_rad = np.radians(lat)
    fdm_data.lon_rad = np.radians(lon)
    fdm_data.alt_m = alt
    fdm_data.phi_rad = 0.0
    fdm_data.theta_rad = theta
    fdm_data.psi_rad = HEADING_RAD
    fdm_data.alpha_rad = alpha
    fdm_data.elevator = elev / 25.0                   # -1..1 (cosmetic)
    return fdm_data


if __name__ == "__main__":                            # REQUIRED on Windows
    fdm_conn = FDMConnection(fdm_version=24)
    fdm_conn.connect_rx("localhost", 5501, fdm_callback)
    fdm_conn.connect_tx("localhost", 5502)
    fdm_conn.start()
    while True:
        time.sleep(0.025)
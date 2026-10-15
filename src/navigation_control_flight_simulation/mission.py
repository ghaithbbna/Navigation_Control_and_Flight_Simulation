"""The altitude-gate mission: route, gates, altitude references and the closed-loop flight."""

import numpy as np
from dataclasses import replace
from navigation_control_flight_simulation.geodesy import ned_to_llh
from navigation_control_flight_simulation.units import G0_FT, M_PER_FT

ORIGIN = (48.1374, 11.5755, 3048.0)                     # Marienplatz, 10 000 ft (lat, lon, h in m)
HEADING_DEG = 90.0                                      # due east
V0 = 500.0                                              # ft/s
GATE_T = np.array([8.0, 20.0, 32.0, 44.0, 54.0])        # s
GATE_DH = np.array([100.0, 20.0, 160.0, 60.0, 140.0])   # ft above the start altitude
GATE_S = V0 * GATE_T                                    # ft along the track
TOLERANCE_FT = 50.0                                     # frozen in docs/REPORT.md
DT, T_END = 0.01, 60.0
T = np.arange(int(T_END / DT)) * DT


def track_llh(s_ft, dh_ft):
    """Along-track distance (ft) and altitude deviation (ft) -> (lat, lon, h in m).

    Horizontal position comes from the NED plane; the height is taken straight
    from the altitude, so the Earth's curvature does not contaminate it.
    """
    psi = np.radians(HEADING_DEG)
    s = np.atleast_1d(s_ft) * M_PER_FT
    lat, lon, _ = np.array([ned_to_llh([x * np.cos(psi), x * np.sin(psi), 0.0], ORIGIN)
                            for x in s]).T
    return lat, lon, ORIGIN[2] + np.atleast_1d(dh_ft) * M_PER_FT


def raw_reference(t=T):
    """Staircase: right after each gate, command the next gate's altitude."""
    r = np.zeros_like(t)
    for start, dh in zip(np.concatenate([[0.0], GATE_T[:-1]]), GATE_DH):
        r[t >= start] = dh
    return r

def linear_reference(t=T):
    """Straight lines between the gates; the last altitude is held."""
    return np.interp(t, np.concatenate([[0.0], GATE_T]), np.concatenate([[0.0], GATE_DH]))


def shifted(ref, delay):
    """The same reference `delay` seconds later (negative: earlier). The ends are held."""
    return np.interp(T - delay, T, ref)

def shaped_reference(t=T, step=0.1):
    """Smoothest path through the gates: minimise the squared second difference
    of the altitude, subject to passing exactly through each gate (equality QP)."""
    tg = np.arange(0.0, t[-1] + step / 2, step)
    N = len(tg)
    D = np.zeros((N - 2, N))
    for i in range(N - 2):
        D[i, i:i + 3] = [1.0, -2.0, 1.0]
    idx = np.concatenate([[0], np.round(GATE_T / step).astype(int)])
    val = np.concatenate([[0.0], GATE_DH])
    C = np.zeros((len(idx), N))
    C[np.arange(len(idx)), idx] = 1.0
    KKT = np.block([[2 * D.T @ D + 1e-9 * np.eye(N), C.T],
                    [C, np.zeros((len(idx), len(idx)))]])
    z = np.linalg.solve(KKT, np.concatenate([np.zeros(N), val]))[:N]
    z[idx[-1]:] = GATE_DH[-1]                           # hold the last altitude
    return np.interp(t, tg, z)

def make_world(seed, sigma_w=5.0, tau=3.0, elevator_error=0.2):
    """One random world: a gust and an elevator that is not exactly what the controller believes.

    Gust: vertical wind w (ft/s), first-order Gauss-Markov (rms sigma_w, correlation time tau, s),
    entering as an angle-of-attack disturbance alpha_g = w / V0, i.e. the aerodynamics see alpha + alpha_g.
    Elevator effectiveness: the real elevator column of B is scaled by a uniform factor in 1 +- elevator_error.
    """
    rng = np.random.default_rng(seed)
    a = np.exp(-DT / tau)
    noise = rng.standard_normal(len(T))
    w = np.zeros(len(T))
    w[0] = sigma_w * noise[0]
    for k in range(1, len(T)):
        w[k] = a * w[k - 1] + np.sqrt(1 - a * a) * sigma_w * noise[k]
    return dict(alpha_g=w / V0, elevator_gain=rng.uniform(1 - elevator_error, 1 + elevator_error))


def _model(plant, world):
    """Phi, Gamma, gust input column and gust sequence of the plant that really flies."""
    if world is None:
        Phi, Gam = plant.discretize(DT)
        return Phi, Gam, np.zeros(plant.n), np.zeros(len(T))
    B = plant.B.copy()
    B[:, 1] *= world["elevator_gain"]
    Phi, Gam = replace(plant, B=B).discretize(DT)
    Gg = replace(plant, B=plant.A[:, [3]]).discretize(DT)[1][:, 0]       # alpha_g enters like A[:, alpha]
    return Phi, Gam, Gg, world["alpha_g"]

def _slew(u_cmd, u_last, rate_deg_s):
    """The elevator (second input) may change by at most rate_deg_s * DT per plant step; None means no limit."""
    if rate_deg_s is None:
        return u_cmd
    step = np.deg2rad(rate_deg_s) * DT
    u = np.array(u_cmd, dtype=float)
    u[1] = np.clip(u[1], u_last[1] - step, u_last[1] + step)
    return u


def fly(mpc, href, energy, Np=3.5, world=None, rate_deg_s=None):
    """Close the loop: controller `mpc` flies the plant against the altitude
    reference `href` (ft, one value per time step).

    energy=True also sends the speed target dV = -(g/V0) dh; False sends 0.
    world (from make_world) adds a gust and an elevator error to the plant that flies; the controller does not know.
    """
    plant = mpc.plant
    Phi, Gam, Gg, ag = _model(plant, world)
    ahead = int(Np / DT)                                # the controller looks Np seconds ahead
    x, u = np.zeros(plant.n), np.zeros(plant.m)
    X, U, limited = [], [], []
    for k in range(len(href)):
        h_ref = href[min(k + ahead, len(href) - 1)]
        V_ref = -(G0_FT / V0) * h_ref if energy else 0.0
        u = _slew(mpc.control(x, h_ref, V_ref), u, rate_deg_s)
        limited.append(bool(np.any(mpc.M @ u >= mpc.g - 1e-6)))
        x = Phi @ x + Gam @ u + Gg * ag[k]
        X.append(x)
        U.append(u)
    X, U = np.array(X), np.array(U)
    return dict(h=X[:, 0],                              # altitude deviation, ft
                s=np.cumsum(V0 + X[:, 2]) * DT,         # distance flown, ft
                elevator_deg=np.degrees(U[:, 1] + plant.u_trim[1]),
                theta=X[:, 1] + plant.x_trim[1],        # absolute pitch, rad
                alpha=X[:, 3] + plant.x_trim[3],        # absolute angle of attack, rad
                limited=np.mean(limited))


def fly_c1(mpc, href, energy, Ts=0.1, preview=True, world=None, rate_deg_s=None):
    """Close the loop with the incremental MPC (C1), which decides every Ts seconds.

    C1 sees the reference over its whole horizon (preview=True) or only its last value (preview=False).
    The command is held between decisions; the plant still steps at DT.
    rate_deg_s limits how fast the elevator can actually move (None: instantly), in both flight functions.
    """
    plant = mpc.plant
    Phi, Gam, Gg, ag = _model(plant, world)
    every = int(round(Ts / DT))
    x, u = np.zeros(plant.n), np.zeros(plant.m)
    cmd = u
    X, U = [], []
    for k in range(len(href)):
        if k % every == 0:
            h = href[np.minimum(k + every * np.arange(1, mpc.N + 1), len(href) - 1)]
            if not preview:
                h = np.full(mpc.N, h[-1])
            cmd = mpc.control(x, u, h, -(G0_FT / V0) * h if energy else np.zeros(mpc.N))
        u = _slew(cmd, u, rate_deg_s)
        x = Phi @ x + Gam @ u + Gg * ag[k]
        X.append(x)
        U.append(u)
    X, U = np.array(X), np.array(U)
    return dict(h=X[:, 0], s=np.cumsum(V0 + X[:, 2]) * DT,
                elevator_deg=np.degrees(U[:, 1] + plant.u_trim[1]),
                theta=X[:, 1] + plant.x_trim[1], alpha=X[:, 3] + plant.x_trim[3])
    
def gate_errors(flight):
    """Altitude error (ft) at the moment the aircraft reaches each gate's distance."""
    return np.interp(GATE_S, flight["s"], flight["h"]) - GATE_DH
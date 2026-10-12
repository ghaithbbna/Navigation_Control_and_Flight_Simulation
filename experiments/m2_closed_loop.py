import time

import matplotlib.pyplot as plt
import numpy as np

from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.plant import course_plant
from navigation_control_flight_simulation.units import G0_FT

plant = course_plant()
mpc = ExplicitMPC(plant)
dt, T = 0.01, 60.0
Phi, Gam = plant.discretize(dt)
n_steps = int(T / dt)

H_REF = 100.0                                   # climb, ft
V_energy = -(G0_FT / 500.0) * H_REF             # speed change that pays for the climb, ft/s


def run(V_ref):
    x = np.zeros(plant.n)
    X, U, sat, t_ctrl = [], [], [], []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        u = mpc.control(x, H_REF, V_ref)
        t_ctrl.append(time.perf_counter() - t0)
        sat.append(bool(np.any(mpc.M @ u >= mpc.g - 1e-6)))     # some limit is active
        x = Phi @ x + Gam @ u
        X.append(x.copy())
        U.append(u.copy())
    return np.array(X), np.array(U), np.array(sat), np.array(t_ctrl)


t = np.arange(1, n_steps + 1) * dt
cases = {"speed ref = 0": 0.0, f"speed ref = {V_energy:.2f} ft/s (energy)": V_energy}

fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for name, V_ref in cases.items():
    X, U, sat, t_ctrl = run(V_ref)
    err = H_REF - X[:, 0]
    print(f"\n{name}")
    print(f"  altitude at 60 s     : {X[-1, 0]:7.1f} ft   (target {H_REF:.0f})")
    print(f"  peak |h error| after 10 s : {np.abs(err[int(10 / dt):]).max():7.1f} ft")
    print(f"  speed change at 60 s : {X[-1, 2]:7.2f} ft/s")
    print(f"  steps with a limit active : {100 * sat.mean():5.1f} %")
    print(f"  controller time per step  : mean {1e3 * t_ctrl.mean():.2f} ms, max {1e3 * t_ctrl.max():.1f} ms")
    ax[0].plot(t, X[:, 0], label=name)
    ax[1].plot(t, X[:, 2], label=name)
    ax[2].plot(t, np.degrees(U[:, 1] + plant.u_trim[1]), label=name)

ax[0].axhline(H_REF, color="k", ls="--", lw=0.8); ax[0].set_ylabel("altitude change (ft)")
ax[1].set_ylabel("speed change (ft/s)")
ax[2].axhline(25, color="r", ls=":"); ax[2].axhline(-25, color="r", ls=":")
ax[2].set_ylabel("elevator command (deg)"); ax[2].set_xlabel("time (s)")
ax[0].legend()
plt.tight_layout()
plt.savefig("results/m2_closed_loop.png", dpi=130)
plt.show()
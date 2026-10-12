import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from navigation_control_flight_simulation.plant import course_plant
from navigation_control_flight_simulation.units import G0_FT

os.makedirs("results", exist_ok=True) # creates the results folder for the figures

plant = course_plant() # loads the model from M1

# One eigenvalue per conjugate pair, sorted from most negative to most positive real part.
modes = sorted([z for z in plant.eigenvalues() if z.imag >= 0], key=lambda z: z.real)

print("\n Test 1: natural modes")
print(f"{'eigenvalue':>22} {'wn [rad/s]':>11} {'zeta':>8} {'period [s]':>11} {'tau [s]':>10}")
for lam in modes:
    wn = abs(lam)                      # natural frequency
    tau = 1.0 / abs(lam.real)          # time constant (decay or growth)
    if lam.imag > 0:                   # oscillating mode
        zeta = -lam.real / wn
        period = 2 * np.pi / lam.imag
        text = f"{lam.real:+.4f}{lam.imag:+.4f}j"
        print(f"{text:>22} {wn:11.4f} {zeta:8.3f} {period:11.2f} {tau:10.2f}")
    else:                              # real mode: no oscillation
        text = f"{lam.real:+.4f}"
        print(f"{text:>22} {wn:11.4f} {'-':>8} {'-':>11} {tau:10.2f}")

print("\nTest 2: Hautus margins (smallest singular value, bigger = stronger)")
print(f"{'eigenvalue':>22} {'controllability':>16} {'observability':>14}")
I = np.eye(plant.n)
for lam in modes:
    ctrl_matrix = np.hstack([lam * I - plant.A, plant.B])    # 7 x 9
    obs_matrix = np.vstack([lam * I - plant.A, plant.C])     # 12 x 7
    sv_c = np.linalg.svd(ctrl_matrix, compute_uv=False)[-1]
    sv_o = np.linalg.svd(obs_matrix, compute_uv=False)[-1]
    text = f"{lam.real:+.4f}{lam.imag:+.4f}j" if lam.imag > 0 else f"{lam.real:+.4f}"
    print(f"{text:>22} {sv_c:16.3g} {sv_o:14.3g}")

print("\nTest 3: eigenvalue plot")
eigs = plant.eigenvalues()
fig, ax = plt.subplots(figsize=(8, 5))
ax.axvspan(0, 1, color="red", alpha=0.08)            # unstable half-plane
ax.axvline(0, color="black", linewidth=1)
ax.plot(eigs.real, eigs.imag, "o", markersize=8)
for lam in eigs:
    ax.annotate(f"{lam.real:.3g}", (lam.real, lam.imag),
                textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xscale("symlog", linthresh=0.01)
ax.set_xlim(-60, 0.5)
ax.set_xlabel("real part  [1/s]   (symlog axis)")
ax.set_ylabel("imaginary part  [rad/s]")
ax.set_title("Eigenvalues of A")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/m1_modes.png", dpi=150)

dt = 0.01
T = 200.0
N = int(T / dt)
Phi, Gam = plant.discretize(dt)
u = np.array([0.0, -0.05])           # elevator command step of -0.05 rad

x = np.zeros(plant.n)                # deviation state, starts at trim
t = np.arange(N + 1) * dt
X = np.zeros((N + 1, plant.n))
for k in range(N):
    x = Phi @ x + Gam @ u
    X[k + 1] = x

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(t, X[:, 0])
ax1.set_ylabel("altitude deviation [ft]")
ax1.set_title("Elevator step of -0.05 rad")
ax1.grid(alpha=0.3)
ax2.plot(t, X[:, 2])
ax2.set_ylabel("speed deviation [ft/s]")
ax2.set_xlabel("time [s]")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/m1_step_response.png", dpi=150)

peaks, _ = find_peaks(X[:, 0])
print("\nTest 4: altitude peaks at t =", np.round(t[peaks], 1), "s")
print("time between peaks              =", np.round(np.diff(t[peaks]), 1), "s")
print(f"altitude deviation at t = 30 s  = {X[3000, 0]:+.1f} ft")
print(f"speed deviation at t = 30 s     = {X[3000, 2]:+.1f} ft/s")

energy_residual = X[:, 0] + (500.0 / G0_FT) * X[:, 2]      # dh + (V0/g) dV
print("\nTest 5: energy exchange during the elevator step")
print(f"largest |dh|              = {np.abs(X[:, 0]).max():.1f} ft")
print(f"largest |dh + (V0/g) dV|  = {np.abs(energy_residual).max():.1f} ft")

xa = plant.x_trim.copy()
for k in range(1000):                                      # 10 s
    xa = Phi @ xa + Gam @ plant.u_trim
print("\nTest 6: absolute state fed to the deviation model, after 10 s")
print(f"altitude {xa[0]:.1f} ft, speed {xa[2]:.1f} ft/s, pitch {np.degrees(xa[1]):.1f} deg")

plt.show()
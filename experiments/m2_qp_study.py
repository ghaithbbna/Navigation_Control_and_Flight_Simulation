import os
import numpy as np
import matplotlib.pyplot as plt
from navigation_control_flight_simulation.qp import solve_pqp, solve_exact, cost

os.makedirs("results", exist_ok=True)
# The elevator toy: J(u) = (20 - 2u)^2 + 4u^2 = 8u^2 - 80u + 400, limit u <= 3
r = solve_pqp([[1 / 8]], [-40.0], [[1.0]], [3.0])
u_best, price = r.U[0], r.lam[0]
J = lambda u: 8 * u**2 - 80 * u + 400

u = np.linspace(0, 8, 400)
fig, ax = plt.subplots(figsize=(8, 5))
ax.axvspan(3, 8, color="red", alpha=0.08, label="forbidden by the limit u <= 3")
ax.plot(u, J(u), linewidth=2, label="cost J(u)")
ut = np.linspace(1.8, 4.2, 10)
ax.plot(ut, J(u_best) - price * (ut - u_best), "--", color="black",
        label=f"slope at the limit = -lambda = {-price:.0f}")
ax.plot(5, J(5), "s", markersize=9, label=f"free minimum: u = 5, J = {J(5):.0f}")
ax.plot(u_best, J(u_best), "o", markersize=9, label=f"best allowed: u = {u_best:.0f}, J = {J(u_best):.0f}")
ax.set_xlabel("elevator command u")
ax.set_ylabel("cost J")
ax.set_title("Toy problem: the limit costs 32 per unit of u")
ax.set_ylim(150, 420)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/m2_qp_toy.png", dpi=150)
print(f"Toy: best allowed u = {u_best:.3f}, price lambda = {price:.3f}")

# Example B: watch PQP walk to the answer
H = np.array([[2.0, 0.5], [0.5, 1.0]])
f = np.array([-2.0, -3.0])
M = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
g = np.array([1.5, 1.0, 2.0])
H_inv = np.linalg.inv(H)

history = []
def record(k, lam):
    history.append(lam)

res = solve_pqp(H_inv, f, M, g, callback=record)
exact = solve_exact(H, f, M, g)
lam_hist = np.array(history)                       # one row per pass (row 0 = start)
U_hist = -(f + 0.5 * lam_hist @ M) @ H_inv.T       # primal point after each pass

print(f"\nExample B: PQP stopped after {res.iterations} passes, converged = {res.converged}")
print("  PQP   U =", np.round(res.U, 6), " lambda =", np.round(res.lam, 6))
print("  exact U =", np.round(exact.U, 6), " lambda =", np.round(exact.lam, 6))
print(f"  constraint violation at exit = {res.max_violation:.2e}")

x1, x2 = np.meshgrid(np.linspace(-0.3, 1.9, 300), np.linspace(0.3, 3.3, 300))
Jgrid = H[0, 0] * x1**2 + 2 * H[0, 1] * x1 * x2 + H[1, 1] * x2**2 + 2 * f[0] * x1 + 2 * f[1] * x2
feasible = np.ones_like(x1, dtype=bool)
for row, limit in zip(M, g):
    feasible &= (row[0] * x1 + row[1] * x2 <= limit)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.contour(x1, x2, Jgrid, levels=15, colors="gray", linewidths=0.7)
ax1.contourf(x1, x2, feasible, levels=[0.5, 1.5], colors=["#9ecae1"], alpha=0.5)
ax1.plot([1.5, 1.5], [0.3, 3.3], "k-", linewidth=1)
ax1.plot([-0.3, 1.9], [1, 1], "k-", linewidth=1)
ax1.plot([-0.3, 1.9], [2.3, 0.1], "k-", linewidth=1)
ax1.plot(U_hist[:, 0], U_hist[:, 1], ".-", color="tab:red", label="PQP after each pass")
ax1.plot(*U_hist[0], "o", color="tab:red", markersize=9)
ax1.annotate("pass 0", U_hist[0], textcoords="offset points", xytext=(8, 4))
unc = -np.linalg.solve(H, f)
ax1.plot(*unc, "s", color="tab:green", markersize=9, label="free minimum (ignores limits)")
ax1.plot(*exact.U, "*", color="black", markersize=14, label="exact constrained minimum")
ax1.set_xlim(-0.3, 1.9)
ax1.set_ylim(0.3, 3.3)
ax1.set_xlabel("U1")
ax1.set_ylabel("U2")
ax1.set_title("Example B: blue = allowed region, grey = lines of equal cost")
ax1.legend(loc="upper right", fontsize=8)

passes = np.arange(len(lam_hist))
for i, label in enumerate(["limit 1: U1 <= 1.5", "limit 2: U2 <= 1", "limit 3: U1+U2 <= 2"]):
    ax2.semilogy(passes, lam_hist[:, i], label=label)
ax2.axhline(3.25, color="gray", linestyle="--", linewidth=1)
ax2.text(passes[-1], 3.4, "exact lambda2 = 3.25", ha="right", fontsize=8)
ax2.set_xlabel("pass of the update")
ax2.set_ylabel("multiplier lambda (log scale)")
ax2.set_title("Inactive limits shrink toward 0 but never reach it")
ax2.grid(alpha=0.3, which="both")
ax2.legend()
fig.tight_layout()
fig.savefig("results/m2_qp_example_B.png", dpi=150)

plt.show()
"""
Unknowns: command changes dU at the block starts.  u(step i) = u_prev + sum of dU with start <= i.
    H = Gb' Q Gb + R,   f = -Gb' Q E,   E = r - F x - G (1 x u_prev)
The limits are boxes on the absolute block commands z, so each decision is a bounded least squares.
Deviations from trim, like C0.  With N = 1, Ts = Np and one block, C1 is C0.
"""

import numpy as np
from scipy.optimize import lsq_linear

class IncrementalMPC:
    def __init__(self, plant, Ts=0.1, N=35, blocks=(0, 1, 2, 4, 8, 16), Q=None, R=None,
                 elevator_limit_deg=25.0):
        self.plant, self.N, self.nb = plant, N, len(blocks)
        Q = np.diag([1.0, 1.0, 49.0, 1.0, 1.0]) if Q is None else Q
        R = np.diag([20.0, 50.0]) if R is None else R
        n, m, p = plant.n, plant.m, plant.p
        self.m = m

        Phi, Gam = plant.discretize(Ts)
        F = np.zeros((N * p, n))
        G = np.zeros((N * p, N * m))                     # u_0 .. u_{N-1}  ->  y_1 .. y_N
        Phi_j = np.eye(n)
        for j in range(N):
            Phi_j = Phi @ Phi_j                          # Phi^(j+1)
            F[j * p:(j + 1) * p] = plant.C @ Phi_j
            for i in range(j + 1):
                G[j * p:(j + 1) * p, i * m:(i + 1) * m] = plant.C @ np.linalg.matrix_power(Phi, j - i) @ Gam
        T = (np.arange(N)[:, None] >= np.asarray(blocks)[None, :]).astype(float)   # step i uses block b
        self.F = F
        self.G_hold = G @ np.kron(np.ones((N, 1)), np.eye(m))                       # effect of u_prev
        Gb = G @ np.kron(T, np.eye(m))

        Qbar, Rbar = np.kron(np.eye(N), Q), np.kron(np.eye(self.nb), R)
        self.H = Gb.T @ Qbar @ Gb + Rbar
        self.Gb_Q = Gb.T @ Qbar

        el = np.deg2rad(elevator_limit_deg)              # real limits, shifted by the trim
        self.u_min = np.array([0.0, -el]) - plant.u_trim
        self.u_max = np.array([1.0, el]) - plant.u_trim
        self.lo, self.hi = np.tile(self.u_min, self.nb), np.tile(self.u_max, self.nb)

        self.Lc = np.linalg.cholesky(self.H)             # H = Lc Lc'
        self.A = self.Lc.T @ np.kron(np.eye(self.nb) - np.eye(self.nb, k=-1), np.eye(m))   # dU = D (z - z0)

    def control(self, x, u_prev, h_ref, V_ref):
        """New command u (deviation). h_ref, V_ref: N values each (steps 1..N ahead)."""
        r = np.zeros((self.N, 5))
        r[:, 0] = h_ref
        r[:, 2] = self.plant.C[2, 2] * np.asarray(V_ref)
        E = (r.ravel() - self.F @ x - self.G_hold @ u_prev).reshape(self.N, 5)
        E[:, [1, 3, 4]] = 0.0                            # only h and -7V are tracked
        f = -self.Gb_Q @ E.ravel()

        b = self.A @ np.tile(u_prev, self.nb) - np.linalg.solve(self.Lc, f)
        z = lsq_linear(self.A, b, bounds=(self.lo, self.hi), method="bvls").x
        return z[:self.m]
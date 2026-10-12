"""C0: explicit MPC. Prediction y = F x + G u at the horizon end Np.

    H = G'QG + R,   f = -G'QE,   K = H^-1 G'Q,   E = r - F x
    u = K E                      if it respects the limits
    u = QP(H, f, M, g)           otherwise
All quantities are deviations from the trim point.
"""

import numpy as np

from navigation_control_flight_simulation.qp import solve_pqp


class ExplicitMPC:
    def __init__(self, plant, Np=3.5, Q=None, R=None, elevator_limit_deg=25.0):
        self.plant = plant
        self.Q = np.diag([1.0, 1.0, 49.0, 1.0, 1.0]) if Q is None else Q
        self.R = np.diag([20.0, 50.0]) if R is None else R

        Phi, Gamma = plant.discretize(Np)
        self.F = plant.C @ Phi
        self.G = plant.C @ Gamma
        self.H = self.G.T @ self.Q @ self.G + self.R
        self.H_inv = np.linalg.inv(self.H)
        self.K = self.H_inv @ self.G.T @ self.Q

        # real limits: thrust in [0, 1], elevator in [-25, 25] deg,
        # shifted by the trim because u is a deviation
        el = np.deg2rad(elevator_limit_deg)
        lower = np.array([0.0, -el]) - plant.u_trim
        upper = np.array([1.0, el]) - plant.u_trim
        self.M = np.vstack([np.eye(2), -np.eye(2)])
        self.g = np.concatenate([upper, -lower])

    def control(self, x, h_ref, V_ref):
        """Command u (deviation) for state x and targets h_ref (ft), V_ref (ft/s)."""
        r = np.zeros(5)
        r[0] = h_ref
        r[2] = self.plant.C[2, 2] * V_ref           # -7 * V
        E = np.zeros(5)
        E[[0, 2]] = (r - self.F @ x)[[0, 2]]

        u = self.K @ E
        if np.all(self.M @ u <= self.g):
            return u
        f = -self.G.T @ self.Q @ E
        return solve_pqp(self.H_inv, f, self.M, self.g).U
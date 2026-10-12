"""Longitudinal plant model (7 states, 2 inputs, 5 outputs)

Model (deviation form around the operating point)
-------------------------------------------------
    x~_dot = A x~ + B u~          y~ = C x~
    x = x_trim + x~               u = u_trim + u~

States   : [h (ft), theta (rad), V (ft/s), alpha (rad), q (rad/s), thrust actuator, elevator actuator (rad)]
Inputs   : [thrust command, elevator command (rad)]
Outputs  : [h, theta, -7*V, alpha, q]       (note the -7 scaling on speed)
The A, B, C numbers come from an online course.

A @ x_trim + B @ u_trim = 0 
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

STATE_NAMES = ("h", "theta", "V", "alpha", "q", "thrust_act", "elev_act")
INPUT_NAMES = ("thrust", "elevator")
OUTPUT_NAMES = ("h", "theta", "neg7V", "alpha", "q")

@dataclass(frozen=True)
class LongitudinalPlant:
    A: np.ndarray 
    B: np.ndarray
    C: np.ndarray
    x_trim: np.ndarray
    u_trim: np.ndarray

    # ---- sizes -----------------------------------------------------------
    @property
    def n(self):
        return self.A.shape[0]

    @property
    def m(self):
        return self.B.shape[1]

    @property
    def p(self):
        return self.C.shape[0]

    # ---- model variants --------------------------------------------------
    def with_thrust_scale(self, kappa):
        """Return a copy whose thrust column of B is multiplied by kappa.

        kappa = 1 is the course model.
        """
        B = self.B.copy()
        B[:, 0] *= kappa
        return replace(self, B=B)

    # ---- analysis --------------------------------------------------------
    def eigenvalues(self):
        return np.linalg.eigvals(self.A)

    def discretize(self, dt):
        """Zero-order-hold discretisation: x_{k+1} = Phi x_k + Gamma u_k.

        Uses the block-matrix exponential
            expm([[A, B], [0, 0]] * dt) = [[Phi, Gamma], [0, I]]
        """
        n, m = self.n, self.m
        M = np.zeros((n + m, n + m))
        M[:n, :n] = self.A
        M[:n, n:] = self.B
        E = expm(M * dt)
        return E[:n, :n], E[:n, n:]

    # ---- coordinates -----------------------------------------------------
    def to_absolute(self, x_dev):
        return self.x_trim + np.asarray(x_dev, dtype=float)

    def to_deviation(self, x_abs):
        return np.asarray(x_abs, dtype=float) - self.x_trim


def course_plant():
    A = np.array([
        [0,      500.0,  0,       -500.0,   0,       0,      0],
        [0,      0,      0,       0,        1,       0,      0],
        [0.0001, -32.17, -0.013,  -2.9483,  -1.0283, 0.0016, 0.1018],
        [0,      0,      -0.0003, -0.7506,  0.9281,  0.0,    -0.0016],
        [0,      0,      0,       -1.8365,  -1.0271, 0,      -0.1335],
        [0,      0,      0,       0,        0,       -1,     0],
        [0,      0,      0,       0,        0,       0,      -20.2],
    ])
    B = np.zeros((7, 2))
    B[5, 0] = 1.0
    B[6, 1] = 20.2
    C = np.zeros((5, 7))
    C[0, 0] = 1.0     # h
    C[1, 1] = 1.0     # theta
    C[2, 2] = -7.0    # -7 * V
    C[3, 3] = 1.0     # alpha
    C[4, 4] = 1.0     # q
    x_trim = np.array([10000.0, 0.0638, 500.0, 0.0638, 0.0, 0.06, -0.0393])
    u_trim = np.array([0.06, -0.0393])
    return LongitudinalPlant(A=A, B=B, C=C, x_trim=x_trim, u_trim=u_trim)

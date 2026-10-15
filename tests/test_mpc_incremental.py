import numpy as np
from scipy.optimize import minimize

from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.mpc_incremental import IncrementalMPC
from navigation_control_flight_simulation.plant import course_plant

plant = course_plant()
x0, u0 = np.zeros(plant.n), np.zeros(plant.m)
c1 = IncrementalMPC(plant)


def test_c0_is_the_special_case():
    one = IncrementalMPC(plant, Ts=3.5, N=1, blocks=(0,))
    for h, v in [(1.0, -0.064), (100.0, -6.43)]:       # a small request, and one that hits the limits
        assert np.allclose(one.control(x0, u0, [h], [v]), ExplicitMPC(plant).control(x0, h, v), atol=1e-4)


def test_nothing_to_do_means_no_command():
    assert np.allclose(c1.control(x0, u0, np.zeros(35), np.zeros(35)), 0.0)


def test_big_climb_respects_limits_and_is_optimal():
    h, v = np.full(35, 100.0), np.full(35, -6.43)
    u = c1.control(x0, u0, h, v)
    assert np.all(u >= c1.u_min - 1e-9) and np.all(u <= c1.u_max + 1e-9)
    E = np.zeros((35, 5)); E[:, 0], E[:, 2] = h, 7 * 6.43
    E = (E.ravel() - c1.F @ x0 - c1.G_hold @ u0).reshape(35, 5)
    E[:, [1, 3, 4]] = 0.0
    f = -c1.Gb_Q @ E.ravel()
    D = np.kron(np.eye(6) - np.eye(6, k=-1), np.eye(2))             # dU = D z  (u_prev = 0)
    ref = minimize(lambda z: (D @ z) @ c1.H @ (D @ z) + 2 * f @ (D @ z), np.zeros(12),
                   bounds=list(zip(c1.lo, c1.hi)), method="L-BFGS-B",
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000, "maxfun": 100000}).x
    assert np.allclose(u, ref[:2], atol=1e-3)
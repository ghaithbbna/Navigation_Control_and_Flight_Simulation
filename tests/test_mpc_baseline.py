import numpy as np
import pytest
from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.plant import course_plant
from navigation_control_flight_simulation.qp import solve_exact

plant = course_plant()
mpc = ExplicitMPC(plant)
x0 = np.zeros(plant.n)


def test_G_matches_course_formula():
    Phi, _ = plant.discretize(3.5)
    G_course = plant.C @ np.linalg.inv(plant.A) @ (Phi - np.eye(plant.n)) @ plant.B
    assert np.allclose(mpc.G, G_course, atol=1e-8)


def test_small_request_returns_K_times_E():
    u = mpc.control(x0, h_ref=1.0, V_ref=-0.064)
    E = np.zeros(5)
    E[0], E[2] = 1.0, 7 * 0.064
    assert np.allclose(u, mpc.K @ E)


def test_big_request_respects_limits_and_is_optimal():
    u = mpc.control(x0, h_ref=100.0, V_ref=-6.43)       # needs ~ -38 deg
    assert np.all(mpc.M @ u <= mpc.g + 1e-6)
    E = np.zeros(5)
    E[0], E[2] = 100.0, 7 * 6.43
    f = -mpc.G.T @ mpc.Q @ E
    assert np.allclose(u, solve_exact(mpc.H, f, mpc.M, mpc.g).U, atol=1e-4)
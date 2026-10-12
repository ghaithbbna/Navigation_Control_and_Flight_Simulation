import numpy as np
import pytest
from navigation_control_flight_simulation.qp import solve_pqp, solve_exact, cost

# Worked example A: only the second limit is active.
H_A = np.array([[2.0, 0.0], [0.0, 1.0]])
F_A = np.array([-2.0, -3.0])
M_A = np.eye(2)
G_A = np.array([5.0, 1.0])

# Worked example B: three limits, only the second is active.
H_B = np.array([[2.0, 0.5], [0.5, 1.0]])
F_B = np.array([-2.0, -3.0])
M_B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
G_B = np.array([1.5, 1.0, 2.0])


def pqp(H, f, M, g, **kw):
    return solve_pqp(np.linalg.inv(H), f, M, g, **kw)

# pin the referee against answers worked out by hand
def test_exact_example_A():
    r = solve_exact(H_A, F_A, M_A, G_A)
    assert np.allclose(r.U, [1.0, 1.0], atol=1e-12)
    assert np.allclose(r.lam, [0.0, 4.0], atol=1e-12)

def test_exact_example_B():
    # by hand: with U2 = 1 the cost is 2*U1**2 - 3*U1 - 5, minimal at U1 = 3/4
    r = solve_exact(H_B, F_B, M_B, G_B)
    assert np.allclose(r.U, [0.75, 1.0], atol=1e-12)
    assert np.allclose(r.lam, [0.0, 3.25, 0.0], atol=1e-12)

# compare PQP to those hand answers. The tolerance for A is tight (1e-6, it converges in 2 passes), and for B it is looser (1e-3) because PQP leaves a small leftover error.
def test_pqp_example_A():
    r = pqp(H_A, F_A, M_A, G_A)
    assert r.converged
    assert np.allclose(r.U, [1.0, 1.0], atol=1e-6)
    assert np.allclose(r.lam, [0.0, 4.0], atol=1e-6)

def test_pqp_example_B():
    r = pqp(H_B, F_B, M_B, G_B)
    assert r.converged
    assert np.allclose(r.U, [0.75, 1.0], atol=1e-3)
    assert r.lam[1] == pytest.approx(3.25, abs=1e-3)
    assert r.max_violation < 1e-4          # slightly infeasible at exit: documented

# If no limit is near, the constrained answer must equal the unconstrained
def test_loose_limits_give_unconstrained_solution():
    g_loose = np.array([100.0, 100.0, 100.0])
    unconstrained = -np.linalg.solve(H_B, F_B)
    r = pqp(H_B, F_B, M_B, g_loose)
    assert np.allclose(r.U, unconstrained, atol=1e-6)
    assert np.allclose(solve_exact(H_B, F_B, M_B, g_loose).U, unconstrained, atol=1e-12)

def test_single_constraint_works():
    # the course code crashes here (np.squeeze makes a 0-d array)
    r = pqp(H_B, F_B, [0.0, 1.0], [1.0])
    e = solve_exact(H_B, F_B, [0.0, 1.0], [1.0])
    assert np.allclose(r.U, e.U, atol=1e-6)
    assert np.allclose(e.U, [0.75, 1.0], atol=1e-12)


def test_elevator_toy_example():
    # J(u) = (20 - 2u)^2 + 4u^2  ->  H = 8, f = -40
    H = np.array([[8.0]]); f = np.array([-40.0]); M = np.array([[1.0]])
    tight = pqp(H, f, M, [3.0])
    assert tight.U[0] == pytest.approx(3.0, abs=1e-9)
    assert tight.lam[0] == pytest.approx(32.0, abs=1e-6)       # the "price" of the limit
    loose = pqp(H, f, M, [8.0])
    assert loose.U[0] == pytest.approx(5.0, abs=1e-9)
    assert loose.lam[0] == 0.0                                  # limit not touched


def _random_problem(rng, n=2, m=4):
    A = rng.normal(size=(n, n))
    H = A.T @ A + 0.5 * np.eye(n)
    f = 3.0 * rng.normal(size=n)
    M = rng.normal(size=(m, n))
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    U0 = rng.normal(size=n)
    g = M @ U0 + rng.uniform(0.05, 2.0, size=m)      # feasible by construction
    return H, f, M, g


def test_random_problems_against_exact():
    rng = np.random.default_rng(0)
    worst_dU, worst_viol = 0.0, 0.0
    for _ in range(200):
        H, f, M, g = _random_problem(rng)
        r = pqp(H, f, M, g)
        e = solve_exact(H, f, M, g)
        assert r.converged
        assert np.all(r.lam >= 0.0)
        worst_dU = max(worst_dU, np.abs(r.U - e.U).max())
        worst_viol = max(worst_viol, r.max_violation)
    assert worst_dU < 5e-3
    assert worst_viol < 1e-3


def test_iteration_cap():
    r = pqp(H_B, F_B, M_B, G_B, max_iter=1)
    assert r.iterations == 1
    assert not r.converged


def test_inconsistent_constraints_do_not_converge():
    # U <= 1 and U >= 2 cannot both hold
    H = np.array([[1.0]]); f = np.array([0.0])
    M = np.array([[1.0], [-1.0]]); g = np.array([1.0, -2.0])
    r = pqp(H, f, M, g, max_iter=200)
    assert not r.converged
    assert r.max_violation > 0.5
    with pytest.raises(ValueError):
        solve_exact(H, f, M, g)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        pqp(H_B, F_B, M_B, [1.0, 2.0])          # 3 rows in M, 2 entries in g
    with pytest.raises(ValueError):
        pqp(H_B, [1.0, 2.0, 3.0], M_B, G_B)      # f has 3 entries, M has 2 columns
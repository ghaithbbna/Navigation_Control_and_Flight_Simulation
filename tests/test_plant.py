import numpy as np
import pytest
from scipy.linalg import expm
from navigation_control_flight_simulation.plant import course_plant


@pytest.fixture
def plant():
    return course_plant()


def test_sizes(plant):
    assert (plant.n, plant.m, plant.p) == (7, 2, 5)


def test_eigenvalues_match_measured_values(plant):
    # Values measured earlier from the course matrices (rounded to 4 digits).
    expected = np.array([-20.2, -1.0,
                         -0.8915 + 1.2969j, -0.8915 - 1.2969j,
                         -0.0046 + 0.0846j, -0.0046 - 0.0846j,
                         0.0016])
    got = np.sort_complex(plant.eigenvalues())
    assert np.allclose(got, np.sort_complex(expected), atol=2e-4)


def test_short_period_and_phugoid_damping(plant):
    ev = plant.eigenvalues()
    for wn_expected, zeta_expected, re in [(1.574, 0.566, -0.8915), (0.0847, 0.054, -0.0046)]:
        lam = [z for z in ev if abs(z.real - re) < 1e-3 and z.imag > 0][0]
        wn = abs(lam)
        zeta = -lam.real / wn
        assert wn == pytest.approx(wn_expected, rel=0.01)
        assert zeta == pytest.approx(zeta_expected, abs=0.01)


def test_hautus_controllable_and_observable(plant):
    """Hautus rank test, one eigenvalue at a time.

    The margin is NOT uniform: the barely-unstable mode (+0.0016) is only
    weakly controllable (smallest singular value ~1.8e-5). The test uses
    1e-6 as a floor and the margin is documented, not hidden.
    """
    I = np.eye(plant.n)
    for lam in plant.eigenvalues():
        sv_c = np.linalg.svd(np.hstack([lam * I - plant.A, plant.B]), compute_uv=False)[-1]
        sv_o = np.linalg.svd(np.vstack([lam * I - plant.A, plant.C]), compute_uv=False)[-1]
        assert sv_c > 1e-6, f"weakly controllable at {lam}: {sv_c}"
        assert sv_o > 1e-6, f"weakly observable at {lam}: {sv_o}"


def test_trim_residual_is_reported_not_assumed(plant):
    """A x_trim + B u_trim is NOT zero for the course numbers (V-dot residual
    about -7.7 ft/s^2). So A, B must be treated as DEVIATION dynamics."""
    res = plant.A @ plant.x_trim + plant.B @ plant.u_trim
    assert abs(res[2]) > 1.0          # documents the fact
    assert res[0] == 0 and res[5] == 0 and res[6] == 0


def test_zoh_matches_continuous_solution(plant):
    dt = 0.01
    Phi, Gam = plant.discretize(dt)
    assert np.allclose(Phi, expm(plant.A * dt), atol=1e-14)
    # Constant-input step over 1 s: compare 100 ZOH steps with the exact
    # continuous solution x(T) = int_0^T e^{A s} B u ds, computed with a bigger block exponential.
    n, m = plant.n, plant.m
    M = np.zeros((n + m, n + m)); M[:n, :n] = plant.A; M[:n, n:] = plant.B
    exact = expm(M * 1.0)[:n, n:] @ np.array([0.3, -0.02])
    x = np.zeros(n)
    for _ in range(100):
        x = Phi @ x + Gam @ np.array([0.3, -0.02])
    assert np.allclose(x, exact, rtol=1e-9, atol=1e-10)


def test_zoh_eigenvalues_are_exp_of_continuous(plant):
    dt = 0.01
    Phi, _ = plant.discretize(dt)
    a = np.sort_complex(np.linalg.eigvals(Phi))
    b = np.sort_complex(np.exp(plant.eigenvalues() * dt))
    assert np.allclose(a, b, atol=1e-12)


def test_thrust_scale_changes_only_thrust_column(plant):
    p10 = plant.with_thrust_scale(10.0)
    assert np.allclose(p10.B[:, 0], 10 * plant.B[:, 0])
    assert np.array_equal(p10.B[:, 1], plant.B[:, 1])
    assert np.array_equal(p10.A, plant.A)
    # original untouched (frozen dataclass + copy)
    assert plant.B[5, 0] == 1.0


def test_absolute_deviation_round_trip(plant):
    x = np.arange(7, dtype=float)
    assert np.allclose(plant.to_deviation(plant.to_absolute(x)), x)

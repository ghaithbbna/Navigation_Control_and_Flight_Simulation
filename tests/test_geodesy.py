"""
What each test protects
Known points: on the equator at longitude 0 the position is (a, 0, 0), and at the pole it is (0, 0, a(1−f)). This checks the formula against known geometry.
Origin: the origin is the zero of NED.
Rotation: R is orthonormal with determinant +1, so it is a rotation and not a mirror.
Degree lengths: 1° latitude is about 111.25 km and 1° longitude is about 74.46 km, matching the theory note.
Up is negative D: climbing 30.48 m (100 ft) gives D = −30.48.
Round trip: NED to geodetic and back is accurate to 1e-6 m on 100 random points.
Curvature trap: the 5.3 m (17 ft) rise from section 5 of the theory note.
"""""

import numpy as np
import pytest

from navigation_control_flight_simulation.geodesy import (
    A, F, llh_to_ecef, ecef_to_llh, ned_rotation, llh_to_ned, ned_to_llh)

ORIGIN = (48.1374, 11.5755, 3048.0)          # Munich-like


def test_known_ecef_points():
    assert llh_to_ecef(0, 0, 0) == pytest.approx([A, 0, 0])
    assert llh_to_ecef(90, 0, 0) == pytest.approx([0, 0, A * (1 - F)], abs=1e-6)


def test_origin_is_zero_in_ned():
    assert llh_to_ned(ORIGIN, ORIGIN) == pytest.approx([0, 0, 0], abs=1e-9)


def test_rotation_is_a_proper_rotation():
    R = ned_rotation(48.1374, 11.5755)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_degree_lengths_at_munich():
    d = 1e-3                                           # small step, then scale
    north = llh_to_ned((ORIGIN[0] + d, ORIGIN[1], ORIGIN[2]), ORIGIN)[0] / d
    east = llh_to_ned((ORIGIN[0], ORIGIN[1] + d, ORIGIN[2]), ORIGIN)[1] / d
    assert north == pytest.approx(111246, abs=20)
    assert east == pytest.approx(74462, abs=20)


def test_up_is_negative_down():
    ned = llh_to_ned((ORIGIN[0], ORIGIN[1], ORIGIN[2] + 30.48), ORIGIN)
    assert ned == pytest.approx([0, 0, -30.48], abs=1e-6)


def test_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        ned = rng.uniform(-1e4, 1e4, 3)
        assert llh_to_ned(ned_to_llh(ned, ORIGIN), ORIGIN) == pytest.approx(ned, abs=1e-6)


def test_tangent_plane_leaves_the_earth():
    """8230 m east at D = 0 is about 5.3 m above the ellipsoid (the curvature trap)."""
    h = ned_to_llh([0, 8230.0, 0], ORIGIN)[2]
    assert h - ORIGIN[2] == pytest.approx(5.3, abs=0.1)
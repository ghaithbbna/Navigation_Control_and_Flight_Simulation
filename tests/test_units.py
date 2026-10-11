import numpy as np
from aeromission.units import ft2m, m2ft, G0_FT, M_PER_FT


def test_exact_foot_definition():
    assert M_PER_FT == 0.3048
    assert ft2m(1.0) == 0.3048


def test_trim_altitude_in_metres():
    # 10,000 ft MSL used by the Munich mission = 3048 m exactly
    assert np.isclose(ft2m(10000.0), 3048.0, rtol=0, atol=1e-9)


def test_round_trip_array():
    x = np.array([0.0, 1.0, -250.5, 1e5])
    assert np.allclose(m2ft(ft2m(x)), x, rtol=1e-14, atol=0)


def test_gravity_in_ft():
    # standard gravity: 32.17405 ft/s^2 (course uses 32.17)
    assert abs(G0_FT - 32.17405) < 1e-5

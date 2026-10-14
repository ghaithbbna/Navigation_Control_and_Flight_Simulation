import numpy as np
import pytest

from navigation_control_flight_simulation.mission import (
    GATE_DH, GATE_S, GATE_T, ORIGIN, TOLERANCE_FT, T, fly, gate_errors,
    raw_reference, shaped_reference, track_llh)
from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.plant import course_plant
from navigation_control_flight_simulation.units import M_PER_FT

mpc = ExplicitMPC(course_plant())


def test_gates_in_metres():
    assert GATE_S * M_PER_FT == pytest.approx([1219.2, 3048.0, 4876.8, 6705.6, 8229.6])


def test_gates_lie_east_of_the_start_at_the_right_height():
    lat, lon, h = track_llh(GATE_S, GATE_DH)
    assert np.all(np.diff(lon) > 0) and lon[0] > ORIGIN[1]
    assert lat == pytest.approx(ORIGIN[0], abs=1e-3)
    assert h == pytest.approx(ORIGIN[2] + GATE_DH * M_PER_FT)


def test_shaped_reference_passes_through_every_gate():
    r = shaped_reference()
    assert r[0] == pytest.approx(0.0, abs=1e-6)
    assert r[(GATE_T / 0.01).round().astype(int)] == pytest.approx(GATE_DH, abs=1e-3)


def test_shaped_reference_has_no_jumps():
    biggest_step = lambda r: np.abs(np.diff(r)).max()
    assert biggest_step(shaped_reference()) < 0.01 * biggest_step(raw_reference())


def test_course_style_flight_misses_the_gates():
    err = gate_errors(fly(mpc, raw_reference(), energy=False))
    assert np.abs(err).max() > 100.0


def test_energy_consistent_shaped_flight_passes_all_gates():
    err = gate_errors(fly(mpc, shaped_reference(), energy=True))
    assert np.all(np.abs(err) <= TOLERANCE_FT)
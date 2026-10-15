import numpy as np
import pytest

from navigation_control_flight_simulation.mission import (
    GATE_DH, GATE_S, GATE_T, ORIGIN, TOLERANCE_FT, T, fly, fly_c1, gate_errors, make_world,
    linear_reference, raw_reference, shaped_reference, shifted, track_llh)
from navigation_control_flight_simulation.mpc_baseline import ExplicitMPC
from navigation_control_flight_simulation.mpc_incremental import IncrementalMPC
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
    
def test_c1_passes_all_gates_and_beats_c0_on_the_shaped_reference():
    ref = shaped_reference()
    e0 = gate_errors(fly(mpc, ref, energy=True))
    e1 = gate_errors(fly_c1(IncrementalMPC(course_plant()), ref, energy=True))
    assert np.all(np.abs(e1) <= TOLERANCE_FT)
    assert np.abs(e1).max() < np.abs(e0).max()
    
def test_world_is_reproducible_and_has_the_stated_size():
    a, b = make_world(7), make_world(7)
    assert np.array_equal(a["alpha_g"], b["alpha_g"]) and a["elevator_gain"] == b["elevator_gain"]
    assert not np.array_equal(a["alpha_g"], make_world(8)["alpha_g"])
    gust_rms = np.sqrt(np.mean([np.mean((make_world(s)["alpha_g"] * 500) ** 2) for s in range(20)]))
    assert gust_rms == pytest.approx(5.0, rel=0.25)                      # 5 ft/s
    assert all(0.8 <= make_world(s)["elevator_gain"] <= 1.2 for s in range(20))


def test_a_world_changes_the_flight_and_none_means_nominal():
    ref = shaped_reference()
    nominal = gate_errors(fly(mpc, ref, energy=True))
    assert np.allclose(nominal, gate_errors(fly(mpc, ref, energy=True, world=None)))
    assert not np.allclose(nominal, gate_errors(fly(mpc, ref, energy=True, world=make_world(1))), atol=0.5)
    
def test_linear_reference_passes_through_the_gates():
    r = linear_reference()
    assert r[(GATE_T / 0.01).round().astype(int)] == pytest.approx(GATE_DH)


def test_shifted_reference_is_the_same_curve_later():
    r = shaped_reference()
    assert np.allclose(shifted(r, 0.0), r)
    assert shifted(r, 2.0)[1000] == pytest.approx(r[800], abs=1e-9)          # t = 10 s shows what was at 8 s
    assert shifted(r, 2.0)[0] == pytest.approx(r[0])                          # the start is held
    
def test_elevator_rate_limit_is_respected():
    ref = shaped_reference()
    for f in (fly(mpc, ref, True, rate_deg_s=20), fly_c1(IncrementalMPC(course_plant()), ref, True, rate_deg_s=20)):
        assert np.abs(np.diff(f["elevator_deg"])).max() <= 20 * 0.01 + 1e-9
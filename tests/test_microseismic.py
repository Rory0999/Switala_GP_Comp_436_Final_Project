"""Fast checks of the simulation package (run with `python -m pytest -q` from the repo root)."""
import numpy as np
import pytest

from microseismic import (Model, Fracture, velocity_grid, ricker, generate_events, Propagator,
                          point_weights, forward, backpropagate, pick_events, run_experiment,
                          monitor_well, lateral_array)


def small_homogeneous(v=3000.0, **kw):
    args = dict(width=300.0, height=300.0, z_top=0.0, dx=5.0, courant=0.4, frequency=30.0,
                duration=0.2, layers=(("homogeneous", 300.0, v),), pad=20)
    args.update(kw)
    return Model(**args)


def test_layered_velocity_grid():
    m = Model()
    v = velocity_grid(m)
    assert v.shape == (m.nz, m.nx)
    assert v[0, 0] == 3900.0 and v[-1, 0] == 5000.0
    target = (m.z > 2860) & (m.z < 2980)
    assert np.all(v[target] == 3300.0)
    # Velocity error applied to a single layer leaves the others unchanged.
    v2 = velocity_grid(m, 1.1, layer=2)
    assert np.allclose(v2[target], 3630.0) and v2[0, 0] == 3900.0


def test_ricker_peak_frequency():
    t = np.arange(0, 0.5, 0.0002)
    spectrum = abs(np.fft.rfft(ricker(t, 50.0)))
    assert abs(np.fft.rfftfreq(len(t), t[1])[spectrum.argmax()] - 50.0) <= 3


def test_events_inside_fracture_and_ordered_in_time():
    f = Fracture(x=300, z=2900, half_length=100, height=60, upward_fraction=0.5, n_events=6)
    e = generate_events(f)
    assert np.all(abs(e["x"] - 300) <= 100) and np.all(e["z"] >= 2870) and np.all(e["z"] <= 2930)
    # Events farther from the perforation tend to occur later (fracture growth).
    r = np.hypot(e["x"] - 300, e["z"] - 2900)
    assert np.corrcoef(r, e["t0"])[0, 1] > 0.8


def test_first_arrival_time_matches_r_over_v():
    m = small_homogeneous()
    src = point_weights([(150.0, 150.0)], m)
    rec = point_weights([(250.0, 150.0)], m)
    data = Propagator(m).run(src, ricker(m.t, m.frequency)[:, None], record=rec)
    # The 2D Green's function has a tail, so the peak lags r/v + 1.5/f slightly.
    expected = 100.0 / 3000.0 + 1.5 / m.frequency
    peak = m.t[np.argmax(abs(data[:, 0]))]
    assert expected <= peak < expected + 0.25 / m.frequency


def test_reciprocity():
    m = small_homogeneous()
    a, b = (90.3, 140.2), (210.7, 170.6)
    F = ricker(m.t, m.frequency)[:, None]
    ab = Propagator(m).run(point_weights([a], m), F, record=point_weights([b], m))
    ba = Propagator(m).run(point_weights([b], m), F, record=point_weights([a], m))
    # float32 arithmetic: agreement to round-off (~1e-6 of the peak).
    np.testing.assert_allclose(ab, ba, atol=1e-5 * abs(ab).max())


def test_time_reversal_locates_single_event_with_surrounding_array():
    m = small_homogeneous(duration=0.25)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    receivers = np.column_stack((150 + 120 * np.cos(ang), 150 + 120 * np.sin(ang)))
    events = dict(x=np.array([132.0]), z=np.array([161.0]), t0=np.array([0.01]), amplitude=np.array([1.0]))
    fwd = forward(m, events, receivers)
    back = backpropagate(m, fwd["data"], receivers)
    picks = pick_events(m, back, receivers, max_events=1, mute_radius=30.0)
    err = np.hypot(picks["x"][0] - 132.0, picks["z"][0] - 161.0)
    assert err < 0.25 * 3000.0 / m.frequency           # within a quarter wavelength
    assert abs(picks["t0"][0] - 0.01) < 0.5 / m.frequency


def test_default_experiment_l_array_recovers_events():
    receivers = np.vstack([monitor_well(n=12), lateral_array()])
    r = run_experiment(Model(), Fracture(), receivers)
    metrics = r["runs"]["true"]["metrics"]
    assert metrics["n_matched"] >= 5
    assert metrics["mean_error"] < 10.0
    assert np.isfinite(r["data"]).all()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Model(courant=0.8)
    with pytest.raises(ValueError):
        Fracture(n_events=0)
    with pytest.raises(ValueError):
        run_experiment(Model(), Fracture(), [])
    with pytest.raises(ValueError):
        run_experiment(Model(), Fracture(x=590.0), monitor_well())

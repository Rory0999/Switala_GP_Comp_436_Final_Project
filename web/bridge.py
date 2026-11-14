"""Glue between the browser (JSON in, flat arrays out) and the `microseismic` package.

The web worker calls `run(request_json, progress)`. All physics lives in the
package; this file only converts units/containers. Arrays are returned as flat
float32 vectors plus their shapes so Pyodide can hand them to JavaScript as
typed arrays without building nested lists.
"""
import json
import time
import numpy as np

from microseismic import Model, Fracture, generate_events, run_experiment


def _model(p):
    layers = tuple((str(name), float(th), float(v)) for name, th, v in p["layers"])
    return Model(width=600.0, height=600.0, z_top=2600.0, dx=float(p["dx"]), courant=0.45,
                 frequency=float(p["frequency"]), duration=float(p["duration"]), layers=layers)


def _fracture(p):
    return Fracture(x=float(p["x"]), z=float(p["z"]), half_length=float(p["half_length"]),
                    height=float(p["height"]), upward_fraction=float(p["upward_percent"]) / 100.0,
                    n_events=int(p["n_events"]), time_spread=float(p["time_spread_ms"]) / 1000.0,
                    magnitude_range=float(p["magnitude_range"]), seed=int(p["seed"]))


def _flat(a):
    a = np.asarray(a, dtype=np.float32)
    return dict(shape=list(a.shape), data=a.ravel())


def _events(e):
    return dict(x=e["x"].tolist(), z=e["z"].tolist(), t0=e["t0"].tolist(),
                amplitude=e["amplitude"].tolist() if "amplitude" in e else [],
                strength=e["strength"].tolist() if "strength" in e else [])


def _clean(d):
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in d.items() if k != "pairs"}


def run(request_json, progress):
    req = json.loads(request_json)
    start = time.perf_counter()
    fracture = _fracture(req["fracture"])
    if req["kind"] == "events":
        # Cheap preview: event positions only (same seeded generator as the simulation).
        return dict(kind="events", events=_events(generate_events(fracture)))

    model = _model(req["model"])
    if model.nx * model.nz > 241 * 241:
        raise ValueError("Grid too large for the browser; use a grid spacing of 2.5 m or more.")
    snr = req["processing"]["snr"]
    snr = np.inf if snr in (None, 0, "off") else float(snr)
    layer = req["processing"]["error_layer"]
    layer = None if layer in (None, "all", -1) else int(layer)
    result = run_experiment(model, fracture, req["receivers"],
                            velocity_error=float(req["processing"]["velocity_error_percent"]) / 100.0,
                            error_layer=layer, snr=snr, progress=progress)
    runs = {}
    for name, r in result["runs"].items():
        img = np.sqrt(r["image"])                 # display amplitude, not energy
        runs[name] = dict(scale=r["scale"], image=_flat(img / max(img.max(), 1e-30)),
                          frames=_flat(r["frames"]), frame_times=r["frame_times"].tolist(),
                          picks=_events(r["picks"]), metrics=_clean(r["metrics"]),
                          pairs=[list(p) for p in r["metrics"]["pairs"]])
    data = result["data"]
    return dict(kind="experiment", events=_events(result["events"]), runs=runs,
                t=result["t"].tolist(), data=_flat(data), dt=model.dt, nt=model.nt,
                nx=model.nx, nz=model.nz, dx=model.dx, ppw=model.points_per_wavelength(),
                forward_frames=_flat(result["forward_frames"]),
                frame_times=result["frame_times"].tolist(),
                elapsed=time.perf_counter() - start)

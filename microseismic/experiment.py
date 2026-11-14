"""One complete microseismic experiment: forward model -> noise -> time-reverse imaging.

`run_experiment` is what the web app calls; it is equally usable from a
notebook or script, e.g.

    from microseismic import Model, Fracture, run_experiment, monitor_well
    result = run_experiment(Model(), Fracture(), monitor_well(), velocity_error=0.05)
"""
import numpy as np
from .model import generate_events
from .solver import forward, backpropagate, add_noise
from .imaging import pick_events, location_metrics


def monitor_well(x=560.0, z_from=2700.0, z_to=3140.0, n=16):
    """Vertical receiver array (x fixed) in a monitoring well, TVD z_from..z_to."""
    z = np.linspace(z_from, z_to, n)
    return np.column_stack((np.full(n, x), z))


def lateral_array(z=2760.0, x_from=80.0, x_to=520.0, n=12):
    """Horizontal receiver array, e.g. geophones in a nearby lateral well."""
    x = np.linspace(x_from, x_to, n)
    return np.column_stack((x, np.full(n, z)))


def run_experiment(model, fracture, receivers, velocity_error=0.0, error_layer=None, snr=np.inf,
                   threshold=0.1, progress=None, n_frames=80, movie_stride=2):
    """Forward once, add noise, then image with the true and (optionally) a wrong velocity.

    velocity_error : fractional error of the back-propagation model, e.g. 0.05 = +5 %
    error_layer    : None = error in every layer; integer = error only in that layer
    snr            : peak-signal / RMS-noise ratio (np.inf = noise-free)
    progress       : optional callback(percent, phase)

    Returns a dict with events, data (clean and noisy), and one imaging result per
    velocity model ('true' always; 'wrong' when velocity_error != 0).
    """
    receivers = np.asarray(receivers, float).reshape(-1, 2)
    if not 1 <= len(receivers) <= 64:
        raise ValueError("Place between 1 and 64 receivers.")
    if not -0.2 <= velocity_error <= 0.2:
        raise ValueError("Velocity error must be between -20 % and +20 %.")
    events = generate_events(fracture)
    inside = ((events["x"] >= 0) & (events["x"] <= model.width) & (events["z"] >= model.z_top)
              & (events["z"] <= model.z_top + model.height))
    if not inside.all():
        raise ValueError("The fracture extends outside the model; move the perforation or shrink it.")
    passes = 2 if velocity_error == 0 else 3

    def stage(k, name):
        if progress is None:
            return None
        return lambda f: progress(int(100 * (k + f) / passes), name)

    fwd = forward(model, events, receivers, stage(0, "Forward modelling"), n_frames, movie_stride)
    noisy = add_noise(fwd["data"], snr, model)
    runs = {}
    scales = [("true", 1.0)] + ([("wrong", 1.0 + velocity_error)] if velocity_error != 0 else [])
    for k, (name, scale) in enumerate(scales, 1):
        label = "Time reversal (true velocity)" if name == "true" else \
            f"Time reversal ({100 * velocity_error:+.0f} % velocity)"
        back = backpropagate(model, noisy, receivers, scale, error_layer if name == "wrong" else None,
                             stage(k, label), n_frames, movie_stride)
        picks = pick_events(model, back, receivers, max_events=fracture.n_events, threshold=threshold)
        del back["volume"]                     # large; not needed after picking
        metrics = location_metrics(model, events, picks)
        runs[name] = dict(scale=scale, image=picks.pop("image_muted"), raw_image=back["image"],
                          frames=back["frames"], frame_times=back["frame_times"],
                          picks=picks, metrics=metrics)
    if progress is not None:
        progress(100, "Complete")
    return dict(model=model, fracture=fracture, events=events, receivers=receivers,
                t=fwd["t"], data_clean=fwd["data"], data=noisy, forward_frames=fwd["frames"],
                frame_times=fwd["frame_times"], runs=runs)

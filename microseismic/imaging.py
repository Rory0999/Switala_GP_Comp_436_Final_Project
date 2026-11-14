"""Turn a time-reversal image into located microseismic events and error metrics."""
import numpy as np


def receiver_mute(model, receivers, radius):
    """Taper (0 at receivers -> 1 away) suppressing the injection points in the image.

    During back-propagation the receivers act as sources, so p^2 is largest right
    at the array.  This Gaussian-shaped mute removes that artefact; events closer
    than ~`radius` to a receiver are therefore hard to image.
    """
    X, Z = np.meshgrid(model.x, model.z)
    d = np.full(X.shape, np.inf)
    for rx, rz in np.asarray(receivers, float).reshape(-1, 2):
        d = np.minimum(d, np.hypot(X - rx, Z - rz))
    return 1.0 - np.exp(-(d / radius) ** 2)


def sliding_max(a, half, axis):
    """Maximum over a window of +/- `half` samples along one axis (edges padded with -inf)."""
    if half <= 0:
        return a
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (half, half)
    b = np.pad(a, pad, constant_values=-np.inf)
    out = a.copy()
    for k in range(2 * half + 1):
        out = np.maximum(out, np.take(b, np.arange(k, k + n), axis=axis))
    return out


def pick_events(model, back, receivers, max_events, threshold=0.1,
                min_separation=None, mute_radius=None):
    """Locate events as separated space-time maxima of the reverse wavefield energy.

    back           : output of solver.backpropagate
    threshold      : picks weaker than this fraction of the strongest peak (in p^2) are ignored
    min_separation : metres; two picks closer than this AND within one wavelet
                     period of each other are treated as one (default half a
                     dominant wavelength in the slowest layer)
    mute_radius    : metres around each receiver removed from the image (default one wavelength)
    Returns dict(x, z, t0, strength, image_muted).
    """
    lam = model.v_min / model.frequency
    min_separation = 0.5 * lam if min_separation is None else min_separation
    mute_radius = lam if mute_radius is None else mute_radius
    mute = receiver_mute(model, receivers, mute_radius)
    muted = back["image"] * mute
    empty = dict(x=np.array([]), z=np.array([]), t0=np.array([]), strength=np.array([]),
                 image_muted=muted)
    s = back["volume_stride"]
    energy = back["volume"].astype(np.float64) ** 2 * mute[::s, ::s][None]
    # Ignore a rim of half a wavelength at the model edge, where the sponge begins
    # and the image is unreliable.
    rim = max(1, int(round(0.5 * lam / (model.dx * s))))
    energy[:, :rim, :] = energy[:, -rim:, :] = 0
    energy[:, :, :rim] = energy[:, :, -rim:] = 0
    peak = energy.max()
    if peak <= 0:
        return empty
    # Local maxima in (tau, z, x): a point equal to the maximum of its neighbourhood.
    tvol = back["volume_times"]
    dtv = tvol[1] - tvol[0] if len(tvol) > 1 else model.dt
    ht = max(1, int(round(1.0 / model.frequency / dtv)))       # one wavelet period
    hs = max(1, int(round(min_separation / (model.dx * s))))
    neigh = energy
    for axis, half in ((0, ht), (1, hs), (2, hs)):
        neigh = sliding_max(neigh, half, axis)
    cand = np.argwhere((energy == neigh) & (energy >= threshold * peak))
    vals = energy[tuple(cand.T)]
    order = np.argsort(vals)[::-1][:max_events]
    if len(order) == 0:
        return empty
    xs, zs, t0s, strength = [], [], [], []
    T = model.t[-1]
    for k in order:
        it, iz, ix = cand[k]
        # Refine on the full-resolution image within +/- one coarse cell.
        z0, x0 = iz * s, ix * s
        win = muted[max(z0 - s, 0):z0 + s + 1, max(x0 - s, 0):x0 + s + 1]
        dz, dxx = np.unravel_index(np.argmax(win), win.shape)
        zi, xi = max(z0 - s, 0) + dz, max(x0 - s, 0) + dxx
        xs.append(model.x[xi]); zs.append(model.z[zi])
        # Focus at reverse time tau <=> forward time T - tau; remove the wavelet delay 1.5/f.
        t0s.append(T - tvol[it] - 1.5 / model.frequency)
        strength.append(np.sqrt(vals[k] / peak))
    return dict(x=np.array(xs), z=np.array(zs), t0=np.array(t0s), strength=np.array(strength),
                image_muted=muted)


def match_events(true, picked, max_distance):
    """Greedy nearest-neighbour matching of picked to true events.

    Returns list of (true_index, picked_index, distance) with distance <= max_distance.
    """
    if len(picked["x"]) == 0:
        return []
    D = np.hypot(true["x"][:, None] - picked["x"][None, :],
                 true["z"][:, None] - picked["z"][None, :])
    pairs = []
    used_t, used_p = set(), set()
    for flat in np.argsort(D, axis=None):
        i, j = divmod(int(flat), D.shape[1])
        if D[i, j] > max_distance:
            break
        if i in used_t or j in used_p:
            continue
        pairs.append((i, j, float(D[i, j])))
        used_t.add(i); used_p.add(j)
    return pairs


def location_metrics(model, true, picked):
    """Summary statistics comparing the located cloud with the true events (metres)."""
    lam = model.v_min / model.frequency
    pairs = match_events(true, picked, max_distance=1.5 * lam)
    n_true, n_pick = len(true["x"]), len(picked["x"])
    out = dict(n_true=n_true, n_picked=n_pick, n_matched=len(pairs))
    if pairs:
        i = np.array([p[0] for p in pairs]); j = np.array([p[1] for p in pairs])
        dx = picked["x"][j] - true["x"][i]
        dz = picked["z"][j] - true["z"][i]
        out.update(mean_error=float(np.mean(np.hypot(dx, dz))),
                   mean_dx=float(np.mean(dx)), mean_dz=float(np.mean(dz)),
                   mean_dt0_ms=float(1000 * np.mean(picked["t0"][j] - true["t0"][i])))
    else:
        out.update(mean_error=float("nan"), mean_dx=float("nan"), mean_dz=float("nan"),
                   mean_dt0_ms=float("nan"))
    # Apparent fracture dimensions from the extent of each cloud.
    def extent(e):
        if len(e["x"]) == 0:
            return float("nan"), float("nan")
        return float(np.ptp(e["x"])), float(np.ptp(e["z"]))
    out["true_length"], out["true_height"] = extent(true)
    out["picked_length"], out["picked_height"] = extent(picked)
    if n_pick:
        out["centroid_shift_x"] = float(np.mean(picked["x"]) - np.mean(true["x"]))
        out["centroid_shift_z"] = float(np.mean(picked["z"]) - np.mean(true["z"]))
    else:
        out["centroid_shift_x"] = out["centroid_shift_z"] = float("nan")
    out["pairs"] = pairs
    return out

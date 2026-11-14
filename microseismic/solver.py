"""Explicit finite-difference solver for the 2D acoustic wave equation.

Governing PDE (heterogeneous constant-density acoustic wave equation, course
notes 06_Hyperbolic_PDEs, Eq. 21/25) with an absorbing sponge term:

    p_tt + 2 sigma(x,z) p_t = v(x,z)^2 (p_xx + p_zz) + s(x,z,t)

Discretization: 2nd-order centered differences in time (leapfrog) and the
8th-order centered stencil in space from the course notes, written in the
course's UUo (time n) / UUm (time n-1) style.

The visible model is surrounded by `pad` sponge nodes in which sigma grows
quadratically, so outgoing waves are damped instead of reflecting off the
outer Dirichlet (p = 0) edge.
"""
import numpy as np
from .model import velocity_grid, ricker

# 8th-order second-derivative coefficients c_k for k = 1..4 (c_0 = -205/72).
C0 = -205.0 / 72.0
CK = (8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)


def pad_velocity(v, pad):
    """Extend v[z, x] into the sponge by repeating the edge values."""
    return np.pad(v, pad, mode="edge")


def sponge_damping(shape, pad, dx, v_ref):
    """Quadratic damping profile sigma*dt-ready array (units 1/s) on the padded grid."""
    nz, nx = shape
    def ramp(n):
        j = np.arange(n)
        depth = np.maximum(np.maximum(pad - j, j - (n - pad - 1)), 0) / max(pad - 4, 1)
        return np.clip(depth, 0, None)
    dz_, dx_ = ramp(nz), ramp(nx)
    # sigma_max = 4 v / (sponge width): tuned so a normally incident 50 Hz plane wave
    # loses ~97 % of its amplitude in a 45-node (112 m) sponge (see README).
    return 4.0 * v_ref / (pad * dx) * np.maximum(dz_[:, None], dx_[None, :]) ** 2


def awe_2d_explicit_solver(UUo, UUm, C2, damping_dt):
    """Advance one time step; overwrites and returns UUm as p^{n+1}.

    UUo        : p at time n   (padded grid)
    UUm        : p at time n-1 (padded grid)
    C2         : (v dt / dx)^2 at interior nodes [4:-4, 4:-4]
    damping_dt : sigma * dt at interior nodes

    Update (centered first derivative for the damping term):
        (1 + d) p^{n+1} = 2 p^n - (1 - d) p^{n-1} + C^2 * lap8(p^n)
    The outer four rows/columns stay zero (Dirichlet), hidden inside the sponge.
    """
    nz, nx = UUo.shape
    center = UUo[4:-4, 4:-4]
    lap = (2 * C0) * center                     # x and z centre coefficients
    for k, ck in enumerate(CK, 1):
        lap += ck * (UUo[4 + k:nz - 4 + k, 4:-4] + UUo[4 - k:nz - 4 - k, 4:-4]
                     + UUo[4:-4, 4 + k:nx - 4 + k] + UUo[4:-4, 4 - k:nx - 4 - k])
    d = damping_dt
    UUm[4:-4, 4:-4] = (2 * center - (1 - d) * UUm[4:-4, 4:-4] + C2 * lap) / (1 + d)
    return UUm


def point_weights(points, model):
    """Bilinear interpolation indices/weights for (x, z_TVD) points on the padded grid.

    The same weights are used to inject sources and to sample receivers,
    which keeps forward modelling and back-propagation consistent (adjoint pair).
    Returns (iz, ix, w), each shaped (n_points, 4).
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError("Locations must be finite (x, z) pairs.")
    if (np.any(p[:, 0] < 0) or np.any(p[:, 0] > model.width)
            or np.any(p[:, 1] < model.z_top) or np.any(p[:, 1] > model.z_top + model.height)):
        raise ValueError("Sources and receivers must lie inside the displayed model.")
    q = np.column_stack((p[:, 0] / model.dx, (p[:, 1] - model.z_top) / model.dx)) + model.pad
    ix, iz = np.floor(q).astype(int).T
    fx, fz = (q - np.floor(q)).T
    return (np.stack((iz, iz, iz + 1, iz + 1), 1),
            np.stack((ix, ix + 1, ix, ix + 1), 1),
            np.stack(((1 - fx) * (1 - fz), fx * (1 - fz), (1 - fx) * fz, fx * fz), 1))


class Propagator:
    """Precomputes padded velocity, Courant numbers and sponge for one velocity model."""

    def __init__(self, model, velocity_scale=1.0, layer=None, dtype=np.float32):
        # float32 halves memory traffic and runs ~2x faster than float64 in NumPy
        # (important in the browser); the verification notebook shows its
        # round-off is far below the discretization error at the grids used.
        self.dtype = dtype
        self.model = model
        pad = model.pad
        v = pad_velocity(velocity_grid(model, velocity_scale, layer), pad)
        self.shape = v.shape
        self.v = v
        dt, dx = model.dt, model.dx
        if v.max() * dt / dx > 0.555:
            raise ValueError("Velocity scale makes the scheme unstable; lower the Courant number.")
        self.C2 = ((v * dt / dx) ** 2)[4:-4, 4:-4].astype(dtype)
        sigma = sponge_damping(v.shape, pad, dx, v.max())
        self.damping_dt = (sigma * dt)[4:-4, 4:-4].astype(dtype)
        # Source term s enters as dt^2 * s / dx^2 at a node (point source, delta ~ 1/dx^2).
        self.src_scale = dt ** 2 / dx ** 2
        self.crop = np.s_[pad:pad + model.nz, pad:pad + model.nx]

    def run(self, inject, forcing, record=None, on_step=None, progress=None):
        """Time-step from rest, injecting forcing[it, j] at injection point j.

        inject  : (iz, ix, w) from point_weights
        forcing : array (nt, n_inject), the source time functions s_j(t_n)
        record  : optional (iz, ix, w); returns sampled traces (nt, n_rec)
        on_step : optional callback(it, p_visible) called each step with p at t_n
        progress: optional callback(fraction) in [0, 1]
        """
        nt = forcing.shape[0]
        UUo = np.zeros(self.shape, self.dtype)
        UUm = np.zeros(self.shape, self.dtype)
        iz, ix, w = inject
        w = w.astype(self.dtype)
        forcing = np.asarray(forcing * self.src_scale, dtype=self.dtype)
        data = None if record is None else np.zeros((nt, record[0].shape[0]))
        last = -1
        for it in range(nt):
            if record is not None:
                data[it] = np.sum(UUo[record[0], record[1]] * record[2], axis=1, dtype=np.float64)
            if on_step is not None:
                on_step(it, UUo[self.crop])
            if progress is not None:
                frac = int(100 * it / nt)
                if frac != last:
                    progress(frac / 100)
                    last = frac
            if it == nt - 1:
                break
            tmp = awe_2d_explicit_solver(UUo, UUm, self.C2, self.damping_dt)
            # Add the source term for step n -> n+1 (centred: s evaluated at t_n).
            np.add.at(tmp, (iz, ix), w * forcing[it][:, None])
            UUm, UUo = UUo, tmp
        return data


def source_functions(model, events):
    """Forcing matrix (nt, n_events): amplitude * Ricker delayed by each origin time."""
    t = model.t
    return np.column_stack([a * ricker(t - t0, model.frequency)
                            for t0, a in zip(events["t0"], events["amplitude"])])


def frame_indices(nt, n_frames=80):
    return np.unique(np.linspace(0, nt - 1, min(n_frames, nt)).astype(int))


def forward(model, events, receivers, progress=None, n_frames=80, movie_stride=2):
    """Forward-model all events in one run (superposition) and record at receivers.

    Returns dict(t, data[nt, nrec], frames[n_frames, nz', nx'], frame_times).
    Movie frames are decimated in time and by `movie_stride` in space; the
    recordings use every time step.
    """
    prop = Propagator(model)
    src = point_weights(np.column_stack((events["x"], events["z"])), model)
    rec = point_weights(receivers, model)
    F = source_functions(model, events)
    keep = set(frame_indices(model.nt, n_frames).tolist())
    frames = []

    def on_step(it, p):
        if it in keep:
            frames.append(p[::movie_stride, ::movie_stride].astype(np.float32).copy())

    data = prop.run(src, F, record=rec, on_step=on_step, progress=progress)
    idx = frame_indices(model.nt, n_frames)
    return dict(t=model.t, data=data, frames=np.stack(frames), frame_times=model.t[idx])


def backpropagate(model, data, receivers, velocity_scale=1.0, layer=None, progress=None,
                  n_frames=80, movie_stride=2):
    """Time-reverse imaging: re-inject time-reversed traces at the receivers.

    Origin times are UNKNOWN in microseismic monitoring, so instead of imaging
    at a known emission time we keep the reverse wavefield in space AND time:
      * image      : max over all reverse times of p^2 (full resolution), and
      * volume     : p on a coarser space-time grid (every `movie_stride` node,
                     about 10 samples per wavelet period) used to pick events as
                     local maxima of p^2 in (tau, z, x).
    A focus at reverse time tau corresponds to forward time T - tau.

    Returns dict(image, volume, volume_times, volume_stride, frames, frame_times).
    """
    prop = Propagator(model, velocity_scale, layer)
    rec = point_weights(receivers, model)
    reversed_data = np.ascontiguousarray(data[::-1])
    image = np.zeros((model.nz, model.nx))
    t = model.t
    # Time decimation giving ~10 samples per central period of the wavelet.
    tstep = max(1, int(1.0 / (10 * model.frequency * model.dt)))
    vol_idx = np.arange(0, model.nt, tstep)
    sz = slice(None, None, movie_stride)
    volume = np.zeros((len(vol_idx),) + image[sz, sz].shape, np.float32)
    keep = set(frame_indices(model.nt, n_frames).tolist())
    frames = []

    def on_step(it, p):
        np.maximum(image, p * p, out=image)
        if it % tstep == 0:
            volume[it // tstep] = p[sz, sz]
        if it in keep:
            frames.append(p[sz, sz].astype(np.float32).copy())

    prop.run(rec, reversed_data, on_step=on_step, progress=progress)
    idx = frame_indices(model.nt, n_frames)
    return dict(image=image, volume=volume, volume_times=t[vol_idx], volume_stride=movie_stride,
                frames=np.stack(frames), frame_times=t[idx])


def add_noise(data, snr, model, seed=11):
    """Add band-limited random noise at signal-to-noise ratio `snr`.

    SNR is defined per record as  (median over traces of each trace's peak |p|) / (RMS noise),
    so a "typical" trace has the stated SNR (near traces higher, far traces lower).
    snr = inf (or <= 0, interpreted as no noise) returns a copy of data.
    White noise is filtered with the source wavelet so it shares the signal band.
    """
    if snr is None or not np.isfinite(snr) or snr <= 0:
        return data.copy()
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(data.shape)
    t = model.t
    w = ricker(t, model.frequency)
    w = w[: int(np.ceil(3.0 / model.frequency / model.dt)) + 1]
    noise = np.apply_along_axis(lambda tr: np.convolve(tr, w, mode="same"), 0, white)
    noise /= np.sqrt(np.mean(noise ** 2))
    typical_peak = np.median(np.max(np.abs(data), axis=0))
    return data + noise * typical_peak / snr

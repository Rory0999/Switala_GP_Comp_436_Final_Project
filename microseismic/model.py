"""Model geometry, layered velocity, source wavelet and fracture-event generation.

Coordinates used everywhere in this package:
    x : horizontal distance along the fracture plane (m), 0 at the left model edge
    z : true vertical depth, TVD (m), increasing downward
Arrays are indexed [z, x] (row = depth, column = distance).
Time is in seconds; wavefield amplitudes are arbitrary pressure units.
"""
from dataclasses import dataclass, field
import numpy as np

# Generic shale-play P-wave model, listed top to bottom as (name, thickness m, Vp m/s).
# The last layer extends to the bottom of the model regardless of its thickness.
DEFAULT_LAYERS = (
    ("Overburden siltstone", 190.0, 3900.0),
    ("Upper frac barrier (limestone)", 60.0, 4800.0),
    ("Target shale", 140.0, 3300.0),
    ("Lower shale", 90.0, 3700.0),
    ("Basal limestone", 120.0, 5000.0),
)


@dataclass(frozen=True)
class Model:
    """Grid, time stepping and layered P-wave velocity model."""
    width: float = 600.0          # horizontal extent of the visible model (m)
    height: float = 600.0         # vertical extent of the visible model (m)
    z_top: float = 2600.0         # TVD of the top of the visible model (m)
    dx: float = 2.5               # grid spacing, equal in x and z (m)
    courant: float = 0.45         # C = v_max dt / dx
    frequency: float = 50.0       # Ricker central frequency (Hz)
    duration: float = 0.42        # recording length (s)
    layers: tuple = DEFAULT_LAYERS
    pad: int = 45                 # absorbing-sponge width outside the visible model (nodes)

    def __post_init__(self):
        for name in ("width", "height", "dx", "courant", "frequency", "duration"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        # Stability limit of 2nd-order time / 8th-order space in 2D is C_max ~ 0.555
        # (course notes, Eq. 30); keep a small safety margin.
        if self.courant > 0.55:
            raise ValueError("Courant number must be at most 0.55 for the 8th-order 2D stencil.")
        for extent in (self.width, self.height):
            if not np.isclose(extent / self.dx, round(extent / self.dx)):
                raise ValueError("Grid spacing must divide the model width and height exactly.")
        if len(self.layers) == 0:
            raise ValueError("At least one layer is required.")
        for _, thickness, velocity in self.layers:
            if thickness <= 0 or not (500.0 <= velocity <= 8000.0):
                raise ValueError("Layer thickness must be positive and Vp between 500 and 8000 m/s.")
        if self.duration < 4 / self.frequency:
            raise ValueError("Recording must include at least four wavelet periods.")

    # ---- grid helpers -------------------------------------------------------
    @property
    def nx(self):
        return int(round(self.width / self.dx)) + 1

    @property
    def nz(self):
        return int(round(self.height / self.dx)) + 1

    @property
    def x(self):
        return np.arange(self.nx) * self.dx

    @property
    def z(self):
        return self.z_top + np.arange(self.nz) * self.dx

    @property
    def v_max(self):
        return max(v for _, _, v in self.layers)

    @property
    def v_min(self):
        return min(v for _, _, v in self.layers)

    @property
    def dt(self):
        # The fastest velocity sets the stable time step: dt = C dx / v_max (notes Eq. 31).
        return self.courant * self.dx / self.v_max

    @property
    def nt(self):
        return int(np.ceil(self.duration / self.dt)) + 1

    @property
    def t(self):
        return np.arange(self.nt) * self.dt

    def points_per_wavelength(self):
        """Grid points per minimum wavelength at f_max = 2.5 f (dispersion check, notes Eq. 29)."""
        return self.v_min / (2.5 * self.frequency) / self.dx

    def layer_tops(self):
        """TVD (m) of the top of every layer."""
        tops = [self.z_top]
        for _, thickness, _ in self.layers[:-1]:
            tops.append(tops[-1] + thickness)
        return np.array(tops)


def velocity_grid(model, scale=1.0, layer=None):
    """Return v[z, x] (m/s) on the visible grid.

    scale != 1 builds the *wrong* back-propagation model used for the
    velocity-error experiment: it multiplies every layer (layer=None) or only
    the layer with index `layer`.
    """
    v = np.empty((model.nz, model.nx))
    tops = model.layer_tops()
    z = model.z
    for k, (top, (_, _, velocity)) in enumerate(zip(tops, model.layers)):
        factor = scale if (layer is None or layer == k) else 1.0
        v[z >= top - 1e-9, :] = velocity * factor
    return v


def ricker(t, frequency):
    """Unit-peak Ricker wavelet delayed by 1.5 central periods (peak at t = 1.5/f)."""
    a = (np.pi * frequency * (t - 1.5 / frequency)) ** 2
    return (1 - 2 * a) * np.exp(-a)


@dataclass(frozen=True)
class Fracture:
    """A planar hydraulic fracture seen in the 2D section through its plane.

    Events are scattered over a rectangle centred on the perforation:
    horizontally +/- half_length, vertically `height` with a fraction
    `upward_fraction` of the height above the perforation.
    """
    x: float = 280.0               # perforation x (m)
    z: float = 2920.0              # perforation TVD (m)
    half_length: float = 120.0     # m
    height: float = 80.0           # total height (m)
    upward_fraction: float = 0.5   # 0.5 = symmetric, 1.0 = all growth upward
    n_events: int = 6
    time_spread: float = 0.15      # origin times span 0 ... time_spread (s)
    magnitude_range: float = 0.3   # events span this many magnitude units
    min_separation: float = 35.0   # minimum distance between events (m)
    seed: int = 7

    def __post_init__(self):
        if not 1 <= self.n_events <= 30:
            raise ValueError("Number of events must be between 1 and 30.")
        if min(self.half_length, self.height, self.time_spread, self.magnitude_range,
               self.min_separation) < 0:
            raise ValueError("Fracture dimensions, time spread and magnitude range must be non-negative.")
        if not 0 <= self.upward_fraction <= 1:
            raise ValueError("Upward fraction must lie between 0 and 1.")


def generate_events(fracture):
    """Return a dict of event arrays x (m), z (m TVD), t0 (s), amplitude.

    Origin times grow with distance from the perforation, mimicking a fracture
    that grows outward during the stage, plus a small random jitter.
    Amplitude ~ 10**M with M drawn uniformly over `magnitude_range`.
    """
    rng = np.random.default_rng(fracture.seed)
    n = fracture.n_events
    up = fracture.height * fracture.upward_fraction
    down = fracture.height - up
    # Rejection sampling keeps events at least `min_separation` apart so that the
    # image can resolve them (resolution ~ half a wavelength).  If the fracture is
    # too small to fit them all, the separation is relaxed step by step.
    sep = fracture.min_separation
    xs, zs = [fracture.x], [fracture.z]          # the first event sits at the perforation
    tries = 0
    while len(xs) < n:
        cx = fracture.x + rng.uniform(-1, 1) * fracture.half_length
        cz = fracture.z + rng.uniform(-up, down)
        if all(np.hypot(cx - a, cz - b) >= sep for a, b in zip(xs, zs)):
            xs.append(cx); zs.append(cz)
        tries += 1
        if tries > 400:
            sep *= 0.8
            tries = 0
    x, z = np.array(xs), np.array(zs)
    # Origin times follow fracture growth: events are ranked by distance from the
    # perforation and spaced evenly over `time_spread` (+/- 15 % jitter of the spacing).
    # Even spacing keeps events separable in time when they are close in space.
    rank = np.argsort(np.argsort(np.hypot(x - fracture.x, z - fracture.z)))
    spacing = fracture.time_spread / max(n - 1, 1)
    t0 = rank * spacing + rng.uniform(-0.15, 0.15, n) * spacing
    t0 = np.clip(t0, 0, None)
    magnitude = rng.uniform(-fracture.magnitude_range, 0, n)
    amplitude = 10.0 ** magnitude
    return dict(x=x, z=z, t0=t0, amplitude=amplitude)

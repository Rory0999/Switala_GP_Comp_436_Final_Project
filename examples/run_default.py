"""Working example: the app's default scenario, run without the browser.

    python examples/run_default.py

Runs the default model (generic shale play), the default 6-event fracture and the
"Well + lateral" receiver array, then images with the correct velocity and with
+10 % velocity in all layers. Prints location metrics and saves
examples/default_result.png. Expected output is in examples/expected_output.txt.
"""
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from microseismic import Model, Fracture, run_experiment, monitor_well, lateral_array  # noqa: E402

model, fracture = Model(), Fracture()
receivers = np.vstack([monitor_well(n=12), lateral_array(n=12)])
start = time.perf_counter()
result = run_experiment(model, fracture, receivers, velocity_error=0.10)
print(f"grid {model.nx} x {model.nz}, dt = {1e3*model.dt:.3f} ms, {model.nt} steps, "
      f"{model.points_per_wavelength():.1f} points per minimum wavelength")
print(f"run time {time.perf_counter() - start:.1f} s\n")

ev = result["events"]
print("True events:  x (m)    TVD (m)   t0 (ms)")
for x, z, t0 in zip(ev["x"], ev["z"], ev["t0"]):
    print(f"            {x:7.1f}  {z:8.1f}  {1e3*t0:7.1f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, (name, run) in zip(axes, result["runs"].items()):
    m, p = run["metrics"], run["picks"]
    print(f"\n[{name} velocity, scale {run['scale']:.2f}]  matched {m['n_matched']}/{m['n_true']}, "
          f"false picks {m['n_picked'] - m['n_matched']}, mean error {m['mean_error']:.1f} m, "
          f"bias dx {m['mean_dx']:+.1f} m, dz {m['mean_dz']:+.1f} m, t0 {m['mean_dt0_ms']:+.1f} ms")
    img = np.sqrt(run["image"]); img /= img.max()
    ax.imshow(img, extent=[0, model.width, model.z[-1], model.z[0]], cmap="inferno", aspect="auto")
    for top in model.layer_tops()[1:]:
        ax.axhline(top, color="w", ls="--", lw=0.6, alpha=0.6)
    ax.plot(receivers[:, 0], receivers[:, 1], "v", color="c", ms=5, label="receivers")
    ax.plot(ev["x"], ev["z"], "*", color="gold", ms=13, mec="k", label="true events")
    ax.plot(p["x"], p["z"], "o", mfc="none", mec="lime", ms=13, mew=2, label="located")
    ax.set_title(f"{name} velocity · mean error {m['mean_error']:.1f} m")
    ax.set_xlabel("x (m)"); ax.set_ylabel("TVD (m)")
axes[0].legend(loc="lower left", fontsize=8)
fig.tight_layout()
out = ROOT / "examples" / "default_result.png"
fig.savefig(out, dpi=110)
print(f"\nsaved {out.relative_to(ROOT)}")

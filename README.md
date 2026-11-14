# Locating hydraulic-fracture microseismicity with time-reverse imaging

**GPGN 436/536 Final Project (Fall 2026) · Rory Switala · Colorado School of Mines**

**Live app:** https://rory0999.github.io/Switala_GP_Comp_436_Final_Project/
(Python + NumPy run in your browser through Pyodide; no installation or server.)

## The geophysical problem

During a hydraulic-fracture stage, the rock that breaks releases tiny earthquakes (microseismic events, roughly M −3 to 0). Locating them is the main way operators see how far and how high a fracture actually grew. That information drives well spacing, stage design, and checks on whether the fracture stayed in the target zone or broke into a barrier. Every location, though, depends on two things: the **velocity model** used to locate the events, and the **receiver geometry** (usually one or two monitoring wells).

This app simulates the whole workflow in a 2D vertical section through a fracture in a layered shale-play model:

1. **Forward model** a set of events scattered over a fracture, all in one run (superposition), and record them on monitoring arrays.
2. Add band-limited **noise**.
3. **Locate** the events by time-reverse imaging (back-propagate the time-reversed recordings with the same wave-equation solver) — once with the **correct velocity model** and once with a **wrong** one (±10 %, all layers or a single layer).
4. Compare the located event clouds side by side: location errors, biases, origin-time errors, and the apparent fracture length and height.

## Governing PDE and numerical method

The constant-density acoustic wave equation in a heterogeneous medium, plus an absorbing-sponge damping term:

$$p_{tt} + 2\sigma(x,z)\,p_t = v(x,z)^2\,(p_{xx} + p_{zz}) + s(x,z,t)$$

* **Explicit finite differences** in the course's `UUo`/`UUm` style (`06_Hyperbolic_PDEs`): 2nd-order centred time stepping (leapfrog) and the **8th-order centred spatial stencil** from the notes (Eqs. 23–25), with the spatially varying Courant number $C_{i,j} = v_{i,j}\Delta t/\Delta x$.
* **Stability:** $\Delta t = C\,\Delta x / v_{max}$ with $C = 0.45$ (8th-order 2D limit ≈ 0.555, notes Eq. 30). **Dispersion:** the app reports grid points per minimum wavelength at $2.5f$ (notes Eq. 29); the default is about 10.6.
* **Boundaries:** a 45-node quadratic sponge outside the visible 600 m × 600 m model ($\sigma_{max} = 4v/L$), surrounded by $p = 0$. For a normally incident 50 Hz plane wave in 3300 m/s rock it reflects about 3 % of the amplitude.
* **Initial conditions:** $p = p_t = 0$. Sources are Ricker wavelets (peak at $1.5/f$ after each event's origin time) injected as $\Delta t^2 s/\Delta x^2$ with bilinear weights. Receivers sample with the same weights, so injection and recording are an adjoint pair.
* **Imaging (time reversal, unknown origin time):** the recorded traces are reversed in time and injected at the receivers. In microseismic monitoring the origin times are unknown, so events are picked as local maxima of $p^2$ in **space and reverse time**. A focus at reverse time $\tau$ gives the origin time $\hat t_0 = T - \tau - 1.5/f$. A one-wavelength mute around each receiver removes the injection points from the image.

## How this differs from the course demo

The instructor's acoustic demo does time reversal of **one source** in a **homogeneous** medium and images at the **known** emission time. This project adds:

* a **layered** velocity model (variable $v(x,z)$ in the stencil);
* **many events** with **unknown, staggered origin times**, located by space-time picking;
* realistic **one-sided monitoring arrays** (vertical well, lateral, or both);
* the **velocity-model-error experiment**, run side by side;
* noise, location metrics, and apparent fracture dimensions.

## Using the app

* **Velocity model:** edit layer names, thicknesses and Vp. The default is a generic shale play: siltstone overburden, a limestone frac barrier, the target shale, lower shale, and basal limestone.
* **Fracture and events:** set the perforation, half-length, height, upward-growth fraction, number of events, origin-time spread, magnitude range and random seed. The stars preview the events that will be simulated.
* **Receivers:** add, drag, or delete receivers on the left panel, drag the whole array, or use the presets (monitor well, lateral array, well + lateral, two wells). Right-click also deletes a receiver.
* **Processing:** set the back-propagation velocity error (−10 % to +10 %, all layers or one layer) and the noise SNR.
* **Views:** model and geometry, the forward wavefield movie, the reverse wavefield movie, and the focus image with located events, each for both velocity models. The location-results table and the receiver gather are below the panels.

Things worth trying:

* **Single-well ambiguity:** choose *Monitor well*, then drag it to x ≈ 450 m. Mirror-image foci appear on the far side of the well, because a single vertical array in a layered model cannot tell left from right. A single well also smears events along the direction away from the array.
* **Velocity error in one layer:** choose *Only: Target shale* at −10 %. Events shift in depth as well as range, which changes the apparent fracture height.
* **Noise:** set SNR to 3 or lower and watch the false picks appear.

## Repository layout

```
microseismic/            simulation package (NumPy only) — used by the app, tests, notebook and example
  model.py               Model (grid, layers, dt), velocity_grid, ricker, Fracture, generate_events
  solver.py              awe_2d_explicit_solver, sponge, Propagator, forward, backpropagate, add_noise
  imaging.py             receiver mute, space-time picking, matching and location metrics
  experiment.py          run_experiment (forward → noise → image with true and wrong velocity), array presets
web/                     browser app: index.html, style.css, app.js (UI only),
                         worker.js (Pyodide Web Worker), bridge.py (JSON ↔ package)
scripts/build_site.py    builds _site/ = web/ + an unchanged copy of microseismic/
.github/workflows/pages.yml   runs the tests, builds the site, deploys to GitHub Pages
verification.ipynb       analytical verification (executed, outputs saved)
tests/                   pytest checks
examples/                working example script, expected output and figure
AI_USE.md                AI-use statement and references
```

## Running the simulation without the browser

```bash
python -m venv .venv && source .venv/bin/activate      # or: conda env create -f environment.yml
pip install -r requirements.txt
python -m pytest -q                                     # fast checks (~10 s)
python examples/run_default.py                          # working example (~10 s); compare with examples/expected_output.txt
jupyter notebook verification.ipynb                     # verification notebook (~40 s to rerun)
```

```python
from microseismic import Model, Fracture, run_experiment, monitor_well, lateral_array
import numpy as np
receivers = np.vstack([monitor_well(n=12), lateral_array(n=12)])
result = run_experiment(Model(), Fracture(), receivers, velocity_error=0.10)
print(result["runs"]["true"]["metrics"]["mean_error"], result["runs"]["wrong"]["metrics"]["mean_error"])
```

**Expected result** for the default scenario (`examples/expected_output.txt`): with the correct velocity, 6 of 6 events are located with a mean error of about 3 m and origin times within about 0.1 ms. With +10 % velocity, 5 of 6 are matched, the mean error is about 10 m, and there is one false pick.

## Verification

`verification.ipynb` imports the same package and checks it against analytical solutions:

1. **Point source, homogeneous medium** vs. the 2D Green's function convolved with the Ricker wavelet. Relative RMS error is about 2×10⁻³ at Δx = 5 m. The error converges at **second order** in Δx (at fixed Courant number) and in Δt: the 2nd-order time stepping limits accuracy even though the spatial stencil is 8th order.
2. **Plane-wave reflection** at a shale/limestone interface vs. the 1D solution with $R = (v_2 - v_1)/(v_2 + v_1)$. This checks the variable-velocity code path; R is within 0.2 % at Δx = 2.5 m.
3. **Time-reversal location** of an off-grid source with a surrounding array. The source is located to within 1–2 m and the origin time is recovered.

## Browser dependencies

* **Pyodide 0.27.7** from jsDelivr (`web/worker.js`), with its bundled **NumPy** — the only Python package the app loads.
* No JavaScript libraries. Plots are drawn with the HTML canvas.
* The browser budget: 241 × 241 visible nodes (+ sponge), about 1,900 time steps per pass, and 2–3 passes. Native NumPy takes about 8 s; a browser typically takes several times longer. For faster previews, choose the 5 m grid in *Numerics*.

## Deployment (GitHub Pages via GitHub Actions)

1. Repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to `main`. The workflow `.github/workflows/pages.yml` runs `pytest`, runs `python scripts/build_site.py`, and deploys `_site/`.

Local preview (a web server is needed for module workers):

```bash
python scripts/build_site.py
python -m http.server 8000 --directory _site      # open http://localhost:8000
```

## Limitations

2D acoustic (P-wave only) physics with line-source geometry. There are no S-waves, no source mechanisms, and no out-of-plane distance to the monitoring well. Time-reversal resolution is about half a wavelength (≈35 m in the target shale at 50 Hz). Layer interfaces are staircased onto the grid, and the sponge is not a perfectly matched layer. The velocity-error experiment scales layer velocities; it does not model anisotropy or lateral heterogeneity.

## AI use and references

See [AI_USE.md](AI_USE.md).

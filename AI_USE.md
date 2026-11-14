# AI-use statement

**Tool:** Claude (Anthropic, Claude Opus 5.5), used through the claude.ai app as an AI coding agent.

**How it supported development:**

* **Code generation:** drafted the simulation package (`microseismic/`), the browser interface (`web/`), the build script, the GitHub Actions workflow, the tests, the working example and this documentation. The solver follows the course notes' 8th-order stencil and `UUo`/`UUm` time loop. Design decisions made during development included the space-time picking imaging condition, the sponge tuning, and single-precision arithmetic for browser speed.
* **Verification:** drafted the analytical test cases (2D Green's function, plane-wave reflection coefficient, time-reversal location), ran them, and wrote the verification notebook.
* **Testing:** ran the test suite and exercised the web interface in a headless browser during development.

**My responsibility:** I reviewed the generated code. I checked the governing equation, discretization, boundary and initial conditions, and source scaling against the course notes. I checked the verification results against the analytical solutions, and I confirmed the app's behaviour against my own understanding of the physics.

# References and acknowledgements

* GPGN 436/536 course notes, *06_Hyperbolic_PDEs.ipynb* (Ge Jin, Colorado School of Mines; the notes credit Jeff Shragge for the solver). These are the source of the 8th-order stencil, the stability and dispersion criteria, and the time-reversal concept.
* GPGN 436/536 Final Project demo, https://github.com/jinwar/GPGN436_FinalProject_Demo. Its structure was adapted: a Pyodide Web Worker plus `bridge.py`, a quadratic sponge outside the visible model, and shared bilinear injection/sampling weights.
* Pyodide, https://pyodide.org (v0.27.7), and NumPy.
* Artman, B., Podladtchikov, I., and Witten, B., 2010, Source location using time-reverse imaging: *Geophysical Prospecting*, 58, 861–873.
* Nakata, N., and Beroza, G. C., 2016, Reverse time migration for microseismic sources using the geometric mean as an imaging condition: *Geophysics*, 81(2), KS51–KS60.
* Cerjan, C., Kosloff, D., Kosloff, R., and Reshef, M., 1985, A nonreflecting boundary condition for discrete acoustic and elastic wave equations: *Geophysics*, 50(4), 705–708.
* Maxwell, S., 2014, *Microseismic Imaging of Hydraulic Fracturing: Improved Engineering of Unconventional Shale Reservoirs*: SEG Distinguished Instructor Series No. 17.

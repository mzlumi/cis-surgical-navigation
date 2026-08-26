# Surgical Navigation: Calibration, Registration and Tracking

My implementation of Programming Assignments 1 and 2 from **Computer Integrated Surgery I** at Johns Hopkins University: the math and software behind an electromagnetically tracked surgical navigation system, built from scratch and validated against the course's reference outputs.

![EM tracker distortion before and after Bernstein polynomial correction](figures/distortion_field.png)

*EM marker errors in a slice of the workspace (data set pa2-debug-f). Left: the raw EM tracker error, 9.1 mm RMS. Right: the error left after my distortion correction, 0.58 mm RMS, measured on frames that were held out of the fit. Note the different arrow magnifications.*

## Summary

- A small Python package, `cisnav`, with rigid frames, point-set registration (SVD with the reflection fix), least-squares pivot calibration and 3D Bernstein polynomial distortion correction, all written from scratch. NumPy and SciPy are used only for linear algebra (SVD, least squares) and binomial coefficients.
- **PA1:** expected EM marker positions from the optical tracker, and pivot calibration of the EM and optical probes. Both post positions match the reference in all seven debug sets to the 0.01 mm rounding of the files.
- **PA2:** distortion correction with the polynomial degree chosen by cross-validation, EM pivot calibration in corrected space, EM-to-CT registration and navigation. The probe tip in CT matches the reference to rounding in every debug set without EM noise. With noise it is within 0.2 mm.
- 506 automated tests (synthetic data with known answers, plus every data file), run by GitHub Actions on each push.
- Error analysis with five figures and an 8-page report: [`report/report.pdf`](report/report.pdf).

## Install and run

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Each step writes its results into the repository, and the committed files are the output of these commands:

```bash
# PA1: output/pa1-<set>-output1.txt for every set (or --set debug-a for one)
python -m cisnav.pa1 --data-dir data/pa1 --set all --out output/

# PA2: output/pa2-<set>-output2.txt (and output1 where an optpivot file exists)
python -m cisnav.pa2 --data-dir data/pa2 --set all --out output/
python -m cisnav.pa2 --data-dir data/pa2 --set unknown-g --degree 5   # force a degree

# Validation against the debug sets: results/pa1_validation.md, results/pa2_validation.md
python scripts/compare_debug.py --assignment pa1
python scripts/compare_debug.py --assignment pa2

# Error analysis: figures/*.png, results/error_sources.md, results/monte_carlo.md
python scripts/error_sources.py
python scripts/monte_carlo.py

# Report (needs pandoc and a LaTeX installation)
cd report && pandoc report.md -o report.pdf --pdf-engine=xelatex
```

## Repository layout

| Path | Contents |
|---|---|
| `src/cisnav/io.py` | Readers for every data file, tolerant of the header quirks |
| `src/cisnav/frames.py` | Rotations and the `Frame` class (compose, inverse, apply, homogeneous matrices) |
| `src/cisnav/registration.py` | Point-set to point-set registration (Arun et al. 1987, Umeyama 1991) |
| `src/cisnav/pivot.py` | Stacked least-squares pivot calibration |
| `src/cisnav/distortion.py` | Bernstein basis, distortion correction fit, cross-validation of the degree |
| `src/cisnav/pa1.py`, `pa2.py` | The assignment pipelines and their command line tools |
| `src/cisnav/output.py` | Output writers in the handout format |
| `scripts/` | Validation, error-source analysis and Monte Carlo study |
| `tests/` | pytest suite |
| `output/` | Program outputs for every PA1 and PA2 set |
| `results/`, `figures/` | Validation reports, analysis tables and figures |
| `report/` | Report source (Markdown) and PDF |
| `data/`, `docs/handout/` | Course data and the original handout |

## How it works

**Registration.** To find the rotation `R` and translation `p` that best map points `a_i` onto `b_i`, subtract both centroids, form `H = sum(a~_i b~_i^T)` and take its SVD `H = U S V^T`. The best rotation is `R = V U^T`. If that has determinant -1 (a reflection, possible with noisy or flat point sets), flip the direction with the smallest singular value: `R = V diag(1, 1, det(VU^T)) U^T`. Then `p = mean(b) - R mean(a)`.

**Pivot calibration.** With the tip in a dimple, every probe pose `[R_k, p_k]` satisfies `R_k t_tip + p_k = p_post`. Stacking `[R_k  -I] [t_tip; p_post] = -p_k` for all frames gives a linear least-squares problem. It is solvable only if the probe rotates about at least two axes.

**Distortion correction.** Map each EM reading into a unit box, evaluate the `(N+1)^3` tensor-product Bernstein polynomials of degree `N`, and fit coefficients by least squares so that measured positions map onto the positions predicted by the optical tracker. The degree is chosen by 5-fold cross-validation on whole calibration frames, using only the calibration data. It selects degree 4 for every distorted data set.

The report explains each method in full, including why the SVD solution is optimal and how the error behaves.

## Validation

All values are 3D distances in millimetres between my outputs and the reference outputs. Both are rounded to 0.01 mm, so differences up to about 0.017 mm are rounding. Full tables and explanations: [`results/pa1_validation.md`](results/pa1_validation.md), [`results/pa2_validation.md`](results/pa2_validation.md).

**PA1**

| Set | Error sources | C_expected max | C_expected RMS | EM post | Optical post |
|---|---|---|---|---|---|
| a | none | 0.017 | 0.005 | 0.000 | 0.000 |
| b | EM noise | 0.751 | 0.491 | 0.000 | 0.000 |
| c | EM distortion | 0.953 | 0.403 | 0.000 | 0.000 |
| d | optical jiggle | 0.024 | 0.012 | 0.000 | 0.000 |
| e | distortion, jiggle | 3.741 | 1.710 | 0.010 | 0.000 |
| f | all three | 4.282 | 1.788 | 0.010 | 0.010 |
| g | all three | 3.359 | 1.680 | 0.010 | 0.000 |

Both pivot calibrations match every set. C_expected matches where the EM tracker is error free (a, d). In the other sets the reference C_expected itself contains EM error. In set b it is identical to the noisy EM readings, and elsewhere it is not a rigid image of the calibration body. Since `F_D^-1 F_A c` depends only on optical data, I did not change the program to chase those values.

**PA2**

| Set | Error sources | Degree | FRE RMS | Tip in CT, max | Tip in CT, RMS | Dewarped EM post |
|---|---|---|---|---|---|---|
| a | none | 1 | 0.006 | 0.014 | 0.009 | 0.004 |
| b | EM noise | 1 | 0.420 | 0.065 | 0.048 | 0.176 |
| c | EM distortion | 4 | 0.016 | 0.017 | 0.013 | 0.004 |
| d | optical jiggle | 1 | 0.007 | 0.014 | 0.010 | 0.003 |
| e | all three | 4 | 0.204 | 0.197 | 0.154 | 0.006 |
| f | all three | 4 | 0.329 | 0.064 | 0.054 | 0.007 |

Set b's post differs because the reference appears to use degree 4 even without distortion. That fits some of the calibration noise; forcing degree 4 reproduces the reference to 0.006 mm, and my degree-1 post is closer to the true value. The remaining tip differences in e and f are the size of the EM noise in those sets.

## Error analysis

![PA2 debug sets with and without distortion correction](figures/error_sources.png)

The debug sets switch on one error source at a time. Optical jiggle costs nothing once each frame is mapped through its own `F_D`. EM noise sets a floor of a few tenths of a millimetre. EM distortion dominates: without correction the navigation error reaches 7.3 mm in set e, and with correction it is 0.2 mm. Further figures cover the choice of polynomial degree ([`figures/degree_cross_validation.png`](figures/degree_cross_validation.png)) and a Monte Carlo study of registration and pivot error against noise. In that study the simulated probe-tip error agrees within 4% with the prediction of Fitzpatrick, West and Maurer (1998) ([`figures/registration_monte_carlo.png`](figures/registration_monte_carlo.png), [`figures/pivot_monte_carlo.png`](figures/pivot_monte_carlo.png)).

## Status

- [x] Cartesian math library (frames, rotations) with tests
- [x] Point-set registration
- [x] Pivot calibration
- [x] PA1 pipeline and outputs for all sets
- [x] Distortion correction (Bernstein polynomial)
- [x] PA2 pipeline and outputs for all sets
- [x] Error analysis and report

## What I learned

The math in this project is short: a centroid, an SVD, a stacked least-squares system and a polynomial fit. Most of the work was in checking it. Writing a test with a known answer before trusting any number caught mistakes early, including one of my own tests, which assumed the inverse of a polynomial warp is also a polynomial. The debug sets taught me to read a mismatch as information. When my expected marker positions disagreed with the reference, the useful question was which error source the set had switched on, not how to make the numbers agree. The answer turned out to be that the reference files themselves carry EM error. Choosing the polynomial degree by cross-validation instead of by matching the answer files gave the same degree on the distorted sets and a better answer on the noise-only set. That showed me a principled choice and a correct one can be the same thing. Finally, the Monte Carlo study made the error formulas concrete: watching the simulated tip error land on the Fitzpatrick prediction, and the pivot error grow as the tilt shrank, explained the theory better than reading the formulas.

## The course

**EN.601.455/655 Computer Integrated Surgery I** (formerly 600.445/645) is taught by Prof. Russell H. Taylor in the Department of Computer Science at Johns Hopkins, through the Laboratory for Computational Sensing and Robotics (LCSR). Taylor led early work on robot-assisted orthopaedic surgery at IBM Research and directed the NSF Engineering Research Center for Computer-Integrated Surgical Systems and Technology (CISST ERC) at Hopkins. The course is one of the standard graduate introductions to medical robotics.

The course covers how computers, imaging, tracking and robots are combined to plan and carry out interventions:

- medical imaging and image-based modeling;
- coordinate frames and transformations;
- calibration of tools and trackers;
- point-based and surface registration;
- tracking and stereotactic navigation;
- surgical robot systems and human-machine cooperation.

Five programming assignments run through the semester on one simulated scenario:

| Assignment | Topic |
|---|---|
| PA1 | Frame transformations, 3D point-set registration, pivot calibration |
| PA2 | EM tracker distortion correction, EM-to-CT registration, navigation |
| PA3 | Closest point on a triangle mesh (first half of ICP) |
| PA4 | Full iterative closest point registration to a bone surface mesh |
| PA5 | Deformable registration with a statistical shape model |

This repository covers **PA1 and PA2**.

### Links

- Course page: <https://ciis.lcsr.jhu.edu/doku.php?id=courses:455-655:455-655>
- Fall 2025 schedule, with current handouts and data: <https://ciis.lcsr.jhu.edu/doku.php?id=courses:455-655:2025:fall-2025-schedule>
- Fall 2020 handout used here (JHU Computer Science mirror): <https://www.cs.jhu.edu/cista/455/Homework_2020/Programming%20Assignments%201%20and%202/ProgrammingAssignments1and2.pdf>
- Companion project course, CIS II: <https://ciis.lcsr.jhu.edu/doku.php?id=courses:456>

## The problem

A simulated stereotactic navigation setup has:

- an **electromagnetic (EM) tracker**, with a large but repeatable distortion (up to several mm) and up to about 0.3 mm of noise;
- an **optical tracker** that is accurate but jiggles on its tripod;
- a **calibration object** carrying both EM markers and optical LEDs;
- a **dimpled post** at an unknown position;
- **two pointer probes**, one tracked optically and one tracked by EM.

**PA1: basic transformations and pivot calibration**

1. A Cartesian math package for points, rotations and frames.
2. 3D point-set to point-set registration, written from scratch. The course rules forbid canned registration or calibration routines.
3. For each calibration frame, the expected EM marker positions C_expected = F_D⁻¹ · F_A · c, found by registering the optical markers on the EM base (F_D) and on the calibration object (F_A).
4. Pivot calibration of the EM probe, giving the probe tip and the post position.
5. Pivot calibration of the optical probe, after mapping its readings into EM tracker coordinates frame by frame.

**PA2: distortion calibration and navigation**

1. Fit a 3D Bernstein polynomial distortion correction from measured vs expected EM marker positions.
2. Repeat the EM pivot calibration in corrected (dewarped) EM space.
3. Locate the CT fiducials with the probe, then compute the EM-to-CT registration F_reg.
4. Report the probe tip position in CT coordinates for every navigation frame.

The full statement, the file formats and the grading rubric are in [`docs/handout/PA1-PA2-handout-2020.pdf`](docs/handout/PA1-PA2-handout-2020.pdf). The data sets and their parsing quirks are described in [`data/README.md`](data/README.md).

## Credit

The problem, the handout and the data are by Russell H. Taylor and the CIS I teaching staff at Johns Hopkins University, and are included here for reference only. The code here was written for this project from the handout alone; no code from past students' solutions was used.

Libraries: [NumPy](https://numpy.org) (SVD and least squares, via LAPACK), [SciPy](https://scipy.org) (binomial coefficients), [Matplotlib](https://matplotlib.org) (figures) and [pytest](https://pytest.org). Methods: Arun, Huang and Blostein (1987); Umeyama (1991); Fitzpatrick, West and Maurer (1998); the CIS I lecture notes for pivot calibration and Bernstein distortion correction.

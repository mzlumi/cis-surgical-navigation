# Surgical Navigation: Calibration, Registration and Tracking

My implementation of Programming Assignments 1 and 2 from **Computer Integrated Surgery I** at Johns Hopkins University: the math and software behind an electromagnetically tracked surgical navigation system, built from scratch and validated against the course's reference outputs.

![EM tracker distortion before and after Bernstein polynomial correction](figures/distortion_field.png)

*EM marker errors in a slice of the workspace (data set pa2-debug-f). Left: the raw EM tracker error, 9.1 mm RMS. Right: the error left after my distortion correction, 0.58 mm RMS, measured on frames that were held out of the fit. Note the different arrow magnifications.*

## Summary

- A small Python package, `cisnav`, with rigid frames, point-set registration (SVD with the reflection fix), least-squares pivot calibration and 3D Bernstein polynomial distortion correction, all written from scratch. NumPy and SciPy are used only for linear algebra (SVD, least squares) and binomial coefficients.
- **PA1:** expected EM marker positions from the optical tracker, and pivot calibration of the EM and optical probes. Both post positions match the reference to the 0.01 mm rounding of the files in all seven debug sets and in the four unknown sets (against the answers released after grading).
- **PA2:** distortion correction with the polynomial degree chosen by cross-validation, EM pivot calibration in corrected space, EM-to-CT registration and navigation. The probe tip in CT matches the reference to rounding in every debug set without EM noise. With noise it is within 0.2 mm, and within 0.1 mm in all four unknown sets.
- Compared with nine other public solutions on the same references, this one has the lowest median and worst-case PA2 navigation error of all of them, and the most PA1 post positions within rounding of the reference ([comparison](#comparison-with-other-public-solutions)).
- 510 automated tests (synthetic data with known answers, plus every data file), run by GitHub Actions on each push.
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

# Comparison with other public solutions: results/public_comparison.md
# (fetches the repositories at pinned commits into .cache/public, about 500 MB)
python scripts/compare_public.py

# Report (needs pandoc and a LaTeX installation)
cd report && pandoc report.md -o report.pdf --pdf-engine=xelatex
```

## Repository layout

| Path | Contents |
|---|---|
| `src/cisnav/io.py` | Readers for every data file, tolerant of the header quirks |
| `src/cisnav/frames.py` | Rotations and the `Frame` class (compose, inverse, apply, homogeneous matrices) |
| `src/cisnav/registration.py` | Point-set to point-set registration (Arun et al. 1987, Umeyama 1991) |
| `src/cisnav/pivot.py` | Stacked least-squares pivot calibration with a Procrustes-mean probe model |
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

**Pivot calibration.** With the tip in a dimple, every probe pose `[R_k, p_k]` satisfies `R_k t_tip + p_k = p_post`. Stacking `[R_k  -I] [t_tip; p_post] = -p_k` for all frames gives a linear least-squares problem. It is solvable only if the probe rotates about at least two axes. The poses come from registering a probe marker model `g` to each frame. The handout takes `g` from the first frame, but distorted EM readings change the marker shape from frame to frame, which makes the post depend on which frame comes first. I use the generalized Procrustes mean of all frames instead. It is identical for rigid readings and moves the most distorted set (unknown-h, 6 mm pivot residual) from 0.07 mm to within rounding of the released answer.

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
| e | distortion, jiggle | 3.741 | 1.710 | 0.014 | 0.000 |
| f | all three | 4.282 | 1.788 | 0.014 | 0.010 |
| g | all three | 3.359 | 1.680 | 0.000 | 0.000 |

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

Set b's post differs because the reference appears to use degree 4 even without distortion. That fits some of the calibration noise; forcing degree 4 reproduces the reference to 0.006 mm, and my degree-1 post is closer to the true value. The remaining tip differences in e and f are shared, to within 0.007 mm, by an independent implementation (see the comparison below).

**Unknown sets.** The course releases the unknown-set answers after grading, and identical copies of them are in two independent public repositories. Against those answers:

| PA1 set | EM post | Optical post |
|---|---|---|
| h | 0.010 | 0.000 |
| i | 0.000 | 0.000 |
| j | 0.000 | 0.000 |
| k | 0.000 | 0.000 |

| PA2 set | Tip in CT, max | Dewarped EM post |
|---|---|---|
| g | 0.022 | 0.000 |
| h | 0.094 | 0.000 |
| i | 0.069 | 0.010 |
| j | 0.033 | 0.010 |

## Comparison with other public solutions

Other students have published their CIS I solutions, and the course data changes from year to year. [`scripts/compare_public.py`](scripts/compare_public.py) fetches nine of these repositories at pinned commits. It scores each one's own committed output files against the official reference for the data it was run on. It then runs this program on the same input files and scores it against the same reference. Full per-set tables: [`results/public_comparison.md`](results/public_comparison.md).

**PA2, probe tip in CT coordinates** (worst error over the navigation frames of a set, then median and worst over the sets compared, mm):

| Repository | Data | Sets | Median: theirs / mine | Worst: theirs / mine |
|---|---|---|---|---|
| [wuzijian1997](https://github.com/wuzijian1997/CIS1-Programming-Assignment) | same as `data/` | 10 | 0.052 / 0.049 | 0.200 / 0.197 |
| [Zhiyuan-Ding](https://github.com/Zhiyuan-Ding/JHU-CIS1-PA2) | same as `data/` | 10 | 0.128 / 0.049 | 0.460 / 0.197 |
| [SeanSDarcy2001](https://github.com/SeanSDarcy2001/CISProgrammingAssignments) | same as `data/` | 10 | 1.693 / 0.049 | 14.9 / 0.197 |
| [sameraslan](https://github.com/sameraslan/Computer-Integrated-Surgery-Projects) | same as `data/` | 6 | 195.8 / 0.049 | 245.6 / 0.094 |
| [justiin-wang](https://github.com/justiin-wang/cis-f25) | own year | 6 | 0.039 / 0.026 | 0.310 / 0.170 |
| [cmicek1](https://github.com/cmicek1/CIS) | own year | 6 | 0.122 / 0.047 | 0.481 / 0.154 |
| [endernac](https://github.com/endernac/Computer_Integrated_Surgery) | own year | 6 | 63.4 / 0.029 | 113.8 / 0.075 |
| [dlezcan1](https://github.com/dlezcan1/cis1) | own year | 6 | 187.9 / 0.033 | 268.0 / 0.057 |
| [ahundt](https://github.com/ahundt/cis) | own year | 5 | 130.6 / 0.055 | 189.9 / 0.147 |

**PA1, post positions** (worst error over the sets compared, and the number of sets where both posts are within the 0.017 mm rounding, mm):

| Repository | Data | Sets | EM post worst: theirs / mine | Optical post worst: theirs / mine | Within rounding: theirs / mine |
|---|---|---|---|---|---|
| [wuzijian1997](https://github.com/wuzijian1997/CIS1-Programming-Assignment) | same as `data/` | 4 | 2.123 / 0.010 | 0.007 / 0.000 | 0 / 4 |
| [SeanSDarcy2001](https://github.com/SeanSDarcy2001/CISProgrammingAssignments) | same as `data/` | 11 | 7.804 / 0.014 | 0.010 / 0.010 | 3 / 11 |
| [justiin-wang](https://github.com/justiin-wang/cis-f25) | own year | 7 | 0.022 / 0.010 | 0.010 / 0.010 | 6 / 7 |
| [endernac](https://github.com/endernac/Computer_Integrated_Surgery) | own year | 7 | 0.028 / 0.024 | 0.330 / 0.010 | 3 / 6 |
| [dlezcan1](https://github.com/dlezcan1/cis1) | own year | 7 | 0.022 / 0.024 | 0.234 / 0.010 | 3 / 6 |
| [ahundt](https://github.com/ahundt/cis) | own year | 7 | 2.964 / 0.010 | 1500.4 / 0.010 | 0 / 7 |

The closest solution, wuzijian1997, is within 0.007 mm of my output on eight of the ten PA2 sets, including e and f. Those two sets are where both solutions are furthest from the reference (0.20 and 0.07 mm). That two independent implementations agree there suggests the remaining difference comes from how the reference was generated. Wherever another solution is ahead on a single set, the margin is 0.004 mm or less, below the 0.01 mm rounding of the files. Errors of tens of millimetres mean a committed file doesn't correspond to the reference at all, probably an unfinished run. The committed files may also differ from what was finally submitted or graded, so these numbers describe the public repositories, not anyone's grade.

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
- [x] Validation of the unknown sets and comparison with other public solutions

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

After the implementation was complete, other students' public solutions were used for validation, and I am grateful to their authors for publishing them. The copy of the course data in `data/` comes from [wuzijian1997/CIS1-Programming-Assignment](https://github.com/wuzijian1997/CIS1-Programming-Assignment). The released unknown-set answers come from [sameraslan/Computer-Integrated-Surgery-Projects](https://github.com/sameraslan/Computer-Integrated-Surgery-Projects), [Zhiyuan-Ding/JHU-CIS1-PA2](https://github.com/Zhiyuan-Ding/JHU-CIS1-PA2) and [JiaheXu/CIS](https://github.com/JiaheXu/CIS). Other years' data come from the repositories listed in [`results/public_comparison.md`](results/public_comparison.md). One change came out of that comparison: the Procrustes-mean probe model in pivot calibration. It was prompted by a 0.07 mm gap on unknown-h, not by anyone's code.

Libraries: [NumPy](https://numpy.org) (SVD and least squares, via LAPACK), [SciPy](https://scipy.org) (binomial coefficients), [Matplotlib](https://matplotlib.org) (figures) and [pytest](https://pytest.org). Methods: Arun, Huang and Blostein (1987); Umeyama (1991); Fitzpatrick, West and Maurer (1998); the CIS I lecture notes for pivot calibration and Bernstein distortion correction.

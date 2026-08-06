# Surgical Navigation: Calibration, Registration and Tracking

My implementation of Programming Assignments 1 and 2 from **Computer Integrated Surgery I** at Johns Hopkins University: the math and software behind an electromagnetically tracked surgical navigation system, built from scratch and validated against the course's reference outputs.

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

The full statement, the file formats and the grading rubric are in [`docs/handout/PA1-PA2-handout-2020.pdf`](docs/handout/PA1-PA2-handout-2020.pdf).

## Data and validation

[`data/`](data/README.md) holds the course data sets.

- **Debug sets** come with the instructor's expected outputs and switch on one error source at a time: EM distortion, EM noise, or optical tracker jiggle.
- **Unknown sets** have no answers.

Correctness is shown by matching every debug output, measured as the maximum and RMS error per set. The error sources are then studied one at a time, before results are reported for the unknown sets.

## Status

- [ ] Cartesian math library (frames, rotations) with tests
- [ ] Point-set registration
- [ ] Pivot calibration
- [ ] PA1 pipeline and outputs for all sets
- [ ] Distortion correction (Bernstein polynomial)
- [ ] PA2 pipeline and outputs for all sets
- [ ] Error analysis and report

## Credit

The problem, the handout and the data are by Russell H. Taylor and the CIS I teaching staff at Johns Hopkins University, and are included here for reference only. The code here was written for this project from the handout alone; no code from past students' solutions was used.

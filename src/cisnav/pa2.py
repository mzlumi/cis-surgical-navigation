"""PA2 pipeline: distortion correction, EM-to-CT registration and navigation.

Steps (handout, Assignment 2):

1. Compute ``C_expected`` for every calibration frame, as in PA1.
2. Fit a Bernstein polynomial that maps the measured EM readings ``C`` onto
   ``C_expected``. This is the distortion correction.
3. Dewarp the EM pivot readings and repeat the pivot calibration, giving the
   probe tip ``t_tip`` in a probe frame defined in undistorted space.
4. Dewarp the readings taken while the probe touches each CT fiducial and
   compute the tip positions ``B_j`` in EM base coordinates.
5. Register ``B_j`` to the CT fiducial coordinates ``b_j``, giving ``F_reg``.
6. For every navigation frame, dewarp, compute the tip in EM coordinates and
   map it into CT coordinates with ``F_reg``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cisnav import io, pa1
from cisnav.distortion import DistortionCorrection, choose_degree, cross_validate
from cisnav.frames import Frame
from cisnav.io import CalBody, CalReadings, Output1, PathLike
from cisnav.output import write_output1, write_output2
from cisnav.pivot import PivotResult, pivot_calibration
from cisnav.registration import register

DEGREES = range(1, 8)


@dataclass(frozen=True)
class DistortionFit:
    correction: DistortionCorrection
    cv_scores: dict[int, float]
    C_expected: np.ndarray


def fit_distortion(
    cal: CalBody, readings: CalReadings, degree: int | None = None
) -> DistortionFit:
    """Fit the distortion correction from one calibration data set.

    If ``degree`` is None it is chosen by cross-validation over ``DEGREES``.
    """
    C_expected = pa1.expected_C(cal, readings)
    scores = cross_validate(readings.C, C_expected, DEGREES)
    if degree is None:
        degree = choose_degree(scores)
    correction = DistortionCorrection.fit(readings.C, C_expected, degree)
    return DistortionFit(correction=correction, cv_scores=scores, C_expected=C_expected)


@dataclass(frozen=True)
class ProbeCalibration:
    """EM probe calibration in dewarped space.

    g: probe marker model (probe coordinates), from the first dewarped frame.
    pivot: pivot result; ``pivot.t_tip`` is in the same probe coordinates.
    """

    g: np.ndarray
    pivot: PivotResult


def dewarped_pivot(G: np.ndarray, correction: DistortionCorrection) -> ProbeCalibration:
    """Pivot calibration of the EM probe after dewarping every marker reading."""
    pivot, g = pivot_calibration(correction(G))
    return ProbeCalibration(g=g, pivot=pivot)


def tip_positions(
    G: np.ndarray, correction: DistortionCorrection, probe: ProbeCalibration
) -> np.ndarray:
    """Probe tip in (dewarped) EM coordinates for each frame of readings ``G``.

    Each frame is dewarped, registered to the probe model ``g`` to get the
    probe pose ``F_G[k]``, and the tip is ``F_G[k] t_tip``.
    """
    return np.stack(
        [register(probe.g, Gk).apply(probe.pivot.t_tip) for Gk in correction(G)]
    )


def registration_to_ct(B: np.ndarray, b: np.ndarray) -> Frame:
    """``F_reg`` with ``b_j ~= F_reg B_j``: EM base coordinates to CT coordinates."""
    return register(B, b)


@dataclass(frozen=True)
class PA2Result:
    distortion: DistortionFit
    probe: ProbeCalibration
    B: np.ndarray
    b: np.ndarray
    F_reg: Frame
    nav_em: np.ndarray
    nav_ct: np.ndarray
    optical: PivotResult | None

    @property
    def fre_rms(self) -> float:
        """RMS fiducial registration error ``|F_reg B_j - b_j|`` (mm)."""
        return float(np.sqrt(np.mean(np.sum((self.F_reg.apply(self.B) - self.b) ** 2, axis=1))))

    def to_output1(self) -> Output1 | None:
        """PA2 version of output1 (dewarped EM post); None without optical pivot data."""
        if self.optical is None:
            return None
        return Output1(
            em_post=self.probe.pivot.p_post,
            opt_post=self.optical.p_post,
            C_expected=self.distortion.C_expected,
        )


def run(data_dir: PathLike, prefix: str, degree: int | None = None) -> PA2Result:
    """Run all PA2 steps for one data set, e.g. ``run("data/pa2", "pa2-debug-a")``."""
    def path(kind: str) -> Path:
        return io.data_path(data_dir, prefix, kind)

    cal = io.read_calbody(path("calbody"))
    distortion = fit_distortion(cal, io.read_calreadings(path("calreadings")), degree)
    corr = distortion.correction

    probe = dewarped_pivot(io.read_empivot(path("empivot")), corr)
    B = tip_positions(io.read_em_fiducials(path("em_fiducials")), corr, probe)
    b = io.read_ct_fiducials(path("ct_fiducials"))
    F_reg = registration_to_ct(B, b)

    nav_em = tip_positions(io.read_em_nav(path("em_nav")), corr, probe)
    optical = (
        pa1.optical_pivot(cal, io.read_optpivot(path("optpivot")))
        if path("optpivot").exists()
        else None
    )
    return PA2Result(
        distortion=distortion,
        probe=probe,
        B=B,
        b=b,
        F_reg=F_reg,
        nav_em=nav_em,
        nav_ct=F_reg.apply(nav_em),
        optical=optical,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cisnav.pa2",
        description="PA2: distortion correction, EM-to-CT registration and navigation.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/pa2"))
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        help="set name such as debug-a or unknown-g; repeat for several, or 'all'",
    )
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        help="Bernstein polynomial degree (default: chosen by cross-validation)",
    )
    args = parser.parse_args(argv)

    sets = args.sets or ["all"]
    if "all" in sets:
        sets = pa1.available_sets(args.data_dir, "pa2")
    for name in sets:
        prefix = f"pa2-{name}"
        result = run(args.data_dir, prefix, args.degree)
        out2 = write_output2(args.out / f"{prefix}-output2.txt", result.nav_ct)
        written = [out2]
        output1 = result.to_output1()
        if output1 is not None:
            written.append(write_output1(args.out / f"{prefix}-output1.txt", output1))
        print(
            f"{prefix}: degree {result.distortion.correction.degree}, "
            f"pivot RMS {result.probe.pivot.residual_rms:.3f} mm, "
            f"FRE RMS {result.fre_rms:.3f} mm -> {', '.join(str(p) for p in written)}"
        )


if __name__ == "__main__":
    main()

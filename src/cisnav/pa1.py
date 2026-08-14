"""PA1 pipeline: expected EM marker positions and pivot calibrations.

Coordinate systems:

* EM tracker base (``EM``): where the EM readings ``C`` and ``G`` live.
* Optical tracker (``OT``): where the optical readings ``D``, ``A`` and ``H`` live.
* Calibration object (``cal``): where ``a`` and ``c`` are defined.

``F_D`` maps EM base coordinates to optical tracker coordinates
(``D_j = F_D d_j``) and ``F_A`` maps calibration object coordinates to optical
tracker coordinates (``A_j = F_A a_j``). Both change from frame to frame,
because the object moves and the optical tracker jiggles on its tripod.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cisnav import io
from cisnav.frames import Frame
from cisnav.io import CalBody, CalReadings, OptPivot, Output1, PathLike
from cisnav.output import format_row, write_output1
from cisnav.pivot import PivotResult, pivot_calibration
from cisnav.registration import register


def expected_C_frame(cal: CalBody, D: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Expected EM marker positions for one frame, ``F_D^-1 F_A c``.

    The optical tracker sees where the calibration object is (``F_A``) and
    where the EM base is (``F_D``); chaining them gives the object pose in EM
    base coordinates, which places the EM markers ``c`` where the EM tracker
    should have measured them if it had no distortion or noise.
    """
    F_D = register(cal.d, D)
    F_A = register(cal.a, A)
    F_cal_to_em: Frame = F_D.inv() @ F_A
    return F_cal_to_em.apply(cal.c)


def expected_C(cal: CalBody, readings: CalReadings) -> np.ndarray:
    """``C_expected`` for every frame, shape ``(N_frames, N_C, 3)``."""
    return np.stack(
        [expected_C_frame(cal, D, A) for D, A in zip(readings.D, readings.A)]
    )


def em_pivot(G: np.ndarray) -> PivotResult:
    """Pivot calibration of the EM probe from readings ``G`` ``(N_frames, N_G, 3)``.

    The post position comes out directly in EM tracker coordinates.
    """
    result, _ = pivot_calibration(G)
    return result


def optical_probe_in_em(cal: CalBody, opt: OptPivot) -> np.ndarray:
    """Optical probe markers ``H`` mapped into EM base coordinates, per frame.

    The optical tracker may move between frames (tripod jiggle), so each frame
    gets its own ``F_D = register(d, D_k)`` and ``H_em = F_D^-1 H_k``.
    """
    return np.stack([register(cal.d, D).inv().apply(H) for D, H in zip(opt.D, opt.H)])


def optical_pivot(cal: CalBody, opt: OptPivot) -> PivotResult:
    """Pivot calibration of the optical probe, with the post in EM coordinates."""
    result, _ = pivot_calibration(optical_probe_in_em(cal, opt))
    return result


@dataclass(frozen=True)
class PA1Result:
    C_expected: np.ndarray
    em: PivotResult
    optical: PivotResult

    def to_output1(self) -> Output1:
        return Output1(
            em_post=self.em.p_post,
            opt_post=self.optical.p_post,
            C_expected=self.C_expected,
        )


def run(data_dir: PathLike, prefix: str) -> PA1Result:
    """Run all PA1 steps for one data set, e.g. ``run("data/pa1", "pa1-debug-a")``."""
    cal = io.read_calbody(io.data_path(data_dir, prefix, "calbody"))
    readings = io.read_calreadings(io.data_path(data_dir, prefix, "calreadings"))
    G = io.read_empivot(io.data_path(data_dir, prefix, "empivot"))
    opt = io.read_optpivot(io.data_path(data_dir, prefix, "optpivot"))
    return PA1Result(
        C_expected=expected_C(cal, readings),
        em=em_pivot(G),
        optical=optical_pivot(cal, opt),
    )


def available_sets(data_dir: PathLike, assignment: str = "pa1") -> list[str]:
    """Set names such as ``debug-a`` found in ``data_dir``, in sorted order."""
    names = sorted(Path(data_dir).glob(f"{assignment}-*-calbody.txt"))
    return [p.name[len(assignment) + 1 : -len("-calbody.txt")] for p in names]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cisnav.pa1",
        description="PA1: expected EM marker positions and pivot calibrations.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/pa1"))
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        help="set name such as debug-a or unknown-h; repeat for several, or 'all'",
    )
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args(argv)

    sets = args.sets or ["all"]
    if "all" in sets:
        sets = available_sets(args.data_dir)
    for name in sets:
        prefix = f"pa1-{name}"
        result = run(args.data_dir, prefix)
        path = write_output1(args.out / f"{prefix}-output1.txt", result.to_output1())
        print(
            f"{prefix}: EM post {format_row(result.em.p_post)} "
            f"(pivot RMS {result.em.residual_rms:.3f} mm), "
            f"optical post {format_row(result.optical.p_post)} "
            f"(pivot RMS {result.optical.residual_rms:.3f} mm) -> {path}"
        )


if __name__ == "__main__":
    main()

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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cisnav import pa1
from cisnav.distortion import DistortionCorrection, choose_degree, cross_validate
from cisnav.frames import Frame
from cisnav.io import CalBody, CalReadings
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

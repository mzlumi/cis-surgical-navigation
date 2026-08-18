"""PA2 pipeline: distortion correction, EM-to-CT registration and navigation.

Steps (handout, Assignment 2):

1. Compute ``C_expected`` for every calibration frame, as in PA1.
2. Fit a Bernstein polynomial that maps the measured EM readings ``C`` onto
   ``C_expected``. This is the distortion correction.
3. Dewarp the EM pivot readings and repeat the pivot calibration, giving the
   probe tip ``t_tip`` in a probe frame defined in undistorted space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cisnav import pa1
from cisnav.distortion import DistortionCorrection, choose_degree, cross_validate
from cisnav.io import CalBody, CalReadings
from cisnav.pivot import PivotResult, pivot_calibration

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

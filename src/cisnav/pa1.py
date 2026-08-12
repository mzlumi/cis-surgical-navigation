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

import numpy as np

from cisnav.frames import Frame
from cisnav.io import CalBody, CalReadings
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

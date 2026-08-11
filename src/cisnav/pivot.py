"""Pivot calibration by stacked linear least squares.

During pivoting the probe tip sits in a fixed dimple while the probe body is
swung around it. For each frame ``k`` the tracker reports the probe pose
``F_k = [R_k, p_k]`` (probe coordinates to tracker coordinates). Two unknowns
stay fixed over all frames:

* ``t_tip``: the tip position in probe coordinates;
* ``p_post``: the dimple position in tracker coordinates.

Each frame gives three linear equations ``R_k t_tip + p_k = p_post``, i.e.

    [ R_k  -I ] [ t_tip  ]  =  -p_k
                [ p_post ]

Stacking all K frames gives a ``3K x 6`` system that is solved in the least
squares sense with ``numpy.linalg.lstsq`` (LAPACK ``gelsd``). The system is
well posed as long as the probe is pivoted about at least two different axes;
if all rotations are about one axis, the tip offset along that axis cannot be
told apart from the post offset and the matrix loses rank.

To get the poses, the marker positions from the first frame, relative to their
centroid, define the probe coordinate system: ``g_j = G_j[0] - mean(G[0])``.
Each frame is then registered with ``F_k = register(g, G[k])``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cisnav.frames import Frame
from cisnav.registration import register


@dataclass(frozen=True)
class PivotResult:
    """Result of a pivot calibration.

    t_tip: tip position in probe coordinates.
    p_post: dimple position in tracker coordinates.
    residual_rms: RMS of ``|R_k t_tip + p_k - p_post|`` over all frames (mm).
    rank: rank of the stacked system (6 when well posed).
    """

    t_tip: np.ndarray
    p_post: np.ndarray
    residual_rms: float
    rank: int


def solve_pivot(frames: list[Frame]) -> PivotResult:
    """Solve ``R_k t_tip + p_k = p_post`` for all frames in the least squares sense."""
    if len(frames) < 2:
        raise ValueError("pivot calibration needs at least 2 frames")
    A = np.vstack([np.hstack([F.R, -np.eye(3)]) for F in frames])
    rhs = -np.concatenate([F.p for F in frames])
    x, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)
    t_tip, p_post = x[:3], x[3:]
    resid = (A @ x - rhs).reshape(-1, 3)
    rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
    return PivotResult(t_tip=t_tip, p_post=p_post, residual_rms=rms, rank=int(rank))


def probe_local_markers(G0: np.ndarray) -> np.ndarray:
    """Probe-frame marker coordinates from the first frame: ``g_j = G_j - mean(G)``."""
    G0 = np.asarray(G0, dtype=float)
    return G0 - G0.mean(axis=0)


def probe_frames(g: np.ndarray, G: np.ndarray) -> list[Frame]:
    """Register the probe model ``g`` to every frame of readings ``G`` ``(K, N, 3)``."""
    return [register(g, Gk) for Gk in G]


def pivot_calibration(G: np.ndarray) -> tuple[PivotResult, np.ndarray]:
    """Full pivot calibration from marker readings ``G`` of shape ``(K, N, 3)``.

    Returns the pivot result and the probe-frame marker model ``g`` that
    ``t_tip`` is expressed in, so later frames can be registered to the same
    probe coordinate system.
    """
    G = np.asarray(G, dtype=float)
    g = probe_local_markers(G[0])
    return solve_pivot(probe_frames(g, G)), g

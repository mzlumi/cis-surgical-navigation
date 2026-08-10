"""Point-set to point-set rigid registration (SVD method).

Problem: given corresponding points ``a_i`` and ``b_i`` (i = 1..N), find the
rotation ``R`` and translation ``p`` that minimise

    sum_i || R a_i + p - b_i ||^2.

Method (Arun, Huang and Blostein 1987, with the reflection fix of Umeyama 1991):

1. Subtract the centroids: ``a~_i = a_i - mean(a)``, ``b~_i = b_i - mean(b)``.
   The optimal translation then decouples from the rotation, because for any R
   the best p is ``p = mean(b) - R mean(a)``.
2. With the translation removed, minimising the residual is the same as
   maximising ``trace(R H)`` where ``H = sum_i a~_i b~_i^T`` (a 3x3 matrix).
3. Take the SVD ``H = U S V^T``. The rotation that maximises the trace is
   ``R = V U^T``.
4. ``V U^T`` is orthogonal but can be a reflection (det = -1) when the data is
   noisy or planar. Umeyama's fix flips the sign of the singular direction with
   the smallest singular value: ``R = V diag(1, 1, det(V U^T)) U^T``. This is
   the closest proper rotation and is still optimal among rotations.
5. ``p = mean(b) - R mean(a)``.

The SVD is ``numpy.linalg.svd`` (LAPACK ``gesdd``). Everything else here is
written for this project; no library registration routine is used.
"""

from __future__ import annotations

import numpy as np

from cisnav.frames import Frame


def register(a: np.ndarray, b: np.ndarray, rank_tol: float = 1e-9) -> Frame:
    """Least-squares rigid frame ``F`` with ``F a_i ~= b_i``.

    Parameters
    ----------
    a, b:
        Corresponding point sets of shape ``(N, 3)``, ``N >= 3``.
    rank_tol:
        Relative tolerance for detecting degenerate (coincident or collinear)
        configurations, where the rotation about the line is not determined.

    Raises
    ------
    ValueError
        If the shapes do not match, there are fewer than 3 points, or the
        points are collinear.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected two (N, 3) arrays, got {a.shape} and {b.shape}")
    if a.shape[0] < 3:
        raise ValueError("registration needs at least 3 point pairs")

    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    a_c = a - a_mean
    b_c = b - b_mean

    H = a_c.T @ b_c
    U, S, Vt = np.linalg.svd(H)
    if S[0] == 0.0 or S[1] <= rank_tol * S[0]:
        raise ValueError("point set is degenerate (coincident or collinear)")

    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))
    R = V @ np.diag([1.0, 1.0, d]) @ U.T
    p = b_mean - R @ a_mean
    return Frame(R, p)


def residuals(F: Frame, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point distances ``|F a_i - b_i|``."""
    return np.linalg.norm(F.apply(a) - np.asarray(b, dtype=float), axis=1)


def rms_error(F: Frame, a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square registration residual."""
    return float(np.sqrt(np.mean(residuals(F, a, b) ** 2)))

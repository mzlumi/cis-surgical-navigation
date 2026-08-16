"""EM distortion correction with a 3D Bernstein polynomial.

The EM tracker's distortion is large but repeatable, so it can be modelled as
a smooth function of position and removed. Given measured marker positions
``q`` (distorted) and the positions ``p`` they should have (``C_expected``
from the optical tracker), fit a polynomial ``P`` with ``P(q) ~= p`` and use
it to "dewarp" every later EM reading.

Steps:

1. **Scale to a box.** Pick a bounding box ``[q_min, q_max]`` around the
   calibration data and map each point to ``u = (q - q_min) / (q_max - q_min)``,
   so every coordinate lies in ``[0, 1]``. Bernstein polynomials are defined
   on ``[0, 1]`` and are well conditioned there; raw coordinates in the
   hundreds of millimetres raised to the 5th power would not be.
2. **Basis.** The 1D Bernstein basis of degree ``N`` is
   ``B_{N,k}(u) = C(N, k) u^k (1 - u)^(N - k)`` for ``k = 0..N``. The 3D basis
   is the tensor product ``F_ijk(u) = B_{N,i}(u_x) B_{N,j}(u_y) B_{N,k}(u_z)``,
   giving ``(N + 1)^3`` functions.
3. **Fit.** Stack one row ``[F_000(u_s), ..., F_NNN(u_s)]`` per sample ``s``
   into a matrix ``F`` and solve ``F c = p`` for the coefficient matrix ``c``
   of shape ``((N + 1)^3, 3)`` by linear least squares. The three output
   coordinates share the same basis matrix, so one ``lstsq`` call solves all
   three at once.
4. **Apply.** For a new reading ``q``: scale with the same box, evaluate the
   basis and return ``F(u) c``.

Points outside the fitting box give ``u`` outside ``[0, 1]``; the polynomial
still evaluates, but that is extrapolation and less trustworthy. A small
``margin`` widens the box so readings just outside the calibration volume
stay inside it.

``numpy.linalg.lstsq`` (LAPACK ``gelsd``) solves the least-squares problem and
``scipy.special.comb`` gives the binomial coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import comb


@dataclass(frozen=True)
class ScaleBox:
    """Axis-aligned box used to map coordinates into ``[0, 1]^3``."""

    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def around(cls, points: np.ndarray, margin: float = 0.0) -> ScaleBox:
        """Bounding box of ``points`` widened by ``margin`` times its size on each side."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = margin * (hi - lo)
        return cls(lo - pad, hi + pad)

    def scale(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.lo) / (self.hi - self.lo)


def bernstein_1d(u: np.ndarray, degree: int) -> np.ndarray:
    """All degree-``N`` Bernstein polynomials at ``u``; shape ``u.shape + (N + 1,)``."""
    u = np.asarray(u, dtype=float)[..., None]
    k = np.arange(degree + 1)
    return comb(degree, k) * u**k * (1.0 - u) ** (degree - k)


def bernstein_3d(u: np.ndarray, degree: int) -> np.ndarray:
    """Tensor-product basis for points ``u`` of shape ``(M, 3)``; shape ``(M, (N+1)^3)``.

    Column ``i (N+1)^2 + j (N+1) + k`` holds ``B_i(u_x) B_j(u_y) B_k(u_z)``.
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    bx = bernstein_1d(u[:, 0], degree)
    by = bernstein_1d(u[:, 1], degree)
    bz = bernstein_1d(u[:, 2], degree)
    return np.einsum("mi,mj,mk->mijk", bx, by, bz).reshape(u.shape[0], -1)


@dataclass(frozen=True)
class DistortionCorrection:
    """Fitted Bernstein polynomial mapping distorted points to corrected points."""

    box: ScaleBox
    degree: int
    coeffs: np.ndarray

    @classmethod
    def fit(
        cls,
        measured: np.ndarray,
        expected: np.ndarray,
        degree: int = 5,
        margin: float = 0.0,
    ) -> DistortionCorrection:
        """Fit ``P(measured) ~= expected`` by least squares.

        ``measured`` and ``expected`` may have any shape ending in 3 (for
        example ``(N_frames, N_C, 3)``); they are flattened to point lists.
        """
        q = np.asarray(measured, dtype=float).reshape(-1, 3)
        p = np.asarray(expected, dtype=float).reshape(-1, 3)
        if q.shape != p.shape:
            raise ValueError(f"shape mismatch: {q.shape} vs {p.shape}")
        n_terms = (degree + 1) ** 3
        if q.shape[0] < n_terms:
            raise ValueError(
                f"degree {degree} needs at least {n_terms} samples, got {q.shape[0]}"
            )
        box = ScaleBox.around(q, margin)
        F = bernstein_3d(box.scale(q), degree)
        coeffs, *_ = np.linalg.lstsq(F, p, rcond=None)
        return cls(box=box, degree=degree, coeffs=coeffs)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Correct points of any shape ending in 3."""
        pts = np.asarray(points, dtype=float)
        flat = pts.reshape(-1, 3)
        out = bernstein_3d(self.box.scale(flat), self.degree) @ self.coeffs
        return out.reshape(pts.shape)

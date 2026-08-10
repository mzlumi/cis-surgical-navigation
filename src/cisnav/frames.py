"""Rotations and rigid frame transformations.

A frame ``F = [R, p]`` maps a point ``b`` to ``F b = R b + p``, where ``R`` is a
3x3 rotation matrix (orthonormal, determinant +1) and ``p`` is a translation.
This is the notation used in the course: ``F_AB`` takes coordinates in frame B
into frame A, and frames compose by matrix-style multiplication,
``F1 F2 = [R1 R2, R1 p2 + p1]``.

Points are NumPy arrays of shape ``(3,)`` or ``(N, 3)``. Rotations are plain
3x3 arrays so they work directly with NumPy linear algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix: ``skew(v) @ x == np.cross(v, x)``."""
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by ``angle`` radians about ``axis`` (Rodrigues' formula).

    ``R = I + sin(t) K + (1 - cos(t)) K^2`` with ``K = skew(axis / |axis|)``.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("rotation axis must be non-zero")
    K = skew(axis / norm)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rot_x(angle: float) -> np.ndarray:
    return rot_axis_angle(np.array([1.0, 0.0, 0.0]), angle)


def rot_y(angle: float) -> np.ndarray:
    return rot_axis_angle(np.array([0.0, 1.0, 0.0]), angle)


def rot_z(angle: float) -> np.ndarray:
    return rot_axis_angle(np.array([0.0, 0.0, 1.0]), angle)


def rot_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of the quaternion ``q = (w, x, y, z)`` (normalised first)."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed random rotation (normalised Gaussian quaternion)."""
    return rot_from_quaternion(rng.normal(size=4))


def is_rotation(R: np.ndarray, tol: float = 1e-9) -> bool:
    """True if ``R`` is orthonormal with determinant +1, within ``tol``."""
    R = np.asarray(R, dtype=float)
    return (
        R.shape == (3, 3)
        and np.allclose(R.T @ R, np.eye(3), atol=tol)
        and abs(np.linalg.det(R) - 1.0) < tol
    )


@dataclass(frozen=True)
class Frame:
    """Rigid transformation ``F = [R, p]`` acting as ``F b = R b + p``."""

    R: np.ndarray
    p: np.ndarray

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=float)
        p = np.asarray(self.p, dtype=float).reshape(3)
        if R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {R.shape}")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "p", p)

    @classmethod
    def identity(cls) -> Frame:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, M: np.ndarray) -> Frame:
        """Build a frame from a 4x4 homogeneous matrix ``[[R, p], [0, 1]]``."""
        M = np.asarray(M, dtype=float)
        if M.shape != (4, 4) or not np.allclose(M[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("expected a 4x4 homogeneous matrix with last row 0 0 0 1")
        return cls(M[:3, :3], M[:3, 3])

    def as_matrix(self) -> np.ndarray:
        """4x4 homogeneous matrix ``[[R, p], [0, 1]]``."""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.p
        return M

    def inv(self) -> Frame:
        """Inverse frame ``F^-1 = [R^T, -R^T p]``.

        Uses the transpose rather than a general matrix inverse, which is exact
        for a rotation and cheaper.
        """
        Rt = self.R.T
        return Frame(Rt, -Rt @ self.p)

    def compose(self, other: Frame) -> Frame:
        """``self * other = [R1 R2, R1 p2 + p1]``: apply ``other`` first."""
        return Frame(self.R @ other.R, self.R @ other.p + self.p)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply the frame to one point ``(3,)`` or many points ``(N, 3)``."""
        points = np.asarray(points, dtype=float)
        return points @ self.R.T + self.p

    def __matmul__(self, other: Frame) -> Frame:
        if not isinstance(other, Frame):
            return NotImplemented
        return self.compose(other)

    def allclose(self, other: Frame, atol: float = 1e-9) -> bool:
        return np.allclose(self.R, other.R, atol=atol) and np.allclose(
            self.p, other.p, atol=atol
        )


def random_frame(rng: np.random.Generator, scale: float = 100.0) -> Frame:
    """Random rotation with a translation drawn uniformly in ``[-scale, scale]^3``."""
    return Frame(random_rotation(rng), rng.uniform(-scale, scale, size=3))

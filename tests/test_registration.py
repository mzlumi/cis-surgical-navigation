import numpy as np
import pytest

from cisnav.frames import Frame, is_rotation, random_frame, rot_axis_angle, rot_z
from cisnav.registration import register, residuals, rms_error

SEEDS = range(30)


@pytest.mark.parametrize("seed", SEEDS)
def test_exact_recovery_of_random_transform(seed: int) -> None:
    rng = np.random.default_rng(seed)
    F_true = random_frame(rng, scale=500.0)
    n = rng.integers(3, 40)
    a = rng.uniform(-100, 100, size=(n, 3))
    b = F_true.apply(a)
    F = register(a, b)
    assert is_rotation(F.R)
    assert F.allclose(F_true, atol=1e-8)
    assert rms_error(F, a, b) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_planar_points_still_give_a_proper_rotation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    F_true = random_frame(rng)
    a = np.column_stack([rng.uniform(-50, 50, size=(8, 2)), np.zeros(8)])
    F = register(a, F_true.apply(a))
    assert is_rotation(F.R)
    assert F.allclose(F_true, atol=1e-8)


def test_reflection_case_is_corrected() -> None:
    # b is a mirror image of a; the unconstrained SVD solution V U^T would be
    # the reflection diag(1, 1, -1). The fixed solution must be a rotation.
    rng = np.random.default_rng(1)
    a = rng.normal(size=(10, 3)) * 20
    b = a * np.array([1.0, 1.0, -1.0])
    F = register(a, b)
    assert is_rotation(F.R)
    assert np.linalg.det(F.R) == pytest.approx(1.0)


def test_rotation_beats_every_perturbation_under_noise() -> None:
    # The SVD solution is the least-squares optimum, so small rotations of it
    # can only increase the residual.
    rng = np.random.default_rng(2)
    F_true = random_frame(rng)
    a = rng.uniform(-100, 100, size=(20, 3))
    b = F_true.apply(a) + rng.normal(scale=0.5, size=a.shape)
    F = register(a, b)
    best = rms_error(F, a, b)
    for axis in np.eye(3):
        for eps in (1e-3, -1e-3):
            R = rot_axis_angle(axis, eps) @ F.R
            p = b.mean(axis=0) - R @ a.mean(axis=0)
            assert rms_error(Frame(R, p), a, b) > best


@pytest.mark.parametrize("sigma", [0.01, 0.1, 0.3, 1.0])
def test_error_scales_with_noise(sigma: float) -> None:
    rng = np.random.default_rng(3)
    F_true = random_frame(rng)
    a = rng.uniform(-100, 100, size=(27, 3))
    b = F_true.apply(a) + rng.normal(scale=sigma, size=a.shape)
    F = register(a, b)
    # Expected squared residual sum is sigma^2 (3N - 6): 3N noise terms minus
    # the 6 degrees of freedom absorbed by the fit.
    expected = sigma * np.sqrt((3 * 27 - 6) / 27)
    assert rms_error(F, a, b) == pytest.approx(expected, rel=0.3)
    # Recovered points are much closer to the truth than the noise level.
    tre = np.linalg.norm(F.apply(a) - F_true.apply(a), axis=1).max()
    assert tre < 2.0 * sigma


def test_identity_and_pure_translation() -> None:
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert register(a, a).allclose(Frame.identity())
    t = np.array([5.0, -2.0, 7.0])
    F = register(a, a + t)
    np.testing.assert_allclose(F.R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(F.p, t, atol=1e-12)


def test_half_turn_rotation() -> None:
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    R = rot_z(np.pi)
    F = register(a, a @ R.T)
    np.testing.assert_allclose(F.R, R, atol=1e-12)


def test_order_of_points_does_not_matter_if_pairs_kept() -> None:
    rng = np.random.default_rng(4)
    F_true = random_frame(rng)
    a = rng.normal(size=(12, 3)) * 30
    b = F_true.apply(a)
    perm = rng.permutation(12)
    assert register(a[perm], b[perm]).allclose(register(a, b))


@pytest.mark.parametrize(
    "a",
    [
        np.zeros((5, 3)),
        np.outer(np.arange(5.0), [1.0, 2.0, 3.0]),
    ],
    ids=["coincident", "collinear"],
)
def test_degenerate_inputs_rejected(a: np.ndarray) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        register(a, a + 1.0)


def test_too_few_points_and_bad_shapes_rejected() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        register(np.eye(3)[:2], np.eye(3)[:2])
    with pytest.raises(ValueError, match="expected two"):
        register(np.eye(3), np.eye(4)[:, :3])


def test_residuals_shape() -> None:
    a = np.eye(3)
    assert residuals(Frame.identity(), a, a).shape == (3,)

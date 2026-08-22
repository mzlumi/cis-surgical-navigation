import numpy as np
import pytest
from math import comb

from cisnav.distortion import (
    DistortionCorrection,
    ScaleBox,
    bernstein_1d,
    bernstein_3d,
    choose_degree,
    cross_validate,
    held_out_predictions,
)


def _grid(n: int = 8, lo: float = 0.0, hi: float = 200.0) -> np.ndarray:
    t = np.linspace(lo, hi, n)
    return np.stack(np.meshgrid(t, t, t, indexing="ij"), axis=-1).reshape(-1, 3)


def _smooth_warp(p: np.ndarray) -> np.ndarray:
    """A smooth, non-polynomial distortion of a few millimetres."""
    x, y, z = p.T / 200.0
    return p + np.column_stack(
        [
            3.0 * np.sin(1.5 * x + 0.5 * y),
            2.0 * np.cos(1.2 * y) * z,
            2.5 * x * y - 1.0 * np.sin(2.0 * z),
        ]
    )


def test_bernstein_partition_of_unity() -> None:
    u = np.linspace(0, 1, 11)
    for n in range(1, 7):
        np.testing.assert_allclose(bernstein_1d(u, n).sum(axis=-1), 1.0)
    pts = np.random.default_rng(0).uniform(0, 1, size=(20, 3))
    np.testing.assert_allclose(bernstein_3d(pts, 4).sum(axis=1), 1.0)


def test_bernstein_matches_definition() -> None:
    u, n = 0.3, 5
    expected = [comb(n, k) * u**k * (1 - u) ** (n - k) for k in range(n + 1)]
    np.testing.assert_allclose(bernstein_1d(u, n), expected)


def test_bernstein_3d_column_order() -> None:
    u = np.array([[0.2, 0.5, 0.7]])
    n = 3
    b = bernstein_3d(u, n)
    bx, by, bz = (bernstein_1d(u[0, i], n) for i in range(3))
    i, j, k = 1, 2, 3
    assert b[0, i * (n + 1) ** 2 + j * (n + 1) + k] == pytest.approx(bx[i] * by[j] * bz[k])


def test_scale_box_maps_to_unit_cube() -> None:
    pts = np.random.default_rng(1).uniform(-50, 300, size=(100, 3))
    u = ScaleBox.around(pts).scale(pts)
    np.testing.assert_allclose(u.min(axis=0), 0.0)
    np.testing.assert_allclose(u.max(axis=0), 1.0)
    u_margin = ScaleBox.around(pts, margin=0.1).scale(pts)
    assert u_margin.min() > 0.0 and u_margin.max() < 1.0


def test_exact_recovery_of_polynomial_warp() -> None:
    # If the correction (measured to true) is itself a cubic, degree 3 fits it
    # exactly. The forward warp would not do, since its inverse is not a polynomial.
    measured = _grid(7)
    x, y, z = measured.T
    true = measured + np.column_stack([1e-3 * x * y, -2e-5 * z**2, 1e-6 * x * y * z])
    corr = DistortionCorrection.fit(measured, true, degree=3)
    np.testing.assert_allclose(corr(measured), true, atol=1e-6)


def test_smooth_warp_corrected_on_held_out_points() -> None:
    true = _grid(9)
    measured = _smooth_warp(true)
    corr = DistortionCorrection.fit(measured, true, degree=5)
    rng = np.random.default_rng(2)
    test_true = rng.uniform(10, 190, size=(300, 3))
    test_meas = _smooth_warp(test_true)
    before = np.linalg.norm(test_meas - test_true, axis=1).max()
    after = np.linalg.norm(corr(test_meas) - test_true, axis=1).max()
    assert before > 2.0
    assert after < 0.01


def test_noise_is_averaged_not_amplified() -> None:
    rng = np.random.default_rng(3)
    true = _grid(10)
    measured = _smooth_warp(true) + rng.normal(scale=0.2, size=true.shape)
    corr = DistortionCorrection.fit(measured, true, degree=4)
    resid = np.linalg.norm(corr(measured) - true, axis=1)
    assert np.sqrt(np.mean(resid**2)) < 0.4


def test_apply_keeps_input_shape() -> None:
    true = _grid(6)
    corr = DistortionCorrection.fit(_smooth_warp(true), true, degree=2)
    frames = _smooth_warp(true).reshape(-1, 6, 3)
    assert corr(frames).shape == frames.shape
    assert corr(true[0]).shape == (3,)


def _cubic_correction_frames(rng: np.random.Generator, noise: float) -> tuple:
    measured = rng.uniform(0, 200, size=(60, 27, 3))
    x, y, z = np.moveaxis(measured / 200.0, -1, 0)
    true = measured + np.stack([3 * x * y * z, 2 * y**3 - x, 4 * x * z**2], axis=-1)
    return measured + rng.normal(scale=noise, size=measured.shape), true


def test_cross_validation_picks_the_true_degree() -> None:
    measured, true = _cubic_correction_frames(np.random.default_rng(4), noise=0.05)
    scores = cross_validate(measured, true, degrees=range(1, 6))
    assert scores[1] > scores[2] > scores[3]
    assert choose_degree(scores) == 3


def test_cross_validation_marks_impossible_degrees() -> None:
    measured, true = _cubic_correction_frames(np.random.default_rng(5), noise=0.0)
    scores = cross_validate(measured[:6], true[:6], degrees=[2, 5])
    assert np.isfinite(scores[2]) and scores[5] == float("inf")


def test_held_out_predictions_never_use_the_test_frame() -> None:
    # Corrupt one frame's targets. Its own prediction must not move toward the
    # corrupted values, since it is never part of its own training set.
    measured, true = _cubic_correction_frames(np.random.default_rng(7), noise=0.0)
    corrupted = true.copy()
    corrupted[0] += 50.0
    pred = held_out_predictions(measured, corrupted, degree=3)
    np.testing.assert_allclose(pred[0], true[0], atol=0.5)
    assert pred.shape == measured.shape


def test_choose_degree_prefers_simpler_on_near_ties() -> None:
    assert choose_degree({3: 0.300, 4: 0.290, 5: 0.200}) == 5
    assert choose_degree({3: 0.300, 4: 0.205, 5: 0.200}) == 4


def test_box_choice_does_not_change_the_fit() -> None:
    true = _grid(8)
    measured = _smooth_warp(true)
    a = DistortionCorrection.fit(measured, true, degree=3)
    b = DistortionCorrection.fit(measured, true, degree=3, margin=0.3)
    probe = np.random.default_rng(6).uniform(-20, 220, size=(50, 3))
    np.testing.assert_allclose(a(probe), b(probe), atol=1e-6)


def test_too_few_samples_rejected() -> None:
    pts = _grid(3)
    with pytest.raises(ValueError, match="needs at least"):
        DistortionCorrection.fit(pts, pts, degree=5)

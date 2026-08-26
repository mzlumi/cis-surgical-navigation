import numpy as np
import pytest

from cisnav.frames import Frame, random_frame, rot_axis_angle, rot_z
from cisnav.pivot import (
    mean_marker_shape,
    pivot_calibration,
    probe_local_markers,
    solve_pivot,
)
from cisnav.registration import register

SEEDS = range(15)


def _pivot_poses(
    rng: np.random.Generator, t_tip: np.ndarray, p_post: np.ndarray, k: int = 12
) -> list[Frame]:
    """Poses whose tip ``t_tip`` always lands on ``p_post``, with varied tilts."""
    poses = []
    for _ in range(k):
        axis = rng.normal(size=3)
        R = rot_axis_angle(axis, rng.uniform(0.2, 0.6)) @ rot_z(rng.uniform(0, 2 * np.pi))
        poses.append(Frame(R, p_post - R @ t_tip))
    return poses


@pytest.mark.parametrize("seed", SEEDS)
def test_exact_pivot_from_poses(seed: int) -> None:
    rng = np.random.default_rng(seed)
    t_tip = rng.uniform(-150, 150, size=3)
    p_post = rng.uniform(-300, 300, size=3)
    result = solve_pivot(_pivot_poses(rng, t_tip, p_post))
    np.testing.assert_allclose(result.t_tip, t_tip, atol=1e-8)
    np.testing.assert_allclose(result.p_post, p_post, atol=1e-8)
    assert result.residual_rms < 1e-8
    assert result.rank == 6


@pytest.mark.parametrize("seed", SEEDS)
def test_pivot_from_simulated_marker_readings(seed: int) -> None:
    rng = np.random.default_rng(seed)
    markers = rng.uniform(-30, 30, size=(6, 3))  # true probe model, arbitrary origin
    tip_true = rng.uniform(-150, 150, size=3)
    p_post = rng.uniform(-300, 300, size=3)
    poses = _pivot_poses(rng, tip_true, p_post)
    G = np.stack([F.apply(markers) for F in poses])

    result, g = pivot_calibration(G)

    np.testing.assert_allclose(result.p_post, p_post, atol=1e-7)
    # The tip is reported in the probe frame defined by ``g``, so mapping it
    # through any frame's pose must land on the post.
    for Gk in G:
        np.testing.assert_allclose(register(g, Gk).apply(result.t_tip), p_post, atol=1e-7)


def test_noise_gives_small_error_and_matching_residual() -> None:
    rng = np.random.default_rng(7)
    markers = rng.uniform(-30, 30, size=(6, 3))
    p_post = np.array([100.0, -50.0, 200.0])
    poses = _pivot_poses(rng, np.array([0.0, 0.0, -120.0]), p_post, k=50)
    G = np.stack([F.apply(markers) for F in poses])
    G_noisy = G + rng.normal(scale=0.1, size=G.shape)
    result, _ = pivot_calibration(G_noisy)
    assert np.linalg.norm(result.p_post - p_post) < 1.0
    assert 0.0 < result.residual_rms < 2.0


def test_single_axis_rotation_is_rank_deficient() -> None:
    # Rotations about one axis cannot separate the tip and post along that axis.
    t_tip, p_post = np.array([0.0, 0.0, -100.0]), np.zeros(3)
    poses = [Frame(rot_z(a), p_post - rot_z(a) @ t_tip) for a in np.linspace(0, 1, 8)]
    assert solve_pivot(poses).rank < 6


def test_probe_local_markers_are_centred() -> None:
    rng = np.random.default_rng(0)
    G0 = rng.normal(size=(6, 3)) * 10 + 500
    np.testing.assert_allclose(probe_local_markers(G0).mean(axis=0), 0.0, atol=1e-12)


def test_mean_marker_shape_matches_first_frame_for_rigid_readings() -> None:
    rng = np.random.default_rng(3)
    markers = rng.uniform(-30, 30, size=(6, 3))
    poses = _pivot_poses(rng, np.array([0.0, 0.0, -120.0]), np.zeros(3))
    G = np.stack([F.apply(markers) for F in poses])
    np.testing.assert_allclose(mean_marker_shape(G), probe_local_markers(G[0]), atol=1e-9)


def test_post_from_distorted_readings_does_not_depend_on_frame_order() -> None:
    rng = np.random.default_rng(5)
    markers = rng.uniform(-30, 30, size=(6, 3))
    poses = _pivot_poses(rng, np.array([0.0, 0.0, -120.0]), np.array([10.0, 20.0, 30.0]))
    G = np.stack([F.apply(markers) for F in poses])
    G = G + 0.002 * G**2 / 100.0 + rng.normal(scale=0.5, size=G.shape)
    post, _ = pivot_calibration(G)
    for order in (rng.permutation(len(G)) for _ in range(3)):
        shuffled, _ = pivot_calibration(G[order])
        np.testing.assert_allclose(shuffled.p_post, post.p_post, atol=1e-6)


def test_needs_two_frames() -> None:
    with pytest.raises(ValueError):
        solve_pivot([random_frame(np.random.default_rng(0))])


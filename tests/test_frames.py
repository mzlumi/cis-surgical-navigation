import numpy as np
import pytest

from cisnav.frames import (
    Frame,
    is_rotation,
    random_frame,
    random_rotation,
    rot_axis_angle,
    rot_from_quaternion,
    rot_x,
    rot_y,
    rot_z,
    skew,
)

SEEDS = range(25)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_skew_matches_cross_product(rng: np.random.Generator) -> None:
    v, x = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(skew(v) @ x, np.cross(v, x))


def test_elementary_rotations() -> None:
    np.testing.assert_allclose(rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(rot_x(np.pi / 2) @ [0, 1, 0], [0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(rot_y(np.pi / 2) @ [0, 0, 1], [1, 0, 0], atol=1e-15)


def test_axis_angle_matches_quaternion(rng: np.random.Generator) -> None:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = 0.7
    q = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
    np.testing.assert_allclose(rot_axis_angle(axis, angle), rot_from_quaternion(q))


def test_zero_axis_rejected() -> None:
    with pytest.raises(ValueError):
        rot_axis_angle(np.zeros(3), 1.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_random_rotation_is_proper(seed: int) -> None:
    R = random_rotation(np.random.default_rng(seed))
    assert is_rotation(R)


def test_is_rotation_rejects_reflection_and_scaling() -> None:
    assert not is_rotation(np.diag([1.0, 1.0, -1.0]))
    assert not is_rotation(2.0 * np.eye(3))


@pytest.mark.parametrize("seed", SEEDS)
def test_frame_times_inverse_is_identity(seed: int) -> None:
    F = random_frame(np.random.default_rng(seed))
    I = Frame.identity()
    assert (F @ F.inv()).allclose(I)
    assert (F.inv() @ F).allclose(I)


@pytest.mark.parametrize("seed", SEEDS)
def test_composition_is_associative(seed: int) -> None:
    rng = np.random.default_rng(seed)
    A, B, C = (random_frame(rng) for _ in range(3))
    assert ((A @ B) @ C).allclose(A @ (B @ C))


@pytest.mark.parametrize("seed", SEEDS)
def test_compose_then_apply_equals_sequential_apply(seed: int) -> None:
    rng = np.random.default_rng(seed)
    A, B = random_frame(rng), random_frame(rng)
    pts = rng.uniform(-200, 200, size=(10, 3))
    np.testing.assert_allclose((A @ B).apply(pts), A.apply(B.apply(pts)))


@pytest.mark.parametrize("seed", SEEDS)
def test_inverse_of_product(seed: int) -> None:
    rng = np.random.default_rng(seed)
    A, B = random_frame(rng), random_frame(rng)
    assert (A @ B).inv().allclose(B.inv() @ A.inv())


@pytest.mark.parametrize("seed", SEEDS)
def test_rigid_transform_preserves_distances(seed: int) -> None:
    rng = np.random.default_rng(seed)
    F = random_frame(rng)
    a, b = rng.normal(size=(2, 3)) * 50
    assert np.isclose(np.linalg.norm(F.apply(a) - F.apply(b)), np.linalg.norm(a - b))


@pytest.mark.parametrize("seed", SEEDS)
def test_homogeneous_matrix_round_trip(seed: int) -> None:
    rng = np.random.default_rng(seed)
    A, B = random_frame(rng), random_frame(rng)
    assert Frame.from_matrix(A.as_matrix()).allclose(A)
    np.testing.assert_allclose((A @ B).as_matrix(), A.as_matrix() @ B.as_matrix())
    np.testing.assert_allclose(A.inv().as_matrix(), np.linalg.inv(A.as_matrix()))


def test_apply_single_point_and_batch_agree(rng: np.random.Generator) -> None:
    F = random_frame(rng)
    pts = rng.normal(size=(4, 3))
    np.testing.assert_allclose(
        F.apply(pts), np.stack([F.apply(p) for p in pts])
    )
    assert F.apply(pts[0]).shape == (3,)


def test_from_matrix_rejects_bad_last_row() -> None:
    M = np.eye(4)
    M[3, 0] = 1.0
    with pytest.raises(ValueError):
        Frame.from_matrix(M)

from pathlib import Path

import numpy as np
import pytest

from cisnav import io, pa1
from cisnav.frames import random_frame
from cisnav.registration import register, rms_error

DATA = Path(__file__).resolve().parents[1] / "data"
PA1_SETS = [f"debug-{s}" for s in "abcdefg"] + [f"unknown-{s}" for s in "hijk"]


def _load(set_name: str) -> tuple[io.CalBody, io.CalReadings]:
    prefix = f"pa1-{set_name}"
    d = DATA / "pa1"
    return (
        io.read_calbody(io.data_path(d, prefix, "calbody")),
        io.read_calreadings(io.data_path(d, prefix, "calreadings")),
    )


def test_expected_C_synthetic_exact() -> None:
    rng = np.random.default_rng(0)
    cal = io.CalBody(
        d=rng.uniform(-50, 50, size=(8, 3)),
        a=rng.uniform(-50, 50, size=(8, 3)),
        c=rng.uniform(0, 250, size=(27, 3)),
    )
    F_D, F_A = random_frame(rng, 500), random_frame(rng, 500)
    got = pa1.expected_C_frame(cal, F_D.apply(cal.d), F_A.apply(cal.a))
    np.testing.assert_allclose(got, (F_D.inv() @ F_A).apply(cal.c), atol=1e-9)


@pytest.mark.parametrize("set_name", ["debug-a", "debug-d"])
def test_expected_C_matches_reference_without_em_error(set_name: str) -> None:
    # Sets a and d have no EM distortion or noise, so the reference C_expected
    # is reproducible from the optical data alone, up to the 0.01 mm rounding of
    # the input files.
    cal, readings = _load(set_name)
    ref = io.read_output1(io.data_path(DATA / "pa1", f"pa1-{set_name}", "output1"))
    err = np.linalg.norm(pa1.expected_C(cal, readings) - ref.C_expected, axis=-1)
    assert err.max() < 0.03


def test_em_pivot_synthetic() -> None:
    rng = np.random.default_rng(1)
    markers = rng.uniform(-30, 30, size=(6, 3))
    tip, post = np.array([10.0, -80.0, 5.0]), np.array([200.0, 190.0, 210.0])
    G = []
    for _ in range(12):
        F = random_frame(rng)
        F = type(F)(F.R, post - F.R @ tip)
        G.append(F.apply(markers))
    np.testing.assert_allclose(pa1.em_pivot(np.stack(G)).p_post, post, atol=1e-8)


@pytest.mark.parametrize("set_name", [f"debug-{s}" for s in "abcdefg"])
def test_em_pivot_matches_reference(set_name: str) -> None:
    prefix = f"pa1-{set_name}"
    G = io.read_empivot(io.data_path(DATA / "pa1", prefix, "empivot"))
    ref = io.read_output1(io.data_path(DATA / "pa1", prefix, "output1"))
    assert np.linalg.norm(pa1.em_pivot(G).p_post - ref.em_post) < 0.02


def test_optical_probe_mapped_through_each_frames_F_D() -> None:
    # The optical tracker moves between frames; H must be expressed in EM
    # coordinates with that frame's own F_D, which cancels the motion.
    rng = np.random.default_rng(2)
    d = rng.uniform(-50, 50, size=(8, 3))
    H_em = rng.uniform(100, 300, size=(3, 6, 3))
    poses = [random_frame(rng, 1000) for _ in range(3)]
    opt = io.OptPivot(
        D=np.stack([F.apply(d) for F in poses]),
        H=np.stack([F.apply(h) for F, h in zip(poses, H_em)]),
    )
    cal = io.CalBody(d=d, a=d, c=d)
    np.testing.assert_allclose(pa1.optical_probe_in_em(cal, opt), H_em, atol=1e-9)


@pytest.mark.parametrize("set_name", [f"debug-{s}" for s in "abcdefg"])
def test_optical_pivot_matches_reference(set_name: str) -> None:
    prefix = f"pa1-{set_name}"
    cal = io.read_calbody(io.data_path(DATA / "pa1", prefix, "calbody"))
    opt = io.read_optpivot(io.data_path(DATA / "pa1", prefix, "optpivot"))
    ref = io.read_output1(io.data_path(DATA / "pa1", prefix, "output1"))
    assert np.linalg.norm(pa1.optical_pivot(cal, opt).p_post - ref.opt_post) < 0.02


@pytest.mark.parametrize("set_name", PA1_SETS)
def test_expected_C_is_a_rigid_image_of_c(set_name: str) -> None:
    cal, readings = _load(set_name)
    C_exp = pa1.expected_C(cal, readings)
    assert C_exp.shape == readings.C.shape
    for frame in C_exp:
        assert rms_error(register(cal.c, frame), cal.c, frame) < 0.02

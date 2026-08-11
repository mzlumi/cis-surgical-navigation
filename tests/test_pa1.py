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


@pytest.mark.parametrize("set_name", PA1_SETS)
def test_expected_C_is_a_rigid_image_of_c(set_name: str) -> None:
    cal, readings = _load(set_name)
    C_exp = pa1.expected_C(cal, readings)
    assert C_exp.shape == readings.C.shape
    for frame in C_exp:
        assert rms_error(register(cal.c, frame), cal.c, frame) < 0.02

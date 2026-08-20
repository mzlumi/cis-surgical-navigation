from pathlib import Path

import numpy as np
import pytest

from cisnav import io, pa1, pa2
from cisnav.distortion import DistortionCorrection
from cisnav.frames import Frame, rot_axis_angle, rot_z

DATA = Path(__file__).resolve().parents[1] / "data" / "pa2"


def _load(set_name: str) -> tuple[io.CalBody, io.CalReadings]:
    prefix = f"pa2-{set_name}"
    return (
        io.read_calbody(io.data_path(DATA, prefix, "calbody")),
        io.read_calreadings(io.data_path(DATA, prefix, "calreadings")),
    )


def _warp(p: np.ndarray) -> np.ndarray:
    x, y, z = np.moveaxis(np.asarray(p) / 500.0, -1, 0)
    return p + np.stack([4 * x * y, -3 * z**2 + y, 5 * x * z], axis=-1)


def test_dewarped_pivot_recovers_post_under_known_warp() -> None:
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 500, size=(4000, 3))
    correction = DistortionCorrection.fit(_warp(true), true, degree=4)

    markers = rng.uniform(-30, 30, size=(6, 3))
    tip, post = np.array([0.0, 0.0, -120.0]), np.array([250.0, 240.0, 260.0])
    G = []
    for _ in range(15):
        R = rot_axis_angle(rng.normal(size=3), rng.uniform(0.2, 0.5)) @ rot_z(rng.uniform(0, 6))
        G.append(Frame(R, post - R @ tip).apply(markers))
    G_measured = _warp(np.stack(G))

    raw = pa1.em_pivot(G_measured)
    dewarped = pa2.dewarped_pivot(G_measured, correction)
    assert np.linalg.norm(raw.p_post - post) > 1.0
    assert np.linalg.norm(dewarped.pivot.p_post - post) < 0.01


@pytest.mark.parametrize("set_name", ["debug-c", "debug-e", "debug-f"])
def test_cross_validation_picks_degree_four_on_distorted_sets(set_name: str) -> None:
    fit = pa2.fit_distortion(*_load(set_name))
    assert fit.correction.degree == 4


@pytest.mark.parametrize("set_name", ["debug-a", "debug-c", "debug-d", "debug-e", "debug-f"])
def test_dewarped_em_post_matches_reference(set_name: str) -> None:
    fit = pa2.fit_distortion(*_load(set_name))
    G = io.read_empivot(io.data_path(DATA, f"pa2-{set_name}", "empivot"))
    ref = io.read_output1(io.data_path(DATA, f"pa2-{set_name}", "output1"))
    probe = pa2.dewarped_pivot(G, fit.correction)
    assert np.linalg.norm(probe.pivot.p_post - ref.em_post) < 0.02


def test_fiducials_located_and_registered_to_ct() -> None:
    rng = np.random.default_rng(1)
    grid = rng.uniform(0, 500, size=(100, 3))
    identity = DistortionCorrection.fit(grid, grid, degree=1)
    g = rng.uniform(-30, 30, size=(6, 3))
    g -= g.mean(axis=0)
    t_tip = np.array([5.0, -10.0, -110.0])
    probe = pa2.ProbeCalibration(
        g=g, pivot=pa2.PivotResult(t_tip=t_tip, p_post=np.zeros(3), residual_rms=0.0, rank=6)
    )
    B_true = rng.uniform(100, 400, size=(6, 3))
    G = []
    for Bj in B_true:
        R = rot_axis_angle(rng.normal(size=3), rng.uniform(0, 1))
        G.append(Frame(R, Bj - R @ t_tip).apply(g))
    B = pa2.tip_positions(np.stack(G), identity, probe)
    np.testing.assert_allclose(B, B_true, atol=1e-8)

    F_true = Frame(rot_z(0.4) @ rot_axis_angle([1.0, 1.0, 0.0], 0.3), [-50.0, 20.0, 5.0])
    F_reg = pa2.registration_to_ct(B, F_true.apply(B_true))
    assert F_reg.allclose(F_true, atol=1e-8)


@pytest.mark.parametrize("set_name", ["debug-a", "debug-c", "debug-d"])
def test_fiducial_registration_error_small_without_noise(set_name: str) -> None:
    prefix = f"pa2-{set_name}"
    fit = pa2.fit_distortion(*_load(set_name))
    probe = pa2.dewarped_pivot(io.read_empivot(io.data_path(DATA, prefix, "empivot")), fit.correction)
    B = pa2.tip_positions(io.read_em_fiducials(io.data_path(DATA, prefix, "em_fiducials")), fit.correction, probe)
    b = io.read_ct_fiducials(io.data_path(DATA, prefix, "ct_fiducials"))
    F_reg = pa2.registration_to_ct(B, b)
    assert np.sqrt(np.mean(np.sum((F_reg.apply(B) - b) ** 2, axis=1))) < 0.05


@pytest.mark.parametrize("set_name", ["debug-a", "debug-c", "debug-d"])
def test_navigation_matches_reference_without_noise(set_name: str) -> None:
    prefix = f"pa2-{set_name}"
    result = pa2.run(DATA, prefix)
    ref = io.read_output2(io.data_path(DATA, prefix, "output2"))
    assert np.linalg.norm(result.nav_ct - ref, axis=1).max() < 0.03


def test_cli_writes_output_files(tmp_path: Path, capsys) -> None:
    pa2.main(["--data-dir", str(DATA), "--set", "debug-a", "--set", "debug-c", "--out", str(tmp_path)])
    assert (tmp_path / "pa2-debug-a-output2.txt").exists()
    assert not (tmp_path / "pa2-debug-a-output1.txt").exists()
    assert (tmp_path / "pa2-debug-c-output1.txt").exists()
    tips = io.read_output2(tmp_path / "pa2-debug-c-output2.txt")
    assert tips.shape == (4, 3)
    assert "degree 4" in capsys.readouterr().out


def test_dewarping_shrinks_pivot_residual_on_distorted_data() -> None:
    fit = pa2.fit_distortion(*_load("debug-c"))
    G = io.read_empivot(io.data_path(DATA, "pa2-debug-c", "empivot"))
    assert pa2.dewarped_pivot(G, fit.correction).pivot.residual_rms < 0.1
    assert pa1.em_pivot(G).residual_rms > 1.0

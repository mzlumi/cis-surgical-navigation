import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> dict:
    return runpy.run_path(str(ROOT / "scripts" / name))


def test_compare_debug_pa1_writes_report(tmp_path: Path) -> None:
    mod = _script("compare_debug.py")
    mod["main"](["--assignment", "pa1", "--results-dir", str(tmp_path)])
    text = (tmp_path / "pa1_validation.md").read_text()
    assert "| a | no | no | no |" in text
    assert "Interpretation" in text


def test_error_sources_writes_figures_and_table(tmp_path: Path) -> None:
    mod = _script("error_sources.py")
    mod["main"](["--figures-dir", str(tmp_path), "--results-dir", str(tmp_path)])
    for name in ("error_sources.png", "degree_cross_validation.png", "distortion_field.png"):
        assert (tmp_path / name).stat().st_size > 10_000
    assert "| debug-c | distortion | 4 |" in (tmp_path / "error_sources.md").read_text()


def test_monte_carlo_writes_figures_and_table(tmp_path: Path) -> None:
    mod = _script("monte_carlo.py")
    mod["main"](
        ["--trials", "5", "--figures-dir", str(tmp_path), "--results-dir", str(tmp_path)]
    )
    for name in ("registration_monte_carlo.png", "pivot_monte_carlo.png"):
        assert (tmp_path / name).stat().st_size > 10_000
    assert "Fitzpatrick" in (tmp_path / "monte_carlo.md").read_text()


def test_predicted_tre_matches_simulation() -> None:
    mod = _script("monte_carlo.py")
    rng = np.random.default_rng(0)
    markers = rng.uniform(-30, 30, size=(6, 3))
    target = np.array([0.0, 0.0, -100.0])
    _, tre = mod["registration_trials"](markers, target, 0.2, 2000, rng)
    assert tre == pytest.approx(mod["predicted_tre"](markers, target, 0.2), rel=0.08)


def test_compare_public_reads_every_output_style(tmp_path: Path) -> None:
    mod = _script("compare_public.py")
    styles = {
        "pa2-debug-c-output2.txt": "4, pa2-debug-c-output2.txt\n  54.84,   119.33,   62.27\n",
        "pa2-debug-d-output2.txt": "4,pa2-debug-d-output2.txt, \r\n54.84,119.33,62.27\r\n",
        "PA1-DEBUG-E-OUTPUT1.TXT": "27 8 x\n54.84 \t119.33\t6.227e1\n",
        "pa1-debug-f-output-1.txt": "27, 8, x\n54.84, 119.33, 62.27, 1.0\n",
    }
    for name, text in styles.items():
        (tmp_path / name).write_text(text)
    files = mod["set_files"](tmp_path)
    assert set(files) == {
        ("pa2-debug-c", "output2"), ("pa2-debug-d", "output2"),
        ("pa1-debug-e", "output1"), ("pa1-debug-f", "output1"),
    }
    for path in files.values():
        np.testing.assert_allclose(mod["read_points"](path), [[54.84, 119.33, 62.27]])


def test_compare_public_identity_ignores_separators(tmp_path: Path) -> None:
    mod = _script("compare_public.py")
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("6,pa2-debug-b-ct-fiducials.txt,\n  194.40,    14.75,    29.72\n")
    b.write_text("6, pa2-debug-b-ct-fiducials.txt\r\n194.40,14.75,29.72\r\n")
    assert mod["digest"](a) == mod["digest"](b)
    b.write_text("6, pa2-debug-b-ct-fiducials.txt\n194.40,14.75,29.73\n")
    assert mod["digest"](a) != mod["digest"](b)


def test_compare_debug_pa2_writes_report(tmp_path: Path) -> None:
    mod = _script("compare_debug.py")
    mod["main"](["--assignment", "pa2", "--results-dir", str(tmp_path)])
    text = (tmp_path / "pa2_validation.md").read_text()
    assert "| a | no | no | no | 1 |" in text
    assert "| unknown-j |" in text

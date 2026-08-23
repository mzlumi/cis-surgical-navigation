import runpy
from pathlib import Path

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


def test_compare_debug_pa2_writes_report(tmp_path: Path) -> None:
    mod = _script("compare_debug.py")
    mod["main"](["--assignment", "pa2", "--results-dir", str(tmp_path)])
    text = (tmp_path / "pa2_validation.md").read_text()
    assert "| a | no | no | no | 1 |" in text
    assert "| unknown-j |" in text

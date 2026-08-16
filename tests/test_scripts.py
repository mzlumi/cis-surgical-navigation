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

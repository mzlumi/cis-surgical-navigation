from pathlib import Path

import numpy as np
import pytest

from cisnav import io, pa1
from cisnav.output import format_row, write_output1, write_output2

DATA = Path(__file__).resolve().parents[1] / "data"


def test_format_row_matches_reference_layout() -> None:
    assert format_row([205.88, 195.77, 193.11]) == "  205.88,   195.77,   193.11"
    assert format_row([-1.234, 0.0, 1000.006]) == "   -1.23,     0.00,  1000.01"


@pytest.mark.parametrize("set_name", ["debug-a", "debug-f"])
def test_output1_round_trip_reproduces_reference_bytes(tmp_path: Path, set_name: str) -> None:
    ref_path = io.data_path(DATA / "pa1", f"pa1-{set_name}", "output1")
    out = write_output1(tmp_path / ref_path.name, io.read_output1(ref_path))
    assert out.read_text() == ref_path.read_text()


def test_output2_round_trip(tmp_path: Path) -> None:
    ref_path = io.data_path(DATA / "pa2", "pa2-debug-a", "output2")
    tips = io.read_output2(ref_path)
    out = write_output2(tmp_path / ref_path.name, tips)
    lines = out.read_text().splitlines()
    assert lines[0] == "4, pa2-debug-a-output2.txt"
    assert lines[1:] == ref_path.read_text().splitlines()[1:]
    np.testing.assert_array_equal(io.read_output2(out), tips)


def test_cli_writes_debug_a_matching_reference(tmp_path: Path, capsys) -> None:
    pa1.main(["--data-dir", str(DATA / "pa1"), "--set", "debug-a", "--out", str(tmp_path)])
    written = io.read_output1(tmp_path / "pa1-debug-a-output1.txt")
    ref = io.read_output1(io.data_path(DATA / "pa1", "pa1-debug-a", "output1"))
    np.testing.assert_allclose(written.C_expected, ref.C_expected, atol=0.0101)
    np.testing.assert_allclose(written.em_post, ref.em_post, atol=0.0101)
    np.testing.assert_allclose(written.opt_post, ref.opt_post, atol=0.0101)
    assert "pa1-debug-a" in capsys.readouterr().out


def test_available_sets() -> None:
    sets = pa1.available_sets(DATA / "pa1")
    assert sets[0] == "debug-a" and sets[-1] == "unknown-k" and len(sets) == 11

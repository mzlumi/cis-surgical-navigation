from pathlib import Path

import numpy as np
import pytest

from cisnav import io

DATA = Path(__file__).resolve().parents[1] / "data"
ALL_FILES = sorted(DATA.glob("pa*/*.txt"))


def _kind(path: Path) -> str:
    for kind, suffix in io.SUFFIXES.items():
        if path.name.endswith(f"-{suffix}.txt"):
            return kind
    raise AssertionError(f"unrecognised file {path.name}")


@pytest.mark.parametrize(
    "line, counts, name",
    [
        ("8, 8, 27, pa1-debug-a-calbody.txt", (8, 8, 27), "pa1-debug-a-calbody.txt"),
        ("6,4,pa2-debug-a-EM-nav.txt", (6, 4), "pa2-debug-a-EM-nav.txt"),
        ("6,pa2-debug-a-ct-fiducials.txt,", (6,), "pa2-debug-a-ct-fiducials.txt"),
        ("4, pa2-debug-a-output2.txt,\n", (4,), "pa2-debug-a-output2.txt"),
    ],
)
def test_parse_header_variants(line: str, counts: tuple[int, ...], name: str) -> None:
    header = io.parse_header(line)
    assert header.counts == counts
    assert header.name == name


def test_data_files_present() -> None:
    assert len(list(DATA.glob("pa1/*.txt"))) == 51
    assert len(list(DATA.glob("pa2/*.txt"))) == 81


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_every_file_loads_with_header_counts(path: Path) -> None:
    header, _ = io.read_raw(path)
    assert header.name == path.name
    c = header.counts
    kind = _kind(path)
    if kind == "calbody":
        cb = io.read_calbody(path)
        assert cb.d.shape == (c[0], 3)
        assert cb.a.shape == (c[1], 3)
        assert cb.c.shape == (c[2], 3)
    elif kind == "calreadings":
        cr = io.read_calreadings(path)
        assert cr.D.shape == (c[3], c[0], 3)
        assert cr.A.shape == (c[3], c[1], 3)
        assert cr.C.shape == (c[3], c[2], 3)
    elif kind in ("empivot", "em_fiducials", "em_nav"):
        reader = {
            "empivot": io.read_empivot,
            "em_fiducials": io.read_em_fiducials,
            "em_nav": io.read_em_nav,
        }[kind]
        assert reader(path).shape == (c[1], c[0], 3)
    elif kind == "optpivot":
        op = io.read_optpivot(path)
        assert op.D.shape == (c[2], c[0], 3)
        assert op.H.shape == (c[2], c[1], 3)
    elif kind == "output1":
        out = io.read_output1(path)
        assert out.em_post.shape == (3,)
        assert out.opt_post.shape == (3,)
        assert out.C_expected.shape == (c[1], c[0], 3)
    elif kind == "ct_fiducials":
        assert io.read_ct_fiducials(path).shape == (c[0], 3)
    elif kind == "output2":
        assert io.read_output2(path).shape == (c[0], 3)


def test_known_values_pa1_debug_a() -> None:
    cb = io.read_calbody(DATA / "pa1" / "pa1-debug-a-calbody.txt")
    np.testing.assert_allclose(cb.d[1], [0.0, 0.0, 150.0])
    np.testing.assert_allclose(cb.c[-1], [250.0, 250.0, 250.0])
    out = io.read_output1(DATA / "pa1" / "pa1-debug-a-output1.txt")
    np.testing.assert_allclose(out.em_post, [205.88, 195.77, 193.11])
    np.testing.assert_allclose(out.opt_post, [404.19, 390.64, 192.65])


def test_frames_are_split_in_file_order() -> None:
    path = DATA / "pa1" / "pa1-debug-a-calreadings.txt"
    _, raw = io.read_raw(path)
    cr = io.read_calreadings(path)
    per_frame = 8 + 8 + 27
    np.testing.assert_array_equal(cr.D[1, 0], raw[per_frame])
    np.testing.assert_array_equal(cr.A[0, 0], raw[8])
    np.testing.assert_array_equal(cr.C[2, -1], raw[3 * per_frame - 1])


def test_row_count_mismatch_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad-calbody.txt"
    bad.write_text("2, 1, 1, bad-calbody.txt\n1,2,3\n4,5,6\n7,8,9\n")
    with pytest.raises(ValueError, match="header implies 4 rows"):
        io.read_calbody(bad)


def test_data_path_uses_original_spellings() -> None:
    assert io.data_path("d", "pa2-debug-a", "em_fiducials").name == (
        "pa2-debug-a-em-fiducialss.txt"
    )
    assert io.data_path("d", "pa2-debug-a", "em_nav").name == "pa2-debug-a-EM-nav.txt"


def test_pa2_debug_a_has_no_optpivot() -> None:
    assert not io.data_path(DATA / "pa2", "pa2-debug-a", "optpivot").exists()

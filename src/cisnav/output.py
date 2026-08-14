"""Writers for the output files, in the handout's format.

Rows are ``x, y, z`` with two decimals in 8-character fields, matching the
reference files supplied with the course data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cisnav.io import Output1, PathLike


def format_row(p: np.ndarray) -> str:
    x, y, z = np.asarray(p, dtype=float)
    return f"{x:8.2f}, {y:8.2f}, {z:8.2f}"


def write_output1(path: PathLike, out: Output1) -> Path:
    """Write ``NAME-output1.txt``: header, EM post, optical post, then C_expected."""
    path = Path(path)
    n_frames, n_c, _ = out.C_expected.shape
    lines = [f"{n_c}, {n_frames}, {path.name}", format_row(out.em_post), format_row(out.opt_post)]
    lines += [format_row(p) for p in out.C_expected.reshape(-1, 3)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_output2(path: PathLike, tips: np.ndarray) -> Path:
    """Write ``NAME-output2.txt``: header, then one CT tip position per frame."""
    path = Path(path)
    tips = np.asarray(tips, dtype=float).reshape(-1, 3)
    lines = [f"{tips.shape[0]}, {path.name}"] + [format_row(p) for p in tips]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path

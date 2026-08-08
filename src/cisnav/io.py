"""Readers for the PA1 and PA2 data files.

Every file has one header line followed by rows of comma-separated
``x, y, z`` coordinates in millimetres. The header holds a few integer counts
and the file name, but its formatting varies between files (``", "`` or ``","``
separators, sometimes a trailing comma), so headers are parsed by splitting on
commas and dropping empty fields.

Each reader checks that the body has exactly as many rows as the header
promises and reshapes the rows into per-frame arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PathLike = str | Path


@dataclass(frozen=True)
class Header:
    """Integer counts and file name from the first line of a data file."""

    counts: tuple[int, ...]
    name: str


def parse_header(line: str) -> Header:
    """Parse a header such as ``"8, 8, 27, 8, pa1-debug-a-calreadings.txt"``.

    Integer fields become ``counts``; the first non-integer field is the name.
    """
    fields = [f.strip() for f in line.split(",")]
    fields = [f for f in fields if f]
    counts: list[int] = []
    name = ""
    for field in fields:
        try:
            counts.append(int(field))
        except ValueError:
            name = field
            break
    return Header(tuple(counts), name)


def read_raw(path: PathLike) -> tuple[Header, np.ndarray]:
    """Read a data file into its header and an ``(N, 3)`` array of rows."""
    path = Path(path)
    with path.open() as f:
        header = parse_header(f.readline())
        rows = [
            [float(v) for v in line.split(",") if v.strip()]
            for line in f
            if line.strip()
        ]
    body = np.asarray(rows, dtype=float).reshape(-1, 3) if rows else np.empty((0, 3))
    return header, body


def _expect_rows(path: PathLike, body: np.ndarray, expected: int) -> None:
    if body.shape[0] != expected:
        raise ValueError(
            f"{Path(path).name}: header implies {expected} rows, found {body.shape[0]}"
        )


def _expect_counts(path: PathLike, header: Header, n: int) -> tuple[int, ...]:
    if len(header.counts) < n:
        raise ValueError(
            f"{Path(path).name}: expected {n} counts in header, got {header.counts}"
        )
    return header.counts[:n]


@dataclass(frozen=True)
class CalBody:
    """Calibration object geometry.

    d: optical markers on the EM base, in EM base coordinates, ``(N_D, 3)``.
    a: optical LEDs on the calibration object, in object coordinates, ``(N_A, 3)``.
    c: EM markers on the calibration object, in object coordinates, ``(N_C, 3)``.
    """

    d: np.ndarray
    a: np.ndarray
    c: np.ndarray


@dataclass(frozen=True)
class CalReadings:
    """Per-frame tracker readings of the calibration object.

    D: optical tracker readings of the EM base markers, ``(N_frames, N_D, 3)``.
    A: optical tracker readings of the object LEDs, ``(N_frames, N_A, 3)``.
    C: EM tracker readings of the object EM markers, ``(N_frames, N_C, 3)``.
    """

    D: np.ndarray
    A: np.ndarray
    C: np.ndarray


@dataclass(frozen=True)
class OptPivot:
    """Optical pivot calibration readings.

    D: EM base markers seen by the optical tracker, ``(N_frames, N_D, 3)``.
    H: optical probe markers seen by the optical tracker, ``(N_frames, N_H, 3)``.
    """

    D: np.ndarray
    H: np.ndarray


@dataclass(frozen=True)
class Output1:
    """PA1 output: post positions from both pivots and per-frame expected C."""

    em_post: np.ndarray
    opt_post: np.ndarray
    C_expected: np.ndarray


def read_calbody(path: PathLike) -> CalBody:
    header, body = read_raw(path)
    n_d, n_a, n_c = _expect_counts(path, header, 3)
    _expect_rows(path, body, n_d + n_a + n_c)
    return CalBody(
        d=body[:n_d], a=body[n_d : n_d + n_a], c=body[n_d + n_a :]
    )


def read_calreadings(path: PathLike) -> CalReadings:
    header, body = read_raw(path)
    n_d, n_a, n_c, n_frames = _expect_counts(path, header, 4)
    per_frame = n_d + n_a + n_c
    _expect_rows(path, body, per_frame * n_frames)
    frames = body.reshape(n_frames, per_frame, 3)
    return CalReadings(
        D=frames[:, :n_d],
        A=frames[:, n_d : n_d + n_a],
        C=frames[:, n_d + n_a :],
    )


def _read_marker_frames(path: PathLike) -> np.ndarray:
    """Read files whose header is ``N_markers, N_frames, name``."""
    header, body = read_raw(path)
    n_markers, n_frames = _expect_counts(path, header, 2)
    _expect_rows(path, body, n_markers * n_frames)
    return body.reshape(n_frames, n_markers, 3)


def read_empivot(path: PathLike) -> np.ndarray:
    """EM probe marker readings ``G`` during pivoting, ``(N_frames, N_G, 3)``."""
    return _read_marker_frames(path)


def read_em_fiducials(path: PathLike) -> np.ndarray:
    """EM probe readings ``G`` while touching each fiducial, ``(N_B, N_G, 3)``."""
    return _read_marker_frames(path)


def read_em_nav(path: PathLike) -> np.ndarray:
    """EM probe readings ``G`` for navigation frames, ``(N_frames, N_G, 3)``."""
    return _read_marker_frames(path)


def read_optpivot(path: PathLike) -> OptPivot:
    header, body = read_raw(path)
    n_d, n_h, n_frames = _expect_counts(path, header, 3)
    per_frame = n_d + n_h
    _expect_rows(path, body, per_frame * n_frames)
    frames = body.reshape(n_frames, per_frame, 3)
    return OptPivot(D=frames[:, :n_d], H=frames[:, n_d:])


def read_ct_fiducials(path: PathLike) -> np.ndarray:
    """Fiducial positions ``b`` in CT coordinates, ``(N_B, 3)``."""
    header, body = read_raw(path)
    (n_b,) = _expect_counts(path, header, 1)
    _expect_rows(path, body, n_b)
    return body


def read_output1(path: PathLike) -> Output1:
    header, body = read_raw(path)
    n_c, n_frames = _expect_counts(path, header, 2)
    _expect_rows(path, body, 2 + n_c * n_frames)
    return Output1(
        em_post=body[0],
        opt_post=body[1],
        C_expected=body[2:].reshape(n_frames, n_c, 3),
    )


def read_output2(path: PathLike) -> np.ndarray:
    """Probe tip positions in CT coordinates, ``(N_frames, 3)``."""
    header, body = read_raw(path)
    (n_frames,) = _expect_counts(path, header, 1)
    _expect_rows(path, body, n_frames)
    return body


SUFFIXES = {
    "calbody": "calbody",
    "calreadings": "calreadings",
    "empivot": "empivot",
    "optpivot": "optpivot",
    "output1": "output1",
    "output2": "output2",
    "ct_fiducials": "ct-fiducials",
    "em_fiducials": "em-fiducialss",
    "em_nav": "EM-nav",
}


def data_path(data_dir: PathLike, prefix: str, kind: str) -> Path:
    """Path of one file of a set, e.g. ``data_path("data/pa2", "pa2-debug-a", "em_nav")``.

    ``kind`` is a key of ``SUFFIXES``; the on-disk names keep the original
    spellings (``em-fiducialss``, ``EM-nav``).
    """
    return Path(data_dir) / f"{prefix}-{SUFFIXES[kind]}.txt"

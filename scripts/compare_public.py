"""Compare this implementation with other public CIS I PA1/PA2 solutions.

Usage:
    python scripts/compare_public.py [--clones DIR] [--results-dir DIR]

Fetches the repositories in ``SOURCES`` at pinned commits into ``--clones``
(default ``.cache/public``) and writes ``results/public_comparison.md``.

Every solution is scored the same way: its own output files against the
official reference for the data it was run on, and this program's output for
the same data against the same reference. The course data changes from year
to year, so for a repository with another year's data this program is run on
that repository's input files. Only sets where both sides have an output and
a reference exists are compared.

References are the debug-set answer files shipped with the data and, for the
data in ``data/``, the unknown-set answers released after grading. A released
answer is used only if two independent repositories hold identical copies.
Errors are 3D distances in millimetres: the EM and optical post positions for
PA1, and the worst probe tip in CT coordinates over the navigation frames for
PA2. C_expected is not scored, because the reference C_expected itself
contains EM error (see ``results/pa1_validation.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cisnav import pa1, pa2

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Entry:
    """One assignment's output folder in a repository and its data folder."""

    assignment: str
    outputs: str
    data: str
    npy: bool = False


@dataclass(frozen=True)
class Source:
    repo: str
    commit: str
    entries: tuple[Entry, ...] = ()


SOURCES = (
    Source("wuzijian1997/CIS1-Programming-Assignment", "0ed91a879e309acca368a8a930d36bc2b9cc3e2b", (
        Entry("pa1", "CIS_PA1/OUTPUT", "CIS_PA1/PROGRAM/data"),
        Entry("pa2", "CIS_PA2/OUTPUT", "CIS_PA2/PROGRAM/data"),
    )),
    Source("SeanSDarcy2001/CISProgrammingAssignments", "d4f7f631c80241d9c4e361ad4380096c8f80b4fe", (
        Entry("pa1", "PA2/outputs", "PA2/data"),
        Entry("pa2", "PA2/outputs", "PA2/data"),
    )),
    Source("Zhiyuan-Ding/JHU-CIS1-PA2", "b972edd9cb49c978f7db843cbb29d7d2ef2e7c76", (
        Entry("pa2", "OUTPUT", "ORIGINAL DATA", npy=True),
    )),
    Source("sameraslan/Computer-Integrated-Surgery-Projects", "21df06ca9317df7ef2bb4170522a8b0ac153358b", (
        Entry("pa2", "PA1 & 2/OUTPUT", "PA1 & 2/DATA"),
    )),
    Source("justiin-wang/cis-f25", "0fbb0bfdb66bfcacd7f97fc99a3e6463f2ad6b77", (
        Entry("pa1", "prhw1/out", "prhw1/data"),
        Entry("pa2", "prhw2/out", "prhw2/data"),
    )),
    Source("endernac/Computer_Integrated_Surgery", "b4611c7a5c3f38b233324ead595e63322a5b37e6", (
        Entry("pa1", "PA1/programs/output", "PA1/programs/data"),
        Entry("pa2", "PA2/programs/output", "PA2/programs/data"),
    )),
    Source("dlezcan1/cis1", "373bd993ad4d7adac042a43dcebfce73d9e52d85", (
        Entry("pa1", "pa1_results", "pa1-2_data"),
        Entry("pa2", "pa2_results", "pa1-2_data"),
    )),
    Source("cmicek1/CIS", "7948affff4337913013607c2e3487a7b7d5bb954", (
        Entry("pa2", "PA2/OUTPUT", "PA2/PROGRAMS/PA12 - Student Data"),
    )),
    Source("ahundt/cis", "bd55e8c77ec78994454247ffe7d67f537710a53f", (
        Entry("pa1", "OUTPUT", "data/PA1-2"),
        Entry("pa2", "OUTPUT", "data/PA1-2"),
    )),
    Source("JiaheXu/CIS", "01881beb8db270ecda917af07c2588ed66dddbc9"),
)

# Folders holding the released unknown-set answers for the data in data/.
ANSWER_COPIES = (
    ("sameraslan/Computer-Integrated-Surgery-Projects", "PA1 & 2/DATA"),
    ("Zhiyuan-Ding/JHU-CIS1-PA2", "ORIGINAL DATA"),
    ("JiaheXu/CIS", "CIS1_PA1/DATA"),
)

SET_RE = re.compile(r"^(pa[12])-(debug|unknown)-([a-z])-(.+)\.txt$", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def fetch(source: Source, clones: Path) -> Path:
    """Shallow checkout of ``source`` at its pinned commit; reuses a matching one."""
    dest = clones / source.repo.replace("/", "_")
    if (dest / ".git").exists():
        head = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        if head == source.commit:
            return dest
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    git = ["git", "-C", str(dest)]
    subprocess.run([*git, "init", "-q"], check=True)
    url = f"https://github.com/{source.repo}"
    subprocess.run([*git, "fetch", "-q", "--depth", "1", url, source.commit], check=True)
    subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], check=True)
    return dest


def read_points(path: Path) -> np.ndarray:
    """Rows of 3 numbers after the header line, whatever the separators."""
    lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    return np.array([[float(x) for x in NUMBER_RE.findall(ln)[:3]] for ln in lines[1:]])


def set_files(folder: Path) -> dict[tuple[str, str], Path]:
    """``(prefix, kind) -> path`` for every course file in ``folder``.

    Prefixes are lower case (``pa2-debug-c``) and kinds use the spellings in
    ``data/``, so ``-OUTPUT1``, ``-output-1`` and ``-output1`` all become
    ``output1``.
    """
    found = {}
    for path in folder.iterdir():
        m = SET_RE.match(path.name)
        if m:
            kind = m[4].lower().replace("output-", "output")
            found[(f"{m[1]}-{m[2]}-{m[3]}".lower(), "EM-nav" if kind == "em-nav" else kind)] = path
    return found


def digest(path: Path) -> str:
    """Hash of the numbers in a file, so separators and trailing commas don't matter."""
    numbers = NUMBER_RE.findall(path.read_text(errors="replace"))
    return hashlib.md5(" ".join(numbers).encode()).hexdigest()


def same_inputs(a: dict, b: dict) -> bool:
    """True if every input file present in both folders is identical."""
    shared = [k for k in a if k in b and not k[1].startswith("output")]
    return bool(shared) and all(digest(a[k]) == digest(b[k]) for k in shared)


def released_answers(checkouts: dict[str, Path]) -> dict[tuple[str, str], Path]:
    """Unknown-set answer files that at least two repositories hold identically."""
    copies: dict[tuple[str, str], list[Path]] = {}
    for repo, folder in ANSWER_COPIES:
        for key, path in set_files(checkouts[repo] / folder).items():
            if "unknown" in key[0] and key[1].startswith("output"):
                copies.setdefault(key, []).append(path)
    return {
        key: paths[0]
        for key, paths in copies.items()
        if len(paths) >= 2 and len({digest(p) for p in paths}) == 1
    }


def stage(files: dict[tuple[str, str], Path]) -> Path:
    """Copy course files to a temporary folder under the names ``cisnav.io`` expects."""
    tmp = Path(tempfile.mkdtemp(prefix="cisnav-public-"))
    for (prefix, kind), path in files.items():
        shutil.copy(path, tmp / f"{prefix}-{kind}.txt")
    return tmp


def their_output(entry: Entry, folder: Path, prefix: str, kind: str) -> np.ndarray | None:
    if entry.npy:
        _, status, letter = prefix.split("-")
        path = folder / status / f"{letter}_G_ct_nav.npy"
        return np.load(path).reshape(-1, 3) if kind == "output2" and path.exists() else None
    path = set_files(folder).get((prefix, kind))
    return read_points(path) if path else None


def our_output(assignment: str, prefix: str, same: bool, staged: Path) -> np.ndarray:
    """This program's posts (PA1) or CT tips (PA2) for one set.

    Rounded to 0.01 mm like the files ``cisnav.output`` writes, so both sides
    are scored on what would be handed in.
    """
    if same:
        kind = "output1" if assignment == "pa1" else "output2"
        return read_points(ROOT / "output" / f"{prefix}-{kind}.txt")
    if assignment == "pa1":
        r = pa1.run(staged, prefix)
        return np.round(np.stack([r.em.p_post, r.optical.p_post]), 2)
    return np.round(pa2.run(staged, prefix).nav_ct, 2)


@dataclass
class Score:
    set_name: str
    theirs: tuple[float, ...]
    ours: tuple[float, ...]


def errors(assignment: str, out: np.ndarray, ref: np.ndarray) -> tuple[float, ...]:
    """PA1: (EM post, optical post) errors. PA2: (worst tip error,)."""
    if assignment == "pa1":
        return tuple(float(np.linalg.norm(out[i] - ref[i])) for i in (0, 1))
    return (float(np.linalg.norm(out[: len(ref)] - ref, axis=1).max()),)


def score_entry(
    entry: Entry, checkout: Path, answers: dict, own_data: dict
) -> tuple[list[Score], bool]:
    """Scores for every set with a reference, their output and ours; and whether
    the data is the same as in ``data/``."""
    data = set_files(checkout / entry.data)
    same = same_inputs(data, own_data[entry.assignment])
    kind = "output1" if entry.assignment == "pa1" else "output2"
    refs = {p: f for (p, k), f in data.items() if k == kind and "-debug-" in p}
    if same:
        refs.update({p: f for (p, k), f in answers.items() if k == kind and p.startswith(entry.assignment)})
    staged = stage(data)
    scores = []
    for prefix in sorted(p for p in refs if p.startswith(entry.assignment)):
        ref = read_points(refs[prefix])
        theirs = their_output(entry, checkout / entry.outputs, prefix, kind)
        if theirs is None or len(theirs) < (2 if kind == "output1" else len(ref)):
            continue
        try:
            ours = our_output(entry.assignment, prefix, same, staged)
        except (FileNotFoundError, ValueError):
            continue
        scores.append(Score(
            prefix.split("-", 1)[1],
            errors(entry.assignment, theirs, ref),
            errors(entry.assignment, ours, ref),
        ))
    shutil.rmtree(staged)
    return scores, same


def _fmt(x: float) -> str:
    return f"{x:.3f}" if x < 10 else f"{x:.1f}"


def _within(values: list[float], tol: float) -> str:
    return f"{sum(v <= tol for v in values)}/{len(values)}"


def report(rows: list[tuple[Source, Entry, list[Score], bool]], n_answers: int) -> str:
    pa1_rows = [r for r in rows if r[1].assignment == "pa1" and r[2]]
    pa2_rows = [r for r in rows if r[1].assignment == "pa2" and r[2]]
    data = {True: "same as `data/`", False: "own year"}
    out = [
        "# Comparison with other public solutions",
        "",
        "Generated by `python scripts/compare_public.py`. Each solution's own",
        "output files are scored against the official reference for the data",
        "they were run on, and this program is scored on the same sets against",
        "the same reference. Values are 3D distances in millimetres. The",
        "reference files are rounded to 0.01 mm, so up to about 0.017 mm is",
        "rounding.",
        "",
        "\"Own year\" means that repository's course data differs from `data/`,",
        "so this program was run on its input files. Where the data is the same",
        "as `data/`, the unknown sets are scored too, against the answers",
        f"released after grading ({n_answers} files, each found identical in two",
        "independent repositories).",
        "",
        "These numbers come from the files committed to each repository, which",
        "may not be what was finally submitted or graded. Errors of tens of",
        "millimetres or more mean the committed file does not correspond to the",
        "reference at all (none of them matches any reference of any year within",
        "10 mm), most likely an unfinished run rather than a small inaccuracy.",
        "",
        "## PA2: probe tip in CT coordinates",
        "",
        "For each set, the worst tip error over its navigation frames; then the",
        "median and worst of these over the sets compared, and how many sets are",
        "within 0.1 mm.",
        "",
        "| Repository | Data | Sets | Median: theirs / this repo | Worst: theirs / this repo | Within 0.1 mm: theirs / this repo |",
        "|---|---|---|---|---|---|",
    ]
    for src, _, scores, same in pa2_rows:
        t = [s.theirs[0] for s in scores]
        o = [s.ours[0] for s in scores]
        out.append(
            f"| [{src.repo}](https://github.com/{src.repo}) | {data[same]} | {len(scores)} "
            f"| {_fmt(float(np.median(t)))} / {_fmt(float(np.median(o)))} "
            f"| {_fmt(max(t))} / {_fmt(max(o))} | {_within(t, 0.1)} / {_within(o, 0.1)} |"
        )
    out += [
        "",
        "## PA1: post positions from the pivot calibrations",
        "",
        "Worst error over the sets compared, and how many sets are within",
        "rounding (0.017 mm).",
        "",
        "| Repository | Data | Sets | EM post worst: theirs / this repo | Optical post worst: theirs / this repo | Both posts within rounding: theirs / this repo |",
        "|---|---|---|---|---|---|",
    ]
    for src, _, scores, same in pa1_rows:
        em_t, em_o = [s.theirs[0] for s in scores], [s.ours[0] for s in scores]
        op_t, op_o = [s.theirs[1] for s in scores], [s.ours[1] for s in scores]
        both_t = [max(s.theirs) for s in scores]
        both_o = [max(s.ours) for s in scores]
        out.append(
            f"| [{src.repo}](https://github.com/{src.repo}) | {data[same]} | {len(scores)} "
            f"| {_fmt(max(em_t))} / {_fmt(max(em_o))} | {_fmt(max(op_t))} / {_fmt(max(op_o))} "
            f"| {_within(both_t, 0.017)} / {_within(both_o, 0.017)} |"
        )
    out += ["", "## Per set", "", "PA2 cells are the worst tip error; PA1 cells are `EM post / optical post`.", ""]
    for src, entry, scores, _ in pa2_rows + pa1_rows:
        out += [f"**{src.repo}, {entry.assignment.upper()}**", "", "| Set | Theirs | This repo |", "|---|---|---|"]
        for s in scores:
            out.append(
                f"| {s.set_name} | {' / '.join(map(_fmt, s.theirs))} | {' / '.join(map(_fmt, s.ours))} |"
            )
        out.append("")
    out += ["## Sources", "", "| Repository | Commit |", "|---|---|"]
    for src in SOURCES:
        out.append(f"| [{src.repo}](https://github.com/{src.repo}) | `{src.commit[:10]}` |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clones", type=Path, default=ROOT / ".cache" / "public")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    checkouts = {src.repo: fetch(src, args.clones) for src in SOURCES}
    answers = released_answers(checkouts)
    own_data = {a: set_files(ROOT / "data" / a) for a in ("pa1", "pa2")}
    rows = []
    for src in SOURCES:
        for entry in src.entries:
            scores, same = score_entry(entry, checkouts[src.repo], answers, own_data)
            rows.append((src, entry, scores, same))
            print(f"{src.repo} {entry.assignment}: {len(scores)} sets ({'same' if same else 'own year'} data)")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    path = args.results_dir / "public_comparison.md"
    path.write_text(report(rows, len(answers)))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

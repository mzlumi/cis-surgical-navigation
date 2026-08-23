"""Effect of each error source on the PA2 pipeline, using the course data.

Usage:
    python scripts/error_sources.py

The debug sets switch on one error source at a time, so running the pipeline
with and without distortion correction on each of them separates what each
source costs and what the correction recovers. Writes:

* ``figures/error_sources.png``: calibration error, pivot residual, FRE and
  navigation error per debug set, without and with distortion correction;
* ``figures/degree_cross_validation.png``: held-out correction error against
  polynomial degree for every PA2 set;
* ``figures/distortion_field.png``: the measured distortion in a slice of the
  workspace and what is left of it after correction;
* ``results/error_sources.md``: the numbers behind the figures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cisnav import io, pa1, pa2  # noqa: E402
from cisnav.distortion import DistortionCorrection, held_out_predictions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {  # handout page 13: distortion, noise, jiggle
    "a": "none",
    "b": "noise",
    "c": "distortion",
    "d": "jiggle",
    "e": "all three",
    "f": "all three",
}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(x) ** 2, axis=-1))))


@dataclass
class Metrics:
    calibration: float
    pivot: float
    fre: float
    nav_max: float | None


def pipeline_metrics(
    data_dir: Path, prefix: str, correction: DistortionCorrection, calibration: float
) -> Metrics:
    """Run pivot, fiducial registration and navigation with a given correction."""

    def path(kind: str) -> Path:
        return io.data_path(data_dir, prefix, kind)

    probe = pa2.dewarped_pivot(io.read_empivot(path("empivot")), correction)
    B = pa2.tip_positions(io.read_em_fiducials(path("em_fiducials")), correction, probe)
    b = io.read_ct_fiducials(path("ct_fiducials"))
    F_reg = pa2.registration_to_ct(B, b)
    nav = F_reg.apply(pa2.tip_positions(io.read_em_nav(path("em_nav")), correction, probe))
    nav_max = None
    if path("output2").exists():
        nav_max = float(np.linalg.norm(nav - io.read_output2(path("output2")), axis=1).max())
    return Metrics(
        calibration=calibration,
        pivot=probe.pivot.residual_rms,
        fre=rms(F_reg.apply(B) - b),
        nav_max=nav_max,
    )


def analyse_set(data_dir: Path, prefix: str) -> tuple[Metrics, Metrics, pa2.DistortionFit]:
    """Metrics without correction (identity) and with the cross-validated correction."""
    cal = io.read_calbody(io.data_path(data_dir, prefix, "calbody"))
    readings = io.read_calreadings(io.data_path(data_dir, prefix, "calreadings"))
    fit = pa2.fit_distortion(cal, readings)
    identity = DistortionCorrection.fit(readings.C, readings.C, degree=1)
    raw = pipeline_metrics(data_dir, prefix, identity, rms(readings.C - fit.C_expected))
    corrected = pipeline_metrics(
        data_dir, prefix, fit.correction, fit.cv_scores[fit.correction.degree]
    )
    return raw, corrected, fit


def plot_error_sources(results: dict[str, tuple[Metrics, Metrics]], path: Path) -> None:
    letters = list(results)
    x = np.arange(len(letters))
    panels = [
        ("calibration", "EM calibration error, RMS (mm)", "(a) Measured C vs C_expected"),
        ("pivot", "Pivot residual, RMS (mm)", "(b) EM pivot calibration"),
        ("fre", "FRE, RMS (mm)", "(c) Fiducial registration to CT"),
        ("nav_max", "Max tip error vs reference (mm)", "(d) Navigation in CT"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (field, ylabel, title) in zip(axes.flat, panels):
        raw = [getattr(results[s][0], field) for s in letters]
        cor = [getattr(results[s][1], field) for s in letters]
        ax.bar(x - 0.2, raw, 0.4, label="no correction", color="#c44e52")
        ax.bar(x + 0.2, cor, 0.4, label="Bernstein correction", color="#4c72b0")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 30)
        ax.set_xticks(x, [f"{s}\n{SOURCES[s]}" for s in letters], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", which="major", alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("PA2 debug sets: what each error source costs and what correction recovers")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_degree_cv(fits: dict[str, pa2.DistortionFit], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, fit in fits.items():
        degrees = sorted(fit.cv_scores)
        scores = [fit.cv_scores[d] for d in degrees]
        style = "-" if name.startswith("debug") else "--"
        ax.plot(degrees, scores, style, marker="o", ms=4, label=name)
        chosen = fit.correction.degree
        ax.plot(chosen, fit.cv_scores[chosen], "k*", ms=10)
    ax.set_yscale("log")
    ax.set_xlabel("Bernstein polynomial degree")
    ax.set_ylabel("Held-out RMS error (mm)")
    ax.set_title("Choosing the degree by 5-fold cross-validation (star: chosen)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distortion_field(data_dir: Path, prefix: str, path: Path) -> tuple[float, float]:
    """Arrows of the distortion in a horizontal slice, before and after correction.

    The corrected panel uses held-out predictions, so every arrow is from a
    fit that never saw that frame. Returns the RMS error before and after for
    the points shown.
    """
    cal = io.read_calbody(io.data_path(data_dir, prefix, "calbody"))
    readings = io.read_calreadings(io.data_path(data_dir, prefix, "calreadings"))
    fit = pa2.fit_distortion(cal, readings)
    corrected = held_out_predictions(readings.C, fit.C_expected, fit.correction.degree)

    truth = fit.C_expected.reshape(-1, 3)
    before = (readings.C - fit.C_expected).reshape(-1, 3)
    after = (corrected - fit.C_expected).reshape(-1, 3)
    z_mid = np.median(truth[:, 2])
    sel = np.abs(truth[:, 2] - z_mid) < 60.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    for ax, err, title, magnify in [
        (axes[0], before, "Measured EM error (no correction)", 4.0),
        (axes[1], after, "Remaining error after correction (held-out frames)", 40.0),
    ]:
        mag = np.linalg.norm(err[sel], axis=1)
        q = ax.quiver(
            truth[sel, 0], truth[sel, 1], err[sel, 0], err[sel, 1], mag,
            angles="xy", scale_units="xy", scale=1.0 / magnify,
            cmap="viridis", width=0.003,
        )
        ax.set_title(
            f"{title}\nRMS {rms(err[sel]):.2f} mm, arrows magnified {magnify:.0f}x",
            fontsize=10,
        )
        ax.set_xlabel("x (mm)")
        ax.set_aspect("equal")
        fig.colorbar(q, ax=ax, label="3D error magnitude (mm)", shrink=0.8)
    axes[0].set_ylabel("y (mm)")
    fig.suptitle(
        f"{prefix}: EM marker errors in the slice |z - {z_mid:.0f}| < 60 mm "
        "(arrows show the xy part)",
        fontsize=11,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rms(before[sel]), rms(after[sel])


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "pa2")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    rows, fits, debug = [], {}, {}
    for name in pa1.available_sets(args.data_dir, "pa2"):
        raw, cor, fit = analyse_set(args.data_dir, f"pa2-{name}")
        fits[name] = fit
        kind, letter = name.split("-")
        if kind == "debug":
            debug[letter] = (raw, cor)
        label = SOURCES.get(letter, "all three") if kind == "debug" else "all three"
        rows.append(
            f"| {name} | {label} | {fit.correction.degree} "
            f"| {raw.calibration:.3f} | {cor.calibration:.3f} "
            f"| {raw.pivot:.3f} | {cor.pivot:.3f} "
            f"| {raw.fre:.3f} | {cor.fre:.3f} "
            f"| {fmt(raw.nav_max)} | {fmt(cor.nav_max)} |"
        )

    plot_error_sources(debug, args.figures_dir / "error_sources.png")
    plot_degree_cv(fits, args.figures_dir / "degree_cross_validation.png")
    field_set = "pa2-debug-f"
    before, after = plot_distortion_field(
        args.data_dir, field_set, args.figures_dir / "distortion_field.png"
    )

    text = "\n".join(
        [
            "# Error sources in the PA2 pipeline",
            "",
            "Generated by `python scripts/error_sources.py`. All values in mm.",
            "Each quantity is given without distortion correction (raw) and with",
            "the cross-validated Bernstein correction (corr).",
            "",
            "- **Calibration**: RMS distance between EM readings and C_expected in",
            "  the calibration frames. The corrected value is the held-out",
            "  cross-validation error, not the training fit.",
            "- **Pivot**: RMS residual of the EM pivot calibration.",
            "- **FRE**: RMS fiducial registration error of `F_reg`.",
            "- **Nav max**: largest distance between our CT tip positions and the",
            "  reference output2 (debug sets only).",
            "",
            "| Set | Error sources | Degree | Calibration raw | Calibration corr "
            "| Pivot raw | Pivot corr | FRE raw | FRE corr | Nav max raw | Nav max corr |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
            *rows,
            "",
            f"Distortion field figure ({field_set}, middle slice): RMS error",
            f"{before:.2f} mm before correction and {after:.2f} mm after.",
            "",
            "![Error sources](../figures/error_sources.png)",
            "",
            "![Degree cross-validation](../figures/degree_cross_validation.png)",
            "",
            "![Distortion field](../figures/distortion_field.png)",
            "",
        ]
    )
    out = args.results_dir / "error_sources.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

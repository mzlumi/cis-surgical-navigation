"""Monte Carlo study of registration and pivot calibration error against noise.

Usage:
    python scripts/monte_carlo.py

Uses the real marker geometries from the course data (the 6-marker EM probe,
the 8 optical markers on the EM base and the 27 EM markers on the calibration
body) with synthetic poses and Gaussian marker noise of standard deviation
``sigma`` per coordinate. Writes:

* ``figures/registration_monte_carlo.png``: rotation error and probe-tip error
  of point-set registration against noise, with the tip error compared to the
  prediction of Fitzpatrick, West and Maurer (1998);
* ``figures/pivot_monte_carlo.png``: post position error of pivot calibration
  against noise for several numbers of frames, and against the tilt range of
  the pivoting motion;
* ``results/monte_carlo.md``: the numbers behind the figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cisnav import io, pa1  # noqa: E402
from cisnav.frames import Frame, random_frame, rot_axis_angle, rot_z  # noqa: E402
from cisnav.pivot import pivot_calibration  # noqa: E402
from cisnav.registration import register  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIGMAS = np.array([0.01, 0.03, 0.1, 0.3, 1.0])


def load_geometry(data_dir: Path) -> dict[str, np.ndarray]:
    """Marker models from pa1-debug-a, plus the EM probe tip offset."""
    prefix = "pa1-debug-a"
    cal = io.read_calbody(io.data_path(data_dir, prefix, "calbody"))
    G = io.read_empivot(io.data_path(data_dir, prefix, "empivot"))
    g = G[0] - G[0].mean(axis=0)
    return {
        "probe": g,
        "tip": pa1.em_pivot(G).t_tip,
        "base": cal.d - cal.d.mean(axis=0),
        "body": cal.c - cal.c.mean(axis=0),
    }


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of the rotation ``R`` in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def predicted_tre(markers: np.ndarray, target: np.ndarray, sigma: float) -> float:
    """RMS target registration error predicted by Fitzpatrick, West and Maurer (1998).

    ``TRE^2 = FLE^2 / N (1 + 1/3 sum_k d_k^2 / f_k^2)`` where ``FLE^2 = 3 sigma^2``
    for isotropic noise, ``d_k`` is the distance of the target from principal
    axis ``k`` of the markers and ``f_k`` is the RMS distance of the markers
    from that axis.
    """
    x = markers - markers.mean(axis=0)
    r = target - markers.mean(axis=0)
    _, axes = np.linalg.eigh(x.T @ x)
    ratio = 0.0
    for u in axes.T:
        f2 = np.mean(np.sum(x**2, axis=1) - (x @ u) ** 2)
        d2 = r @ r - (r @ u) ** 2
        ratio += d2 / f2
    return float(np.sqrt(3.0 * sigma**2 / len(x) * (1.0 + ratio / 3.0)))


def registration_trials(
    markers: np.ndarray, target: np.ndarray, sigma: float, n: int, rng: np.random.Generator
) -> tuple[float, float]:
    """RMS rotation error (deg) and RMS error at ``target`` (mm) over ``n`` trials."""
    angles, tre = [], []
    for _ in range(n):
        F = random_frame(rng, 500.0)
        noisy = F.apply(markers) + rng.normal(scale=sigma, size=markers.shape)
        F_est = register(markers, noisy)
        angles.append(rotation_angle_deg(F_est.R @ F.R.T))
        tre.append(np.linalg.norm(F_est.apply(target) - F.apply(target)))
    return float(np.sqrt(np.mean(np.square(angles)))), float(np.sqrt(np.mean(np.square(tre))))


def pivot_poses(
    rng: np.random.Generator, tip: np.ndarray, post: np.ndarray, k: int, tilt_deg: float
) -> list[Frame]:
    """``k`` probe poses pivoting about ``post``, tilted up to ``tilt_deg`` from a mean pose."""
    base = random_frame(rng).R
    poses = []
    for _ in range(k):
        axis = np.array([*rng.normal(size=2), 0.0])
        R = base @ rot_axis_angle(axis, np.radians(rng.uniform(0, tilt_deg))) @ rot_z(
            rng.uniform(-np.pi / 6, np.pi / 6)
        )
        poses.append(Frame(R, post - R @ tip))
    return poses


def pivot_trials(
    geom: dict[str, np.ndarray],
    sigma: float,
    k: int,
    tilt_deg: float,
    n: int,
    rng: np.random.Generator,
) -> float:
    """RMS post position error (mm) of pivot calibration over ``n`` trials."""
    post = np.array([200.0, 200.0, 200.0])
    errors = []
    for _ in range(n):
        poses = pivot_poses(rng, geom["tip"], post, k, tilt_deg)
        G = np.stack([F.apply(geom["probe"]) for F in poses])
        G = G + rng.normal(scale=sigma, size=G.shape)
        result, _ = pivot_calibration(G)
        errors.append(np.linalg.norm(result.p_post - post))
    return float(np.sqrt(np.mean(np.square(errors))))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "pa1")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    geom = load_geometry(args.data_dir)
    n = args.trials

    # Registration: rotation error for three marker sets, tip error for the probe.
    labels = {"probe": "EM probe (6 markers)", "base": "EM base (8 optical)", "body": "Cal. body (27 EM)"}
    rot = {name: [] for name in labels}
    tip_mc, tip_pred = [], []
    for sigma in SIGMAS:
        for name in labels:
            r, t = registration_trials(geom[name], geom["tip"], sigma, n, rng)
            rot[name].append(r)
            if name == "probe":
                tip_mc.append(t)
                tip_pred.append(predicted_tre(geom["probe"], geom["tip"], sigma))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for name, label in labels.items():
        axes[0].loglog(SIGMAS, rot[name], "o-", label=label)
    axes[0].set_xlabel("Marker noise sigma per coordinate (mm)")
    axes[0].set_ylabel("RMS rotation error (deg)")
    axes[0].set_title("(a) Registration rotation error")
    axes[0].legend(fontsize=8)
    axes[1].loglog(SIGMAS, tip_mc, "o", ms=7, label="Monte Carlo")
    axes[1].loglog(SIGMAS, tip_pred, "k--", label="Fitzpatrick et al. 1998 prediction")
    axes[1].set_xlabel("Marker noise sigma per coordinate (mm)")
    axes[1].set_ylabel("RMS tip error (mm)")
    axes[1].set_title(f"(b) EM probe tip error, tip {np.linalg.norm(geom['tip']):.0f} mm from markers")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "registration_monte_carlo.png", dpi=150)
    plt.close(fig)

    # Pivot calibration: noise sweep for several frame counts, then tilt sweep.
    frame_counts = [6, 12, 24, 48]
    tilt_default = 30.0
    post_err = {
        k: [pivot_trials(geom, s, k, tilt_default, n, rng) for s in SIGMAS] for k in frame_counts
    }
    tilts = np.array([2.0, 5.0, 10.0, 20.0, 30.0, 45.0])
    sigma_tilt, k_tilt = 0.3, 12
    tilt_err = [pivot_trials(geom, sigma_tilt, k_tilt, t, n, rng) for t in tilts]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for k in frame_counts:
        axes[0].loglog(SIGMAS, post_err[k], "o-", label=f"{k} frames")
    axes[0].set_xlabel("Marker noise sigma per coordinate (mm)")
    axes[0].set_ylabel("RMS post position error (mm)")
    axes[0].set_title(f"(a) Pivot calibration vs noise (tilt up to {tilt_default:.0f} deg)")
    axes[0].legend(fontsize=8)
    axes[1].loglog(tilts, tilt_err, "o-", color="#55a868")
    axes[1].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(tilts))
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.3, 0.5, 1, 2, 4]))
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].set_xlabel("Maximum tilt of the probe during pivoting (deg)")
    axes[1].set_ylabel("RMS post position error (mm)")
    axes[1].set_title(f"(b) vs tilt range (sigma {sigma_tilt} mm, {k_tilt} frames)")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "pivot_monte_carlo.png", dpi=150)
    plt.close(fig)

    def row(values: list[float]) -> str:
        return " | ".join(f"{v:.4f}" for v in values)

    sig_head = " | ".join(f"sigma {s:g}" for s in SIGMAS)
    sep = "|---|" + "---|" * len(SIGMAS)
    text = "\n".join(
        [
            "# Monte Carlo study of registration and pivot calibration",
            "",
            f"Generated by `python scripts/monte_carlo.py` ({n} trials per point, seed",
            f"{args.seed}). Marker geometries come from `pa1-debug-a`; poses and noise",
            "are synthetic. Noise is Gaussian with standard deviation sigma (mm) per",
            "coordinate, added to every marker reading. Errors are RMS over trials.",
            "",
            "## Registration rotation error (deg)",
            "",
            f"| Marker set | {sig_head} |",
            sep,
            *[f"| {labels[name]} | {row(rot[name])} |" for name in labels],
            "",
            f"## EM probe tip error after registration (mm), tip {np.linalg.norm(geom['tip']):.1f} mm from the marker centroid",
            "",
            f"| | {sig_head} |",
            sep,
            f"| Monte Carlo | {row(tip_mc)} |",
            f"| Fitzpatrick et al. 1998 prediction | {row(tip_pred)} |",
            "",
            f"## Pivot calibration post error (mm), tilt up to {tilt_default:.0f} deg",
            "",
            f"| Frames | {sig_head} |",
            sep,
            *[f"| {k} | {row(post_err[k])} |" for k in frame_counts],
            "",
            f"## Pivot calibration post error (mm) against tilt range, sigma {sigma_tilt} mm, {k_tilt} frames",
            "",
            "| Max tilt (deg) | " + " | ".join(f"{t:g}" for t in tilts) + " |",
            "|---|" + "---|" * len(tilts),
            f"| Post error | {row(tilt_err)} |",
            "",
            "![Registration Monte Carlo](../figures/registration_monte_carlo.png)",
            "",
            "![Pivot Monte Carlo](../figures/pivot_monte_carlo.png)",
            "",
        ]
    )
    out = args.results_dir / "monte_carlo.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

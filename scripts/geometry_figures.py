"""Figures that show the geometry of the PA2 scenario and of each method.

Usage:
    python scripts/geometry_figures.py [--set debug-f]

Everything is computed by the ``cisnav`` pipeline on one course data set
(default ``pa2-debug-f``, which has all three error sources). Writes:

* ``figures/workspace.png``: the navigation scene in EM tracker coordinates:
  calibration volume, optical tracker, post, CT fiducials and navigation
  frames, with the probe drawn where it was read;
* ``figures/pivot_geometry.png``: the EM probe swung around the post, and the
  scatter of its tip with and without distortion correction;
* ``figures/bernstein_model.png``: the degree-4 Bernstein basis and the fitted
  correction across a slice of the calibration volume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cisnav import io, pa2  # noqa: E402
from cisnav.distortion import bernstein_1d  # noqa: E402
from cisnav.pivot import pivot_calibration, probe_frames  # noqa: E402
from cisnav.registration import register  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, CORRECTED = "#c44e52", "#4c72b0"
POST, FIDUCIAL, NAV, OPTICAL = "#d62728", "#dd8452", "#55a868", "#8172b3"


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(x) ** 2, axis=-1))))


def box_edges(lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
    """The 12 edges of an axis-aligned box, each as a (2, 3) array."""
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    edges = []
    for i, a in enumerate(corners):
        for b in corners[i + 1:]:
            if np.sum(a != b) == 1:
                edges.append(lo + np.stack([a, b]) * (hi - lo))
    return edges


def draw_probe(ax, markers: np.ndarray, tip: np.ndarray, color: str, show_markers: bool = True) -> None:
    """Probe shaft from the marker centroid to the tip, with the markers tied to the centroid."""
    centroid = markers.mean(axis=0)
    if show_markers:
        for m in markers:
            ax.plot(*np.stack([centroid, m]).T, color=color, lw=0.5, alpha=0.6)
        ax.scatter(*markers.T, s=5, color=color, depthshade=False)
    ax.plot(*np.stack([centroid, tip]).T, color=color, lw=1.6)
    ax.scatter(*centroid, s=10, color=color, depthshade=False)


def plot_workspace(data_dir: Path, prefix: str, path: Path) -> None:
    def p(kind: str) -> Path:
        return io.data_path(data_dir, prefix, kind)

    cal = io.read_calbody(p("calbody"))
    readings = io.read_calreadings(p("calreadings"))
    res = pa2.run(data_dir, prefix)
    corr = res.distortion.correction
    expected = res.distortion.C_expected

    # Optical tracker origin in EM base coordinates, F_D^-1 0, for every frame.
    tracker = np.stack([register(cal.d, Dk).inv().p for Dk in readings.D])

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(projection="3d")
    pts = expected.reshape(-1, 3)
    ax.scatter(*pts.T, s=1, color="0.75", alpha=0.35, depthshade=False,
               label=f"calibration samples ({len(expected)} frames x {expected.shape[1]} EM markers)")
    ax.scatter(*expected[len(expected) // 2].T, s=10, color="0.25", depthshade=False,
               label="calibration object, one frame")
    for i, edge in enumerate(box_edges(corr.box.lo, corr.box.hi)):
        ax.plot(*edge.T, color="0.5", lw=0.6, ls="--",
                label="Bernstein fitting box" if i == 0 else None)

    ax.scatter(*tracker.T, s=30, marker="s", color=OPTICAL, depthshade=False,
               label="optical tracker (moves slightly each frame)")
    ax.scatter(0, 0, 0, s=40, marker="s", color="k", depthshade=False,
               label="EM base (origin)")

    G_pivot = corr(io.read_empivot(p("empivot")))
    poses = probe_frames(res.probe.g, G_pivot)
    for Gk, F in zip(G_pivot, poses):
        draw_probe(ax, Gk, F.apply(res.probe.pivot.t_tip), POST, show_markers=False)
    ax.scatter(*res.probe.pivot.p_post, s=60, marker="*", color=POST, depthshade=False,
               label=f"post, probe pivoted in {len(G_pivot)} poses")

    for name, G, tips, color in [
        ("CT fiducial", io.read_em_fiducials(p("em_fiducials")), res.B, FIDUCIAL),
        ("navigation frame", io.read_em_nav(p("em_nav")), res.nav_em, NAV),
    ]:
        for Gk, tip in zip(corr(G), tips):
            draw_probe(ax, Gk, tip, color)
        ax.scatter(*tips.T, s=30, color=color, depthshade=False,
                   label=f"probe tip at {len(tips)} {name}s")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_box_aspect((1, 1, 1.6))
    ax.view_init(elev=18, azim=-58)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title(f"{prefix}: the navigation scene in EM tracker coordinates "
                 "(EM readings dewarped)", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pivot_geometry(data_dir: Path, prefix: str, path: Path) -> tuple[float, float]:
    """Probe poses around the post, and tip scatter raw vs dewarped.

    Returns the RMS pivot residual without and with correction.
    """
    res = pa2.run(data_dir, prefix)
    G_raw = io.read_empivot(io.data_path(data_dir, prefix, "empivot"))
    G_cor = res.distortion.correction(G_raw)

    scatter = {}
    for label, G in [("raw", G_raw), ("dewarped", G_cor)]:
        pivot, g = pivot_calibration(G)
        tips = np.stack([F.apply(pivot.t_tip) for F in probe_frames(g, G)])
        scatter[label] = (tips - pivot.p_post, pivot.residual_rms)

    fig = plt.figure(figsize=(14, 4.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.45)
    ax = fig.add_subplot(grid[0], projection="3d")
    post = res.probe.pivot.p_post
    poses = probe_frames(res.probe.g, G_cor)
    cmap = plt.get_cmap("viridis")
    centroids = G_cor.mean(axis=1) - post
    for k, (Gk, F) in enumerate(zip(G_cor, poses)):
        color = cmap(k / max(len(poses) - 1, 1))
        draw_probe(ax, Gk - post, F.apply(res.probe.pivot.t_tip) - post, color, show_markers=False)
    ax.scatter(0, 0, 0, s=120, marker="*", color=POST, depthshade=False, label="post", zorder=5)
    extent = np.vstack([centroids, np.zeros(3)])
    ax.set_box_aspect(np.ptp(extent, axis=0) + 1e-6)
    ax.set_xlabel("x - post (mm)", fontsize=8)
    ax.set_ylabel("y - post (mm)", fontsize=8)
    ax.set_zlabel("z - post (mm)", fontsize=8, labelpad=-2)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=20, azim=-70)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"(a) {len(poses)} probe poses, coloured by frame:\n"
                 "shaft from the marker centroid to the tip", fontsize=10)

    limit = 1.15 * max(np.abs(scatter["raw"][0][:, :2]).max(), 1e-3)
    for i, (label, color) in enumerate([("raw", RAW), ("dewarped", CORRECTED)]):
        ax2 = fig.add_subplot(grid[i + 1])
        offsets, resid = scatter[label]
        ax2.scatter(offsets[:, 0], offsets[:, 1], s=28, color=color, zorder=3)
        circle = plt.Circle((0, 0), resid, fill=False, color=color, ls="--", lw=1)
        ax2.add_patch(circle)
        ax2.axhline(0, color="0.8", lw=0.8)
        ax2.axvline(0, color="0.8", lw=0.8)
        ax2.set_xlim(-limit, limit)
        ax2.set_ylim(-limit, limit)
        ax2.set_aspect("equal")
        ax2.set_xlabel("tip x - post x (mm)")
        ax2.set_ylabel("tip y - post y (mm)")
        ax2.grid(alpha=0.3)
        name = "without correction" if label == "raw" else "after Bernstein correction"
        ax2.set_title(f"({'bc'[i]}) Tip of each pose {name}\n"
                      f"RMS residual {resid:.2f} mm (dashed circle)", fontsize=10)
    fig.suptitle(f"{prefix}: EM pivot calibration. Every pose should put the tip "
                 "on the same point; distortion spreads them.", fontsize=11, y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return scatter["raw"][1], scatter["dewarped"][1]


def plot_bernstein_model(data_dir: Path, prefix: str, path: Path) -> None:
    cal = io.read_calbody(io.data_path(data_dir, prefix, "calbody"))
    readings = io.read_calreadings(io.data_path(data_dir, prefix, "calreadings"))
    fit = pa2.fit_distortion(cal, readings)
    corr = fit.correction
    N = corr.degree

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    u = np.linspace(0, 1, 300)
    basis = bernstein_1d(u, N)
    for k in range(N + 1):
        ax.plot(u, basis[:, k], lw=2, label=f"$B_{{{N},{k}}}(u)$")
    ax.plot(u, basis.sum(axis=1), "k--", lw=1, label="sum (= 1)")
    ax.set_xlabel("u, coordinate scaled to the fitting box")
    ax.set_ylabel("basis value")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    ax.grid(alpha=0.3)
    ax.set_title(f"(a) The degree-{N} Bernstein basis in one coordinate.\n"
                 f"The 3D model uses all $({N}+1)^3 = {(N + 1) ** 3}$ products "
                 "$B_i(u_x)B_j(u_y)B_k(u_z)$", fontsize=10)

    ax = axes[1]
    lo, hi = corr.box.lo, corr.box.hi
    z_mid = 0.5 * (lo[2] + hi[2])
    xs = np.linspace(lo[0], hi[0], 160)
    ys = np.linspace(lo[1], hi[1], 160)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X, Y, np.full_like(X, z_mid)], axis=-1)
    shift = corr(grid) - grid
    mag = np.linalg.norm(shift, axis=-1)
    im = ax.pcolormesh(X, Y, mag, cmap="magma", shading="auto")
    fig.colorbar(im, ax=ax, label="correction |P(q) - q| (mm)")
    step = 16
    ax.quiver(X[::step, ::step], Y[::step, ::step], shift[::step, ::step, 0],
              shift[::step, ::step, 1], color="white", angles="xy", width=0.004, alpha=0.85)
    C = readings.C.reshape(-1, 3)
    near = np.abs(C[:, 2] - z_mid) < 40.0
    ax.scatter(C[near, 0], C[near, 1], s=2, color="#7fd4ff", alpha=0.6,
               label="measured calibration readings within 40 mm of the slice")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("measured x (mm)")
    ax.set_ylabel("measured y (mm)")
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.85, markerscale=4)
    ax.set_title(f"(b) {prefix}: fitted correction in the slice z = {z_mid:.0f} mm\n"
                 "(arrows show the xy direction of the correction)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "pa2")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--set", default="debug-f", help="PA2 set name, e.g. debug-f")
    args = parser.parse_args(argv)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"pa2-{args.set}"

    plot_workspace(args.data_dir, prefix, args.figures_dir / "workspace.png")
    raw, dewarped = plot_pivot_geometry(args.data_dir, prefix, args.figures_dir / "pivot_geometry.png")
    plot_bernstein_model(args.data_dir, prefix, args.figures_dir / "bernstein_model.png")
    print(f"{prefix}: pivot RMS residual {raw:.3f} mm raw, {dewarped:.3f} mm dewarped")
    print(f"wrote workspace.png, pivot_geometry.png, bernstein_model.png to {args.figures_dir}")


if __name__ == "__main__":
    main()

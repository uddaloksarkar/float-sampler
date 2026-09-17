#!/usr/bin/env python3
"""
Plot a bench_out/<dist>/combined_summary.csv (produced by
run_interval_benchmarks.sh -- see that script and benchmarks/
generate_benchmarks.py) as a log-log figure: TV bound vs. each benchmark
box's mean parameter value, one point per successful row.

Supersedes this file's previous purpose (plotting fpsampler.py's old
lambda-only summary.csv from the retired total_error_runs_*/ output) --
that CSV schema (lambda/regime/delta_e/delta_h/total_error/tv) is unrelated
to main.py's interval-mode CSV_FIELDS (see dist_poisson.py/dist_binomial.py/
dist_hypergeometric.py) that this script now reads, so it isn't preserved
as a second mode here.

Usage:
    python plot_summary.py <dist> [options]
    dist: binomial | poisson | hypergeometric

Looks for <bench-dir>/<dist>/combined_summary.csv by default, matching
run_interval_benchmarks.sh's own output layout (default --bench-dir:
bench_out, this script's dir).
"""
import argparse
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PGF_PREAMBLE = "\n".join([
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
])

# (x-axis label, mean formula) per distribution -- mean is each box's
# representative parameter value, taken as the geometric mean of each
# ranged parameter's own (lo, hi) (matching this project's own convention
# for a box's "center", e.g. dist_common.box_label/with_param_tols), then
# combined the same way dist_*.py's own regime switches are keyed: lambda
# itself for poisson, n*K/N for hypergeometric. Binomial is NOT here -- its
# (n, p) is plotted as a 2D mesh (see make_binomial_mesh_plot) instead of
# collapsed to a single n*p mean, since its benchmark suite is a genuine
# gap-free 2D grid (benchmarks/generate_benchmarks.py's chain_edges) well
# suited to one, and collapsing to n*p was hiding real structure -- the
# scatter plot showed dense vertical bands at fixed n*p, i.e. very
# different TV at the same mean depending on the (n, p) shape.
X_LABELS = {
    "poisson": r"$\lambda$",
    "hypergeometric": r"$n \cdot K / N$",
}


TV_CAP = 1.0   # total variation distance is bounded by 1 by definition; a
               # reported value above that (this project found some -- e.g.
               # poisson's TV reaching 10^150 at extreme lambda) reflects a
               # numerical blowup in the bound's own composition (see
               # dist_common.acceptance_tv's unguarded math.expm1), not a
               # real distance. Capped for plotting so one such point
               # doesn't compress the rest of a log-scale axis into
               # invisibility; the underlying bug is tracked separately,
               # this is a display fix, not a fix to main.py's own output.


def clamp_tv(x):
    return min(x, TV_CAP) if x is not None else None


def geomean(lo, hi):
    lo, hi = float(lo), float(hi)
    if lo <= 0 or hi <= 0:
        return max(lo, hi)
    return math.sqrt(lo * hi)


def row_mean(dist, row):
    if dist == "poisson":
        return geomean(row["lambda_lo"], row["lambda_hi"])
    if dist == "hypergeometric":
        n = geomean(row["n_lo"], row["n_hi"])
        K = geomean(row["K_lo"], row["K_hi"])
        N = geomean(row["N_lo"], row["N_hi"])
        return n * K / N
    raise ValueError(f"unknown dist {dist!r}")


def load_csv(path, dist):
    """Points from every OK row with a parseable mean and tv; a tally of
    why any other row was skipped (outcome != OK, or a missing/zero field
    that made row_mean or tv unparseable -- e.g. an ERROR row's blank tv)."""
    points = []
    skipped = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            outcome = row.get("outcome", "")
            if outcome != "OK" or not row.get("tv"):
                skipped[outcome or "(no outcome)"] = skipped.get(outcome or "(no outcome)", 0) + 1
                continue
            try:
                mean = row_mean(dist, row)
                tv = clamp_tv(float(row["tv"]))
            except (KeyError, ValueError, ZeroDivisionError):
                skipped["unparseable"] = skipped.get("unparseable", 0) + 1
                continue

            def optfloat(key):
                v = row.get(key)
                try:
                    return float(v) if v else None
                except ValueError:
                    return None

            points.append({
                "tag": row.get("tag", ""),
                "mean": mean,
                "tv": tv,
                "eps_floor": optfloat("eps_floor"),
                "eps_accept": optfloat("eps_accept"),
                "ref_tv": clamp_tv(optfloat("ref_tv")),
                "regime": row.get("regime", "") or "",
            })
    points.sort(key=lambda p: p["mean"])
    return points, skipped


def load_binomial_grid(path):
    """Every OK row's (n_lo, n_hi, p_lo, p_hi, tv) -- one cell of the
    gap-free 2D (n, p) grid benchmarks/generate_benchmarks.py's
    binomial_rows() builds via chain_edges. Kept as raw box edges (not
    reduced to a mean) so make_binomial_mesh_plot can lay them out as an
    actual mesh rather than a scatter."""
    cells = []
    skipped = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            outcome = row.get("outcome", "")
            if outcome != "OK" or not row.get("tv"):
                skipped[outcome or "(no outcome)"] = skipped.get(outcome or "(no outcome)", 0) + 1
                continue
            try:
                n_lo, n_hi = float(row["n_lo"]), float(row["n_hi"])
                p_lo, p_hi = float(row["p_lo"]), float(row["p_hi"])
                tv = clamp_tv(float(row["tv"]))
            except (KeyError, ValueError):
                skipped["unparseable"] = skipped.get("unparseable", 0) + 1
                continue
            cells.append((n_lo, n_hi, p_lo, p_hi, tv))
    return cells, skipped


def make_binomial_mesh_plot(cells, args):
    """n on x, p on y, TV bound as the color -- a pcolormesh over the
    benchmark grid's own box edges (not a resampled/interpolated grid: the
    edges are read directly off the data, so cells missing from a partial
    or still-running benchmark run are left blank (NaN) rather than
    silently interpolated over)."""
    import numpy as np
    import matplotlib
    matplotlib.use("pgf" if args.pgf else "Agg")
    rc = {
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (args.width, args.height),
    }
    if args.pgf:
        rc.update({
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": PGF_PREAMBLE,
            "font.family": "serif",
            "text.usetex": True,
        })
    matplotlib.rcParams.update(rc)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    n_edges = sorted({c[0] for c in cells} | {c[1] for c in cells})
    p_edges = sorted({c[2] for c in cells} | {c[3] for c in cells})
    n_idx = {v: i for i, v in enumerate(n_edges)}
    p_idx = {v: i for i, v in enumerate(p_edges)}

    grid = np.full((len(p_edges) - 1, len(n_edges) - 1), np.nan)
    for n_lo, n_hi, p_lo, p_hi, tv in cells:
        if tv > 0:
            grid[p_idx[p_lo], n_idx[n_lo]] = tv

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(n_edges, p_edges, grid, cmap="viridis",
                         shading="flat", norm=LogNorm())
    fig.colorbar(mesh, ax=ax, label="TV bound")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$" if args.pgf else "n")
    ax.set_ylabel(r"$p$" if args.pgf else "p")
    ax.set_title(args.title or "binomial: TV bound over the (n, p) grid")
    fig.tight_layout()
    return fig


def plot_series(ax, xs, ys, label, marker, **kw):
    valid = [(x, y) for x, y in zip(xs, ys)
             if y is not None and math.isfinite(y) and y > 0
             and x is not None and math.isfinite(x) and x > 0]
    if not valid:
        return
    vx, vy = zip(*valid)
    ax.loglog(vx, vy, marker=marker, linestyle="none", label=label, **kw)


_REGIME_MARKERS = ["^", "v", "D", "P", "*", "X", "h"]


def make_plot(points, dist, args):
    import matplotlib
    matplotlib.use("pgf" if args.pgf else "Agg")
    rc = {
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (args.width, args.height),
    }
    if args.pgf:
        rc.update({
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": PGF_PREAMBLE,
            "font.family": "serif",
            "text.usetex": True,
        })
    matplotlib.rcParams.update(rc)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if args.plot_components:
        xs = [p["mean"] for p in points]
        plot_series(ax, xs, [p["eps_floor"] for p in points],
                    r"$\varepsilon_{\mathrm{floor}}$" if args.pgf else "eps_floor",
                    "o", markersize=3, alpha=0.5)
        plot_series(ax, xs, [p["eps_accept"] for p in points],
                    r"$\varepsilon_{\mathrm{accept}}$" if args.pgf else "eps_accept",
                    "s", markersize=3, alpha=0.5)

    # One series per regime (rather than one flat "TV" series) so a regime
    # switch -- e.g. poisson's low/ptrs, binomial's inversion/btrs,
    # hypergeometric's hyp/hrua -- is visually distinguishable, since this
    # project found regime alone changes TV behavior by orders of magnitude
    # at the same nominal scale.
    regimes = sorted({p["regime"] for p in points}) or [""]
    for i, regime in enumerate(regimes):
        sub = [p for p in points if p["regime"] == regime]
        label = f"TV ({regime})" if regime else "TV bound"
        plot_series(ax, [p["mean"] for p in sub], [p["tv"] for p in sub],
                   label, _REGIME_MARKERS[i % len(_REGIME_MARKERS)], markersize=4)

    if not args.no_ref and any(p["ref_tv"] is not None for p in points):
        xs = [p["mean"] for p in points]
        plot_series(ax, xs, [p["ref_tv"] for p in points],
                    "analyticError reference", "x", markersize=4)

    ax.set_xlabel(X_LABELS[dist])
    ax.set_ylabel("Total variation distance bound")
    ax.set_ylim(top=TV_CAP)   # clamp_tv already caps the data at TV_CAP; this
                              # just keeps the axis itself from auto-scaling
                              # past 1, so a capped point visibly sits at the
                              # ceiling rather than the plot looking ordinary
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(args.title or f"{dist}: interval-mode TV bound vs. mean")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot bench_out/<dist>/combined_summary.csv (from "
                    "run_interval_benchmarks.sh): a log-log TV-vs-mean "
                    "scatter for poisson/hypergeometric, an (n, p) TV-bound "
                    "heatmap for binomial."
    )
    parser.add_argument("dist", choices=("binomial", *sorted(X_LABELS)),
                        help="Which sampler's benchmark results to plot")
    parser.add_argument("--bench-dir", type=Path, default=ROOT / "bench_out",
                        help="Directory holding <dist>/combined_summary.csv "
                             "(default: bench_out, matching "
                             "run_interval_benchmarks.sh's own default outdir)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Explicit path to a combined_summary.csv, "
                             "overriding --bench-dir/<dist> lookup")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: <csv's dir>/<dist>.png)")
    parser.add_argument("--pgf", action="store_true",
                        help="Also save a PGF figure (requires pdflatex) "
                             "alongside the default PNG")
    parser.add_argument("--plot-components", action="store_true",
                        help="Include eps_floor and eps_accept series")
    parser.add_argument("--no-ref", action="store_true",
                        help="Omit the analyticError reference series (poisson only)")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--width", type=float, default=5.5,
                        help="Figure width in inches (default: 5.5)")
    parser.add_argument("--height", type=float, default=3.5,
                        help="Figure height in inches (default: 3.5)")
    args = parser.parse_args()

    csv_path = args.csv or (args.bench_dir / args.dist / "combined_summary.csv")
    if not csv_path.exists():
        parser.error(f"File not found: {csv_path} "
                     f"(run ./run_interval_benchmarks.sh {args.dist} first?)")

    is_binomial = args.dist == "binomial"
    if is_binomial:
        data, skipped = load_binomial_grid(csv_path)
        noun = "cell"
    else:
        data, skipped = load_csv(csv_path, args.dist)
        noun = "row"
    if not data:
        parser.error(f"{csv_path} has no OK {noun} with a parseable tv "
                     f"(skipped: {skipped})")
    if skipped:
        print(f"note: skipped {sum(skipped.values())} {noun}(s) {skipped} -- "
              f"only OK rows with a numeric tv are plotted")

    out_base = args.out or (csv_path.parent / f"{args.dist}.png")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    mpl_cache = out_base.parent / ".matplotlib"
    xdg_cache = out_base.parent / ".cache"
    mpl_cache.mkdir(exist_ok=True)
    xdg_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    fig = make_binomial_mesh_plot(data, args) if is_binomial else make_plot(data, args.dist, args)

    out_png = out_base.with_suffix(".png")
    fig.savefig(out_png, dpi=150)
    print(f"Wrote PNG: {out_png}")
    if args.pgf:
        out_pgf = out_base.with_suffix(".pgf")
        fig.savefig(out_pgf, backend="pgf")
        print(f"Wrote PGF: {out_pgf}")

    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()

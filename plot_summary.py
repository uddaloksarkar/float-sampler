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
bench_out, this script's dir). combined_summary.csv only exists once that
script's merge step runs, i.e. after every row in <dist>.tsv has been
attempted -- for a still-running or interrupted job (no combined_summary.csv
yet), this falls back to reading the same per-row status/*.status +
runs/<tag>/summary.csv files run_interval_benchmarks.sh's own merge step
reads, so a live/partial job is plottable too, not just a finished one.

Cross-job cache: every invocation (unless --no-update-cache) also scans
every bench_out* directory at the repo root (see find_source_dirs) and
merges any newly-seen OK row into bench_cache/<dist>.csv -- a persistent,
cross-job ledger of which benchmark tags are already known-good. Each SLURM
submission gets its own bench_out_<jobid> (see xrun_fpsamp.slurm) and starts
from an empty status/ dir, so without this a fresh job would blindly
re-attempt tags earlier jobs already finished. run_interval_benchmarks.sh
reads this cache (read-only, once, before dispatching xargs -- see that
script) to skip tags already known-good, so only genuinely-unfinished
(never-attempted, ERROR, or TIMEOUT) tags get (re)run. The cache is
deliberately only ever written here, not from run_interval_benchmarks.sh
itself: that script's workers run with real parallelism (xargs -P), and
several independent jobs can share one working copy on the cluster --
concurrent writers to one cache file would race. This script is a
single-shot, human-triggered, low-frequency command instead, so a plain
read-merge-atomic-rename is enough.
"""
import argparse
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "bench_cache"

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


def iter_dist_rows(dist_dir):
    """Yield one dict per benchmark row attempted so far in dist_dir (a
    run_interval_benchmarks.sh <dist> output directory) -- 'tag', 'outcome',
    and, for OK rows, every CSV_FIELDS column dist_*.py's own summary.csv
    has (n_lo, tv, regime, ...).

    Prefers dist_dir/combined_summary.csv, the merged file
    run_interval_benchmarks.sh writes once every row in <dist>.tsv has been
    attempted. If that doesn't exist yet (a still-running or interrupted
    job), falls back to reading dist_dir/status/*.status +
    dist_dir/runs/<tag>/summary.csv directly -- the same per-row files that
    script's own merge step reads -- so a live job's results so far are
    still plottable, not just a finished one's.
    """
    combined = dist_dir / "combined_summary.csv"
    if combined.exists():
        with open(combined, newline="") as f:
            yield from csv.DictReader(f)
        return

    status_dir = dist_dir / "status"
    if not status_dir.is_dir():
        return
    for status_file in sorted(status_dir.glob("*.status")):
        parts = status_file.read_text().rstrip("\n").split("\t", 4)
        if len(parts) < 2:
            continue
        tag, outcome = parts[0], parts[1]
        row = {"tag": tag, "outcome": outcome}
        if outcome == "OK":
            run_csv = dist_dir / "runs" / tag / "summary.csv"
            if run_csv.exists():
                with open(run_csv, newline="") as f:
                    sub_rows = list(csv.DictReader(f))
                if sub_rows:
                    row.update(sub_rows[-1])
                else:
                    row["outcome"] = "(empty summary.csv)"
            else:
                row["outcome"] = "(missing summary.csv)"
        yield row


def _cache_path(dist):
    return CACHE_DIR / f"{dist}.csv"


def load_cache(dist):
    """{tag: row_dict} for every OK row currently recorded in
    bench_cache/<dist>.csv, or {} if that cache doesn't exist yet."""
    path = _cache_path(dist)
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {row["tag"]: row for row in csv.DictReader(f) if row.get("tag")}


def find_source_dirs(dist, extra_dir=None):
    """Every bench_out* directory at the repo root with a <dist>/ subdir,
    plus extra_dir (an explicitly-passed --bench-dir that may not follow
    that naming convention, e.g. local ad hoc testing)."""
    dirs = set(ROOT.glob("bench_out*"))
    if extra_dir is not None:
        dirs.add(Path(extra_dir))
    return sorted(d for d in dirs if (d / dist).is_dir())


def update_cache(dist, extra_dir=None):
    """Merge every OK row found across find_source_dirs(dist, extra_dir)
    into bench_cache/<dist>.csv. OK is sticky: a tag already cached as OK is
    never overwritten, even if some job's own status file for it later says
    otherwise (e.g. a stale ERROR from a since-fixed bug) -- the same
    deterministic FPTaylor query already has one verified-good answer, and
    that's all run_interval_benchmarks.sh's cache-skip check needs. Returns
    (n_before, n_after) tag counts. Atomic write (tmp file + os.replace) so
    a reader never sees a half-written cache."""
    merged = load_cache(dist)
    n_before = len(merged)

    for src in find_source_dirs(dist, extra_dir):
        for row in iter_dist_rows(src / dist):
            if row.get("outcome") != "OK" or not row.get("tv"):
                continue
            tag = row.get("tag")
            if not tag or tag in merged:
                continue   # OK is sticky -- first OK seen wins
            merged[tag] = row

    if not merged:
        return n_before, 0

    fieldnames = ["tag", "outcome"]
    seen = set(fieldnames)
    for row in merged.values():
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    CACHE_DIR.mkdir(exist_ok=True)
    tmp = _cache_path(dist).with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        for tag in sorted(merged):
            w.writerow(merged[tag])
    os.replace(tmp, _cache_path(dist))
    return n_before, len(merged)


def load_csv(rows, dist):
    """Points from every OK row with a parseable mean and tv; a tally of
    why any other row was skipped (outcome != OK, or a missing/zero field
    that made row_mean or tv unparseable -- e.g. an ERROR row's blank tv).
    `rows` is any iterable of row dicts -- iter_dist_rows(dist_dir) normally,
    or a plain csv.DictReader for an explicit --csv override."""
    points = []
    skipped = {}
    for row in rows:
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


def load_binomial_grid(rows):
    """Every OK row's (n_lo, n_hi, p_lo, p_hi, tv) -- one cell of the
    gap-free 2D (n, p) grid benchmarks/generate_benchmarks.py's
    binomial_rows() builds via chain_edges. Kept as raw box edges (not
    reduced to a mean) so make_binomial_mesh_plot can lay them out as an
    actual mesh rather than a scatter. `rows` is any iterable of row dicts
    -- see load_csv."""
    cells = []
    skipped = {}
    for row in rows:
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


def load_hyper_grid(rows):
    """Every OK row's (N_lo, N_hi, K_lo, K_hi, n_lo, n_hi, tv) -- one cell of
    benchmarks/generate_benchmarks.py's hyper_rows() grid. Unlike
    load_binomial_grid, this is NOT a single 2D grid: hyper_rows() reparametrizes
    as scale N x shape kappa=K/N x shape nu=n/N, so the full space is 3D.
    Kept as raw (N, K, n) box edges (not reduced to a mean) so
    make_hyper_facet_plot can facet by N-band and mesh (K, n) within each band
    -- see that function's docstring for why K/n directly (not kappa/nu) are
    used as each panel's axes. `rows` is any iterable of row dicts -- see
    load_csv."""
    cells = []
    skipped = {}
    for row in rows:
        outcome = row.get("outcome", "")
        if outcome != "OK" or not row.get("tv"):
            skipped[outcome or "(no outcome)"] = skipped.get(outcome or "(no outcome)", 0) + 1
            continue
        try:
            N_lo, N_hi = float(row["N_lo"]), float(row["N_hi"])
            K_lo, K_hi = float(row["K_lo"]), float(row["K_hi"])
            n_lo, n_hi = float(row["n_lo"]), float(row["n_hi"])
            tv = clamp_tv(float(row["tv"]))
        except (KeyError, ValueError):
            skipped["unparseable"] = skipped.get("unparseable", 0) + 1
            continue
        cells.append((N_lo, N_hi, K_lo, K_hi, n_lo, n_hi, tv))
    return cells, skipped


def make_hyper_facet_plot(cells, args):
    """Binomial-style (K, n) TV-bound mesh, faceted into one small-multiple
    panel per N-band. hyper_rows() builds a 3D grid (N x kappa=K/N x nu=n/N),
    so there's no single 2D layout that shows it all at once the way
    make_binomial_mesh_plot does for (n, p) -- this instead groups cells by
    their exact (N_lo, N_hi) band (the same band hyper_rows() generated K and
    n's sub-grid within) and gives each band its own (K, n) mesh panel, all
    on a shared log color scale so panels are visually comparable. K and n
    (not kappa=K/N, nu=n/N) are used as each panel's axes because that's what
    summary.csv actually records per row -- recovering kappa/nu would mean
    re-deriving them from K/N, N/N division on already-rounded integer
    bounds, adding noise for no benefit within a single fixed-N panel.
    A run with many attempted N-bands (up to 66, see benchmarks/
    generate_benchmarks.py) would make a huge, mostly-empty figure this
    early in a job (see --max-panels); this evenly subsamples (log-spaced by
    N) down to that cap rather than truncating to the smallest/first bands,
    so the panels shown still span the full attempted N range."""
    import numpy as np
    import matplotlib
    matplotlib.use("pgf" if args.pgf else "Agg")
    rc = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
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

    bands = {}
    for N_lo, N_hi, K_lo, K_hi, n_lo, n_hi, tv in cells:
        bands.setdefault((N_lo, N_hi), []).append((K_lo, K_hi, n_lo, n_hi, tv))
    band_keys = sorted(bands)

    if len(band_keys) > args.max_panels:
        idx = np.unique(np.linspace(0, len(band_keys) - 1, args.max_panels).round().astype(int))
        chosen = [band_keys[i] for i in idx]
        print(f"note: {len(band_keys)} N-band(s) attempted so far; showing "
              f"{len(chosen)} log-spaced band(s) (--max-panels {args.max_panels})")
        band_keys = chosen

    all_tv = [tv for cs in bands.values() for *_, tv in cs if tv > 0]
    vmin, vmax = (min(all_tv), max(all_tv)) if all_tv else (1e-12, 1.0)
    if vmin == vmax:
        vmin = vmax / 10

    ncols = max(1, math.ceil(math.sqrt(len(band_keys))))
    nrows = max(1, math.ceil(len(band_keys) / ncols))
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                            figsize=(2.2 * ncols, 2.0 * nrows),
                            constrained_layout=True)

    mesh = None
    skipped_panels = 0
    for i, key in enumerate(band_keys):
        ax = axs[i // ncols][i % ncols]
        N_lo, N_hi = key
        band_cells = bands[key]
        K_edges = sorted({c[0] for c in band_cells} | {c[1] for c in band_cells})
        n_edges = sorted({c[2] for c in band_cells} | {c[3] for c in band_cells})
        if len(K_edges) < 2 or len(n_edges) < 2:
            skipped_panels += 1
            ax.axis("off")
            continue
        K_idx = {v: j for j, v in enumerate(K_edges)}
        n_idx = {v: j for j, v in enumerate(n_edges)}
        grid = np.full((len(n_edges) - 1, len(K_edges) - 1), np.nan)
        for K_lo, K_hi, n_lo, n_hi, tv in band_cells:
            if tv > 0:
                grid[n_idx[n_lo], K_idx[K_lo]] = tv
        mesh = ax.pcolormesh(K_edges, n_edges, grid, cmap="viridis",
                             shading="flat", norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"N≈{geomean(N_lo, N_hi):.2g}")

    for i in range(len(band_keys), nrows * ncols):
        axs[i // ncols][i % ncols].axis("off")

    if mesh is not None:
        fig.colorbar(mesh, ax=axs, label="TV bound", shrink=0.8)
    fig.supxlabel(r"$K$" if args.pgf else "K")
    fig.supylabel(r"$n$" if args.pgf else "n")
    fig.suptitle(args.title or
                 "hypergeometric: TV bound over (K, n), faceted by N-band")
    if skipped_panels:
        print(f"note: {skipped_panels} N-band panel(s) had fewer than 2 "
              f"distinct K or n edges and were left blank")
    return fig


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
                             "overriding --bench-dir/<dist> lookup entirely "
                             "(no live/status-dir fallback for this path)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: <csv's dir>/<dist>.png)")
    parser.add_argument("--pgf", action="store_true",
                        help="Also save a PGF figure (requires pdflatex) "
                             "alongside the default PNG")
    parser.add_argument("--plot-components", action="store_true",
                        help="Include eps_floor and eps_accept series")
    parser.add_argument("--no-ref", action="store_true",
                        help="Omit the analyticError reference series (poisson only)")
    parser.add_argument("--mesh", action="store_true",
                        help="hypergeometric only: plot a binomial-style (K, n) "
                             "TV-bound mesh, faceted into one panel per N-band, "
                             "instead of the default TV-vs-mean scatter")
    parser.add_argument("--max-panels", type=int, default=30,
                        help="hypergeometric --mesh only: cap on N-band panels "
                             "(evenly log-spaced bands chosen if more are "
                             "attempted; default: 30)")
    parser.add_argument("--no-update-cache", action="store_true",
                        help="Skip updating bench_cache/<dist>.csv (the "
                             "cross-job ledger run_interval_benchmarks.sh "
                             "reads to skip already-finished tags in a "
                             "future job) from every bench_out* directory")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--width", type=float, default=5.5,
                        help="Figure width in inches (default: 5.5)")
    parser.add_argument("--height", type=float, default=3.5,
                        help="Figure height in inches (default: 3.5)")
    args = parser.parse_args()

    if not args.no_update_cache:
        n_before, n_after = update_cache(args.dist, extra_dir=args.bench_dir)
        if n_after > n_before:
            n_src = len(find_source_dirs(args.dist, args.bench_dir))
            print(f"cache: bench_cache/{args.dist}.csv now has {n_after} "
                  f"known-good tag(s) (+{n_after - n_before} new, merged "
                  f"from {n_src} bench_out* dir(s))")

    if args.csv:
        if not args.csv.exists():
            parser.error(f"File not found: {args.csv}")
        with open(args.csv, newline="") as f:
            rows = list(csv.DictReader(f))
        out_default_dir = args.csv.parent
        source_desc = str(args.csv)
    else:
        dist_dir = args.bench_dir / args.dist
        rows = list(iter_dist_rows(dist_dir))
        if not rows:
            parser.error(
                f"no data found under {dist_dir} (looked for "
                f"combined_summary.csv, then status/*.status) -- "
                f"run ./run_interval_benchmarks.sh {args.dist} first?")
        live = not (dist_dir / "combined_summary.csv").exists()
        out_default_dir = dist_dir
        source_desc = f"{dist_dir} ({'live/in-progress' if live else 'combined_summary.csv'})"

    is_binomial = args.dist == "binomial"
    is_hyper_mesh = args.dist == "hypergeometric" and args.mesh
    if args.mesh and not is_hyper_mesh:
        parser.error("--mesh only applies to hypergeometric "
                      "(binomial is always plotted as a mesh; poisson has no "
                      "second axis to facet by)")
    if is_binomial:
        data, skipped = load_binomial_grid(rows)
        noun = "cell"
    elif is_hyper_mesh:
        data, skipped = load_hyper_grid(rows)
        noun = "cell"
    else:
        data, skipped = load_csv(rows, args.dist)
        noun = "row"
    if not data:
        parser.error(f"{source_desc} has no OK {noun} with a parseable tv "
                     f"(skipped: {skipped})")
    if skipped:
        print(f"note: skipped {sum(skipped.values())} {noun}(s) {skipped} -- "
              f"only OK rows with a numeric tv are plotted")

    default_name = f"{args.dist}_mesh.png" if is_hyper_mesh else f"{args.dist}.png"
    out_base = args.out or (out_default_dir / default_name)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    mpl_cache = out_base.parent / ".matplotlib"
    xdg_cache = out_base.parent / ".cache"
    mpl_cache.mkdir(exist_ok=True)
    xdg_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    if is_binomial:
        fig = make_binomial_mesh_plot(data, args)
    elif is_hyper_mesh:
        fig = make_hyper_facet_plot(data, args)
    else:
        fig = make_plot(data, args.dist, args)

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

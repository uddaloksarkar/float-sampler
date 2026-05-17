#!/usr/bin/env python3
"""
Read a summary.csv produced by fpsampler.py and write a PGF plot.

Usage:
    python plot_summary.py <summary.csv> [options]
"""
import argparse
import csv
import math
import os
from pathlib import Path


PGF_PREAMBLE = "\n".join([
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
])


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row["regime"]:
                continue
            rows.append(row)
    return rows


def build_points(rows):
    points = []
    for row in rows:
        lam = float(row["lambda"])
        if row["regime"] == "low":
            points.append({
                "lambda": lam,
                "delta_e": None,
                "delta_h": None,
                "total": float(row["low_delta"]),
                "tv": float(row["compute_delta_low_range"]),
            })
        else:
            points.append({
                "lambda": lam,
                "delta_e": float(row["delta_e"]),
                "delta_h": float(row["delta_h"]),
                "total": float(row["total_error"]),
                "tv": float(row["tv"]),
            })
    points.sort(key=lambda p: p["lambda"])
    return points


def plot_series(ax, xs, ys, label, marker, **kw):
    valid = [(x, y) for x, y in zip(xs, ys)
             if y is not None and math.isfinite(y) and y > 0]
    if not valid:
        return
    vx, vy = zip(*valid)
    ax.loglog(vx, vy, marker=marker, label=label, **kw)


def make_plot(points, args):
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": PGF_PREAMBLE,
        "font.family": "serif",
        "text.usetex": True,
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (args.width, args.height),
    })
    import matplotlib.pyplot as plt

    xs = [p["lambda"] for p in points]

    fig, ax = plt.subplots()

    if args.plot_components:
        plot_series(ax, xs, [p["delta_e"] for p in points],
                    label=r"$\Delta_E$", marker="o", markersize=4)
        plot_series(ax, xs, [p["delta_h"] for p in points],
                    label=r"$\Delta_H$", marker="s", markersize=4)

    plot_series(ax, xs, [p["total"] for p in points],
                label=r"FPSampler bound", marker="^", markersize=4)

    if not args.no_tv:
        plot_series(ax, xs, [p["tv"] for p in points],
                    label=r"analyticError bound", marker="x", markersize=4)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Statistical distance bound $\Delta$")
    ax.set_ylim(top=0.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    if args.title:
        ax.set_title(args.title)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot summary.csv from fpsampler.py as a PGF figure."
    )
    parser.add_argument("csv", type=Path, help="Path to summary.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: <csv-stem>.pgf next to the CSV)")
    parser.add_argument("--png", action="store_true",
                        help="Also save a PNG alongside the PGF")
    parser.add_argument("--plot-components", action="store_true",
                        help="Include DeltaE and DeltaH series")
    parser.add_argument("--no-tv", action="store_true",
                        help="Omit the analyticError reference series")
    parser.add_argument("--title", default=None,
                        help="Optional plot title")
    parser.add_argument("--width", type=float, default=5.5,
                        help="Figure width in inches (default: 5.5)")
    parser.add_argument("--height", type=float, default=3.5,
                        help="Figure height in inches (default: 3.5)")
    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(f"File not found: {args.csv}")

    rows = load_csv(args.csv)
    if not rows:
        parser.error("CSV contains no complete rows")

    points = build_points(rows)

    out_pgf = args.out or args.csv.with_suffix(".pgf")
    out_pgf.parent.mkdir(parents=True, exist_ok=True)

    mpl_cache = out_pgf.parent / ".matplotlib"
    xdg_cache = out_pgf.parent / ".cache"
    mpl_cache.mkdir(exist_ok=True)
    xdg_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    fig = make_plot(points, args)
    fig.savefig(out_pgf, backend="pgf")
    print(f"Wrote PGF: {out_pgf}")

    if args.png:
        out_png = out_pgf.with_suffix(".png")
        fig.savefig(out_png, dpi=150)
        print(f"Wrote PNG: {out_png}")

    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
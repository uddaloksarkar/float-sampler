#!/usr/bin/env python3
"""
Main entry point for FP-error analysis of discrete-distribution samplers.

Usage:
  python main.py poisson   lambdas.txt  [common opts]
  python main.py binomial  pairs.txt    [common opts]
  python main.py geometric params.txt   [common opts]   # stub
  python main.py hypergeometric ...     [common opts]   # stub
  python main.py zipf ...               [common opts]   # stub

Common options (shared by all distributions):
  --fptaylor PATH   path to FPTaylor executable
  --fp {fp32,fp64,fp128}
  --out-dir PATH
  --plot            plot TV vs parameter
  --plot-components include error component series
  --plot-pgf        also save PGF
  --plot-file PATH
  -v / --verbose

To add a new distribution:
  1. Create dist_<name>.py exporting NAME, CSV_FIELDS,
     add_args(), default_out_dir(), run(), write_plot().
  2. Import it below and add it to DISTRIBUTIONS.
"""
import argparse
import csv
import sys
from pathlib import Path

from dist_common import (
    add_common_args,
    find_fptaylor, fptaylor_env,
    find_cire,
)

import dist_binomial
import dist_poisson
import dist_geometric
import dist_hypergeometric
import dist_zipf

DISTRIBUTIONS = {
    dist_poisson.NAME:       dist_poisson,
    dist_binomial.NAME:      dist_binomial,
    dist_geometric.NAME:     dist_geometric,
    dist_hypergeometric.NAME: dist_hypergeometric,
    dist_zipf.NAME:          dist_zipf,
}


def main():
    parser = argparse.ArgumentParser(
        description="FP-error analysis for discrete-distribution samplers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        allow_abbrev=False,
    )
    add_common_args(parser)

    subparsers = parser.add_subparsers(dest="dist", metavar="DISTRIBUTION")
    subparsers.required = True
    for name, mod in DISTRIBUTIONS.items():
        sub = subparsers.add_parser(name, help=f"{name} distribution analysis")
        mod.add_args(sub)

    args = parser.parse_args()
    mod = DISTRIBUTIONS[args.dist]

    # --- resolve backend tool ---
    if args.backend == "fptaylor":
        tool = find_fptaylor(args.fptaylor)
        if not tool:
            parser.error("FPTaylor not found; pass --fptaylor or set $FPTAYLOR")
        env = fptaylor_env()
    else:  # cire
        tool = find_cire(args.cire)
        if not tool:
            parser.error(
                "CIRE_LLVM not found; pass --cire or build cire/ (expected at cire/build/CIRE_LLVM)"
            )
        if args.fp != "fp64":
            parser.error("--backend cire only supports --fp fp64")
        import os
        env = os.environ.copy()

    # --- output directories ---
    out_dir = (args.out_dir or mod.default_out_dir(args)).resolve()
    inputs_dir  = out_dir / "inputs"
    outputs_dir = out_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # --- run distribution-specific analysis (or load from cache) ---
    summary_path = out_dir / "summary.csv"
    if args.cache and summary_path.exists():
        print(f"Cache hit: loading {summary_path}")
        with summary_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = mod.run(args, tool, inputs_dir, outputs_dir, env)
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=mod.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote summary: {summary_path}")

    # --- plot ---
    if args.plot:
        plot_path = (args.plot_file or (out_dir / "tv_vs_param.png")).resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        mod.write_plot(rows, plot_path,
                       plot_components=args.plot_components,
                       plot_pgf=args.plot_pgf)
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()

"""
Binomial (legacy inversion) sampler FP-error analysis.
Follows the pattern in dist_geometric.py; called by main.py.
"""
import math
import sys
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_deltas_by_problem,
    run_cire_llvm, extract_cire_abs_error,
)

NAME = "binomial"
CSV_FIELDS = ["n", "p", "eps0", "eps1", "eps2", "tv"]


# ---------------------------------------------------------------------------
# FPTaylor template
# ---------------------------------------------------------------------------

def make_template(n, p, fp):
    """
    Single FPTaylor input for (n, p) with three expressions, one per
    elementary FP operation in legacy_random_binomial_inversion's sampling
    loop (distributions/binomial_legacy_inversion.c):

        qn = exp(n * log(q))                       (initial term, q = 1 - p)
        px = ((n - X + 1) * p * px) / (X * q)       i.e. px = z * (n-X+1)*p / (X*q),  z in (1e-6, 1)
        U -= px                <=>  sum += prod     sum in [qn, 1], prod in [0, 1]

      eps0 : rel. error of qn = exp(n * log(q))
      eps1 : rel. error of px = z * (n - X + 1) * p / (X * q)
      eps2 : rel. error of sum + prod
    """
    q = 1.0 - p
    qn_raw = math.exp(n * math.log(q))
    qn = max(qn_raw, sys.float_info.min)
    z_lo = max(min(qn_raw, math.exp(-22) / math.sqrt(2 * math.pi * n * p * q)),
               sys.float_info.min)
    x_hi = min(float(n), n * p + 10.0 * math.sqrt(n * p * q))
    rnd = FP_TO_FPTAYLOR_RND[fp]

    return (
        "Variables\n"
        f"  real z in [{z_lo:.20e}, 1.0],\n"
        f"  real X in [1.0, {x_hi:.1f}],\n"
        f"  real sum in [{qn:.20e}, 1.0],\n"
        f"  real prod in [0.0, 1.0];\n\n"
        + "Definitions\n"
        f"  n = {float(n):.1f},\n"
        f"  p = {p:.20e},\n"
        f"  q = 1.0 - p,\n"
        f"  qn_step  {rnd}= exp(n * log(q)),\n"
        f"  px_step  {rnd}= z * (n - X + 1) * p / (X * q),\n"
        f"  sum_step {rnd}= sum + prod;\n\n"
        + "Expressions\n"
        f"  eps0 = qn_step;\n"
        f"  eps1 = px_step;\n"
        f"  eps2 = sum_step;\n"
    )


# ---------------------------------------------------------------------------
# CIRE C code
# ---------------------------------------------------------------------------

_BINOM_C = """\
#include <math.h>
/* eps0: absolute error of exp(n * log(1-p)) */
double binom_eps0(double n, double p) { double q = 1.0 - p; return exp(n * log(q)); }
/* eps1: absolute error of z * (n - X + 1) * p / (X * (1-p)) */
double binom_eps1(double z, double X, double n, double p)
    { double q = 1.0 - p; return z * (n - X + 1.0) * p / (X * q); }
/* eps2: absolute error of sum + prod */
double binom_eps2(double s, double pr) { return s + pr; }
"""


def _run_cire(cire, n, p, args, inputs_dir, outputs_dir):
    """Return (eps0, eps1, eps2) relative errors via CIRE absolute errors."""
    q = 1.0 - p
    qn_raw = math.exp(n * math.log(q))
    qn = max(qn_raw, sys.float_info.min)
    z_lo = max(min(qn_raw, math.exp(-22) / math.sqrt(2 * math.pi * n * p * q)),
               sys.float_info.min)
    x_hi = min(float(n), n * p + 10.0 * math.sqrt(n * p * q))

    tag = safe_pair_name(n, p)

    def _run(func, domains, label):
        rc, out = run_cire_llvm(
            cire, _BINOM_C, func, domains, tag, inputs_dir, outputs_dir,
            verbose=args.verbose,
        )
        if rc != 0:
            raise RuntimeError(f"CIRE failed for {label} (n={n}, p={p}); "
                               f"see outputs/{tag}_{func}.out")
        return extract_cire_abs_error(out, label)

    abs0 = _run("binom_eps0",
                [(float(n), float(n)), (p, p)],
                "eps0")
    abs1 = _run("binom_eps1",
                [(z_lo, 1.0), (1.0, x_hi),
                 (float(n), float(n)), (p, p)],
                "eps1")
    abs2 = _run("binom_eps2",
                [(qn, 1.0), (0.0, 1.0)],
                "eps2")

    # relative error = abs_error / lower_bound_of_exact_expression
    # eps0 lower bound: qn (the exact value, single-point expression)
    # eps1 lower bound: minimum of z*(n-X+1)*p/(X*q) at z=z_lo, X=x_hi
    # eps2 lower bound: qn (minimum of sum+prod = qn+0)
    eps1_lo = max(z_lo * (n - x_hi + 1.0) * p / (x_hi * q), sys.float_info.min)
    eps0 = abs0 / qn
    eps1 = abs1 / eps1_lo
    eps2 = abs2 / qn
    return eps0, eps1, eps2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_pair_name(n, p):
    p_str = f"{p:.6g}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"n{n}_p{p_str}"


def read_np_pairs(path):
    pairs = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 2:
            raise ValueError(f"{path}:{lineno}: expected 'n p', got {line!r}")
        try:
            n, p = int(tokens[0]), float(tokens[1])
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid (n, p) values") from exc
        if n <= 0:
            raise ValueError(f"{path}:{lineno}: n must be positive")
        if not (0 < p < 1):
            raise ValueError(f"{path}:{lineno}: p must be in (0, 1)")
        pairs.append((n, p))
    return pairs


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with (n, p) pairs, one per line (format: 'n p')")
    source.add_argument("--n", type=int, default=None, help="Single n value")
    parser.add_argument("--p", type=float, default=None,
                        help="Probability p in (0,1), required with --n")


def default_out_dir(args):
    backend = getattr(args, "backend", "fptaylor")
    if getattr(args, "n", None) is not None:
        return ROOT / f"binomial_runs_{backend}"
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / f"binomial_runs_{backend}"
    return ROOT / f"binomial_runs_{lf.stem}_{backend}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    if getattr(args, "n", None) is not None:
        if args.p is None:
            raise ValueError("--p is required when --n is given")
        if args.n <= 0:
            raise ValueError("--n must be positive")
        if not (0 < args.p < 1):
            raise ValueError("--p must be in (0, 1)")
        pairs = [(args.n, args.p)]
    else:
        pairs = read_np_pairs(args.input_file)
    if not pairs:
        raise ValueError("no (n, p) pairs found in input")

    rows = []
    for n, p in pairs:
        tag = safe_pair_name(n, p)
        try:
            if args.backend == "cire":
                eps0, eps1, eps2 = _run_cire(fptaylor, n, p, args, inputs_dir, outputs_dir)
            else:
                input_path = inputs_dir / f"binomial_inversion_{args.fp}_{tag}.txt"
                input_path.write_text(make_template(n, p, args.fp))
                code, output = run_command(
                    [fptaylor, "--rel-error", "true", str(input_path)],
                    cwd=ROOT, env=env,
                )
                out_path = outputs_dir / f"binomial_inversion_{args.fp}_{tag}.out"
                out_path.write_text(output)
                if args.verbose:
                    print(f"--- FPTaylor binomial_inversion (n={n}, p={p}) ---\n{output}")
                if code != 0:
                    raise RuntimeError(f"FPTaylor failed for n={n}, p={p}; see {out_path}")
                deltas = extract_deltas_by_problem(output, f"n={n} p={p}")
                eps0 = deltas["eps0"]
                eps1 = deltas["eps1"]
                eps2 = deltas["eps2"]

            bound = n * p + 10.0 * math.sqrt(n * p * (1.0 - p))
            tv = 0.5 * (eps0 + eps1 * p + eps2 * bound)

            rows.append({
                "n": n,
                "p": f"{p:.17g}",
                "eps0": f"{eps0:.17e}",
                "eps1": f"{eps1:.17e}",
                "eps2": f"{eps2:.17e}",
                "tv": f"{tv:.17e}",
            })
            print(f"n={n} p={p} eps0={eps0:.6e} eps1={eps1:.6e} eps2={eps2:.6e} TV={tv:.6e}")
        except Exception as exc:
            print(f"WARNING: skipping n={n} p={p}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    import os, contextlib, math
    import numpy as np

    fields = [("eps0", "eps0"), ("eps2", "eps2"), ("TV", "tv")]
    if plot_components:
        fields = [("eps0", "eps0"), ("eps1", "eps1"), ("eps2", "eps2"), ("TV", "tv")]

    # Reparametrize: x = log2(n), y = log2(np) = ne - pe  (both integers).
    # This fills a dense rectangle instead of a thin diagonal band.
    ne_vals  = sorted({round(math.log2(float(r["n"]))) for r in rows})
    mnp_vals = sorted({round(math.log2(float(r["n"]) * float(r["p"]))) for r in rows})
    ne_idx   = {v: i for i, v in enumerate(ne_vals)}
    mnp_idx  = {v: i for i, v in enumerate(mnp_vals)}

    def make_grid(key):
        grid = np.full((len(mnp_vals), len(ne_vals)), np.nan)
        for r in rows:
            ne  = round(math.log2(float(r["n"])))
            mnp = round(math.log2(float(r["n"]) * float(r["p"])))
            v   = float(r[key])
            if math.isfinite(v) and v > 0:
                grid[mnp_idx[mnp], ne_idx[ne]] = math.log10(v)
        return grid

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        flat_axes = axes.flat

        # y-axis tick labels: np = 2^mnp
        mnp_labels = [f"$2^{{{v}}}$" for v in mnp_vals]

        grids = [(label, make_grid(key)) for label, key in fields]
        vmin = min(np.nanmin(g) for _, g in grids)
        vmax = max(np.nanmax(g) for _, g in grids)

        for ax, (label, grid) in zip(flat_axes, grids):
            im = ax.pcolormesh(ne_vals, mnp_vals, grid,
                               cmap="viridis", vmin=vmin, vmax=vmax,
                               shading="nearest")
            fig.colorbar(im, ax=ax, label=f"log₁₀({label})")
            ax.set_xlabel("log₂(n)")
            ax.set_ylabel("np  (mean)")
            ax.set_yticks(mnp_vals)
            ax.set_yticklabels(mnp_labels)
            ax.set_title(label)

        for ax in list(flat_axes)[len(fields):]:
            ax.set_visible(False)

        fig.suptitle("Binomial FP error heatmap")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        if plot_pgf:
            plt.savefig(plot_path.with_suffix(".pgf"), backend="pgf")
        plt.close()

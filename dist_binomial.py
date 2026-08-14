"""
Binomial (legacy inversion) sampler FP-error analysis.
Follows the pattern in dist_geometric.py; called by main.py.
"""
import math
import sys
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_deltas_by_problem, extract_abs_errors_by_problem,
    run_cire_llvm, extract_cire_abs_error,
    loggam_defs, eps_logv, eps_logus, vprint,
)

NAME = "binomial"
CSV_FIELDS = ["n", "p", "regime", "eps0", "eps1", "eps2", "eps_floor", "eps_accept", "tv"]

_BTRS_SWITCH = 30.0   # n*p threshold: inversion below, BTRS above


# ---------------------------------------------------------------------------
# FPTaylor template
# ---------------------------------------------------------------------------

def inversion_params(n, p):
    """(qn, z_lo, x_hi): the interval bounds the inversion templates use."""
    q = 1.0 - p
    qn_raw = math.exp(n * math.log(q))
    qn = max(qn_raw, sys.float_info.min)
    z_lo = max(min(qn_raw, math.exp(-22) / math.sqrt(2 * math.pi * n * p * q)),
               sys.float_info.min)
    x_hi = min(float(n), n * p + 10.0 * math.sqrt(n * p * q))
    return qn, z_lo, x_hi


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
    qn, z_lo, x_hi = inversion_params(n, p)
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
# BTRS FPTaylor template  (n*p >= _BTRS_SWITCH)
# ---------------------------------------------------------------------------

def btrs_u_range(n, p, slack=1.0):
    """
    (u_lo, u_hi): the u values that can reach the acceptance test.

    btrs.c draws u in [-0.5, 0.5) and rejects outright unless
        k = floor(y(u)),   y(u) = (2*a/us + b)*u + c,   us = 0.5 - |u|
    lands in [0, n]                                  [btrs.c lines 58-64].
    Real and FP evaluation of y differ by at most eps_floor << 1, so floor
    can disagree by at most one integer: it is enough to enclose the u with
    k in [-1, n+1], i.e. y in [-1, n+2).  Outside that window both samplers
    reject, so those u contribute nothing to the total variation.

    Writing t = us, y = +-(a/t - 2*a + 0.5*b - b*t) + c, so each boundary is
    the positive root of  b*t^2 + beta*t - a = 0  with

        beta = 2*a - 0.5*b + (n + 1 + slack - c)     u > 0, y = n + 1 + slack
        beta = 2*a - 0.5*b + (c + slack)             u < 0, y = -slack

    In the BTRS regime a, b > 0, so y is monotone in t on each side of 0
    (dy/dt = -+(a/t^2 + b)) and each quadratic has exactly one positive root
    -- taken in the cancellation-free form 2*a / (beta + sqrt(disc)).
    """
    q   = 1.0 - p
    spq = math.sqrt(n * p * q)
    b   = 1.15 + 2.53 * spq
    a   = -0.0873 + 0.0248 * b + 0.01 * p
    c   = n * p + 0.5
    if a <= 0.0:
        raise ValueError(f"BTRS shape constant a = {a:.6g} <= 0 "
                         f"(n*p*q = {n * p * q:.6g} too small); "
                         "the reachable u range is not a single interval")

    def us_min(gamma):
        beta = 2.0 * a - 0.5 * b + gamma
        t = 2.0 * a / (beta + math.sqrt(beta * beta + 4.0 * a * b))
        # y is extremely steep at the boundary (dy/dt = -+(a/t^2 + b)), so shave
        # a relative 1e-12 off t: rounding in this solve can then only widen the
        # enclosure, never clip a reachable u.  t <= 0.5 since u = 0 (t = 0.5)
        # always satisfies -1 <= c <= n + 2.
        return min(t * (1.0 - 1e-12), 0.5)

    return -(0.5 - us_min(c + slack)), 0.5 - us_min(n + 1.0 + slack - c)


def btrs_k_range(n, p):
    """(k_lo, k_hi) for the accept template; k_lo >= 6 so that k+1 >= 7
    (the Stirling branch of random_loggam is only valid there)."""
    spq = math.sqrt(n * p * (1.0 - p))
    m   = int(math.floor((n + 1) * p))
    return (float(max(6,   int(m - 10 * spq))),
            float(min(n - 1, int(math.ceil(m + 10 * spq)))))


def make_btrs_floor_template(n, p, fp, u_lo, u_hi):
    """
    FPTaylor expression for eps_floor: absolute error of
    (2*a/us + b)*u + c, with us = 0.5 - |u|          [btrs.c line 61]
    """
    spq = math.sqrt(n * p * (1.0 - p))
    b   = 1.15 + 2.53 * spq
    a   = -0.0873 + 0.0248 * b + 0.01 * p
    c   = n * p + 0.5
    rnd = FP_TO_FPTAYLOR_RND[fp]

    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}];\n\n"
        "Definitions\n"
        f"  a_  = {a:.20e},\n"
        f"  b_  = {b:.20e},\n"
        f"  c_  = {c:.20e},\n"
        f"  us_ {rnd}= 0.5 - abs(u),\n"
        f"  btrs_floor {rnd}= (2.0 * a_ / us_ + b_) * u + c_;\n\n"
        "Expressions\n"
        "  eps_floor = btrs_floor;\n"
    )


def make_btrs_accept_template(n, p, fp, u_lo, u_hi, fast=False):
    """
    FPTaylor expression for eps_accept (excluding -log(v), see
    make_logv_template): absolute error of
    h - loggam(k+1) - loggam(n-k+1) + (k-m)*lpq
      - log(alpha) + log(a/us^2 + b),  with us = 0.5 - |u|   [btrs.c line 85]
    lgamma is approximated by inlining random_loggam's x>=7 Stirling branch.

    If fast is True, the -2*log(us_) term is omitted here and its error is
    computed separately (see make_btrs_logus_template) and summed in by the
    caller. This drops u as a shared variable between the two terms, which
    may yield a more conservative (looser) overall bound.
    """
    q     = 1.0 - p
    spq   = math.sqrt(n * p * q)
    b     = 1.15 + 2.53 * spq
    a     = -0.0873 + 0.0248 * b + 0.01 * p
    alpha = (2.83 + 5.1 / b) * spq
    m     = int(math.floor((n + 1) * p))
    h     = math.lgamma(m + 1) + math.lgamma(n - m + 1)
    lpq   = math.log(p / q)
    rnd   = FP_TO_FPTAYLOR_RND[fp]

    k_lo, k_hi = btrs_k_range(n, p)

    # Build loggam Definitions for k+1 and n-k+1
    defs_k,  name_k  = loggam_defs("k + 1.0",           "lgk",  rnd)
    defs_nk, name_nk = loggam_defs(f"{float(n):.1f} - k + 1.0", "lgnk", rnd)

    log_us_term = "" if fast else " - 2.0 * log(us_)"

    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}],\n"
        f"  real k in [{k_lo:.1f}, {k_hi:.1f}];\n\n"
        "Definitions\n"
        f"  a_     = {a:.20e},\n"
        f"  b_     = {b:.20e},\n"
        f"  h_     = {h:.20e},\n"
        f"  m_     = {float(m):.1f},\n"
        f"  lpq_   = {lpq:.20e},\n"
        f"  alpha_ = {alpha:.20e},\n"
        + "\n".join(defs_k)  + "\n"
        + "\n".join(defs_nk) + "\n"
        + f"  us_         {rnd}= 0.5 - abs(u),\n"
        + f"  us_sq_      {rnd}= us_ * us_,\n"
        + f"  log_num_    {rnd}= a_ + b_ * us_sq_,\n"
        + f"  btrs_accept {rnd}= h_ - {name_k} - {name_nk}"
          f" + (k - m_) * lpq_ - log(alpha_) + log(log_num_){log_us_term};\n\n"
        + "Expressions\n"
          "  eps_accept = btrs_accept;\n"
    )


def _run_btrs_fptaylor(fptaylor, n, p, fp, tag, inputs_dir, outputs_dir, env, verbose, fast=False):
    """Run FPTaylor for BTRS and return a partial row dict."""
    q    = 1.0 - p
    spq  = math.sqrt(n * p * q)
    b    = 1.15 + 2.53 * spq
    a    = -0.0873 + 0.0248 * b + 0.01 * p
    alpha = (2.83 + 5.1 / b) * spq
    u_lo, u_hi = btrs_u_range(n, p)
    us_min = min(0.5 - u_hi, 0.5 + u_lo)
    vtail = 1e-10

    k_lo, k_hi = btrs_k_range(n, p)
    vprint(verbose, f"binomial BTRS n={n} p={p}",
           spq=spq, a=a, b=b, c=n * p + 0.5, alpha=alpha,
           u_lo=u_lo, u_hi=u_hi, us_min=us_min,
           k_lo=k_lo, k_hi=k_hi, vtail=vtail)

    floor_input  = inputs_dir  / f"binomial_btrs_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"binomial_btrs_floor_{fp}_{tag}.out"
    floor_input.write_text(make_btrs_floor_template(n, p, fp, u_lo, u_hi))

    code, output = run_command([fptaylor, str(floor_input)], cwd=ROOT, env=env)
    floor_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor BTRS floor (n={n}, p={p}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS floor failed for n={n}, p={p}; see {floor_output}")

    # No u-tail probability to add: [u_lo, u_hi] is the reachable range, and
    # every u outside it is rejected by both the real and the FP sampler.
    eps_floor = 5 * extract_abs_errors_by_problem(output)["eps_floor"]

    accept_input  = inputs_dir  / f"binomial_btrs_accept_{fp}_{tag}.txt"
    accept_output = outputs_dir / f"binomial_btrs_accept_{fp}_{tag}.out"
    accept_input.write_text(make_btrs_accept_template(n, p, fp, u_lo, u_hi, fast=fast))

    code, output = run_command([fptaylor, str(accept_input)], cwd=ROOT, env=env)
    accept_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor BTRS accept (n={n}, p={p}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS accept failed for n={n}, p={p}; see {accept_output}")

    eps_accept = extract_abs_errors_by_problem(output)["eps_accept"] + eps_logv(
        fptaylor, fp, vtail, inputs_dir, outputs_dir, env, verbose,
    )
    if fast:
        # the -2*log(us) query only sees us, so the smaller of the two tails
        # (a superset of the reachable us range) is the right bound to pass
        eps_accept += eps_logus(
            fptaylor, fp, us_min, inputs_dir, outputs_dir, env, verbose,
        )

    accept_iter = alpha / (math.sqrt(2 * math.pi) * spq)  # btrs is renormalized by the modal pmf f(m) = B(m) ~ 1/(sqrt(2*pi)*spq), so the per-iteration acceptance prob is 1/(alpha*f(m)).
    tv = 2 * (eps_floor + vtail) * accept_iter + 2 * eps_accept / (1 - vtail)
    return eps_floor, eps_accept, tv


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
    qn, z_lo, x_hi = inversion_params(n, p)

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
    parser.add_argument("--fast", action="store_true",
                        help="BTRS only: compute the -2*log(us) term of "
                             "eps_accept in a separate FPTaylor query and "
                             "sum it in, decoupling it from the shared "
                             "variable u. Faster, but may yield a more "
                             "conservative (looser) bound.")


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
            if n * p >= _BTRS_SWITCH:
                # ---- BTRS regime (FPTaylor only; CIRE not yet supported) ----
                eps_floor, eps_accept, tv = _run_btrs_fptaylor(
                    fptaylor, n, p, args.fp, tag, inputs_dir, outputs_dir,
                    env, args.verbose, fast=args.fast,
                )
                rows.append({
                    "n": n, "p": f"{p:.17g}", "regime": "btrs",
                    "eps0": "nan", "eps1": "nan", "eps2": "nan",
                    "eps_floor":  f"{eps_floor:.17e}",
                    "eps_accept": f"{eps_accept:.17e}",
                    "tv": f"{tv:.17e}",
                })
                print(f"n={n} p={p} [BTRS] eps_floor={eps_floor:.6e}"
                      f" eps_accept={eps_accept:.6e} TV={tv:.6e}")
            elif args.backend == "cire":
                # ---- inversion regime, CIRE ----
                qn, z_lo, x_hi = inversion_params(n, p)
                bound = n * p + 10.0 * math.sqrt(n * p * (1.0 - p))
                vprint(args.verbose, f"binomial inversion n={n} p={p}",
                       qn=qn, z_lo=z_lo, x_hi=x_hi, bound=bound)
                eps0, eps1, eps2 = _run_cire(fptaylor, n, p, args, inputs_dir, outputs_dir)
                tv = 0.5 * (eps0 + eps1 * p + eps2 * bound)
                rows.append({
                    "n": n, "p": f"{p:.17g}", "regime": "inversion",
                    "eps0": f"{eps0:.17e}", "eps1": f"{eps1:.17e}", "eps2": f"{eps2:.17e}",
                    "eps_floor": "nan", "eps_accept": "nan",
                    "tv": f"{tv:.17e}",
                })
                print(f"n={n} p={p} eps0={eps0:.6e} eps1={eps1:.6e} eps2={eps2:.6e} TV={tv:.6e}")
            else:
                # ---- inversion regime, FPTaylor ----
                qn, z_lo, x_hi = inversion_params(n, p)
                bound = n * p + 10.0 * math.sqrt(n * p * (1.0 - p))
                vprint(args.verbose, f"binomial inversion n={n} p={p}",
                       qn=qn, z_lo=z_lo, x_hi=x_hi, bound=bound)
                input_path = inputs_dir / f"binomial_inversion_{args.fp}_{tag}.txt"
                input_path.write_text(make_template(n, p, args.fp))
                code, output = run_command(
                    [fptaylor, "--rel-error", "true", str(input_path)],
                    cwd=ROOT, env=env,
                )
                out_path = outputs_dir / f"binomial_inversion_{args.fp}_{tag}.out"
                out_path.write_text(output)
                if args.verbose >= 2:
                    print(f"--- FPTaylor binomial_inversion (n={n}, p={p}) ---\n{output}")
                if code != 0:
                    raise RuntimeError(f"FPTaylor failed for n={n}, p={p}; see {out_path}")
                deltas = extract_deltas_by_problem(output, f"n={n} p={p}")
                eps0, eps1, eps2 = deltas["eps0"], deltas["eps1"], deltas["eps2"]
                tv = 0.5 * (eps0 + eps1 * p + eps2 * bound)
                rows.append({
                    "n": n, "p": f"{p:.17g}", "regime": "inversion",
                    "eps0": f"{eps0:.17e}", "eps1": f"{eps1:.17e}", "eps2": f"{eps2:.17e}",
                    "eps_floor": "nan", "eps_accept": "nan",
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

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
)

NAME = "binomial"
CSV_FIELDS = ["n", "p", "regime", "eps0", "eps1", "eps2", "eps_floor", "eps_accept", "tv"]

_BTRS_SWITCH = 30.0   # n*p threshold: inversion below, BTRS above


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
# BTRS FPTaylor template  (n*p >= _BTRS_SWITCH)
# ---------------------------------------------------------------------------

# Coefficients from random_loggam (hypergeometric_hrua.c)
_LOGGAM_A = [
     8.333333333333333e-02, -2.777777777777778e-03,
     7.936507936507937e-04, -5.952380952380952e-04,
     8.417508417508418e-04, -1.917526917526918e-03,
     6.410256410256410e-03, -2.955065359477124e-02,
     1.796443723688307e-01, -1.39243221690590e+00,
]
_LOGGAM_LG2PI = 1.8378770664093453e+00


def _loggam_defs(x_expr, prefix, rnd):
    """
    Return FPTaylor Definitions lines implementing random_loggam(x_expr)
    for x_expr >= 7 (Stirling / asymptotic branch, straight-line).
    prefix must be unique per call-site.  The last entry is the result name.
    """
    a = _LOGGAM_A
    lines = []
    # x2 = (1/x)*(1/x)
    lines.append(f"  {prefix}_x2 {rnd}= (1.0 / ({x_expr})) * (1.0 / ({x_expr})),")
    # Horner from a[9] down to a[0]
    lines.append(f"  {prefix}_h9 = {a[9]:.20e},")
    for i in range(8, -1, -1):
        lines.append(
            f"  {prefix}_h{i} {rnd}= {prefix}_h{i+1} * {prefix}_x2 + {a[i]:.20e},"
        )
    # gl = h0/x + 0.5*log(2pi) + (x-0.5)*log(x) - x
    lines.append(
        f"  {prefix}_gl {rnd}= {prefix}_h0 / ({x_expr})"
        f" + {0.5 * _LOGGAM_LG2PI:.20e}"
        f" + (({x_expr}) - 0.5) * log({x_expr}) - ({x_expr}),"
    )
    return lines, f"{prefix}_gl"


def make_btrs_floor_template(n, p, fp, utail):
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
        f"  real u in [{-(0.5 - utail):.20e}, {0.5 - utail:.20e}];\n\n"
        "Definitions\n"
        f"  a_  = {a:.20e},\n"
        f"  b_  = {b:.20e},\n"
        f"  c_  = {c:.20e},\n"
        f"  us_ {rnd}= 0.5 - abs(u),\n"
        f"  btrs_floor {rnd}= (2.0 * a_ / us_ + b_) * u + c_;\n\n"
        "Expressions\n"
        "  eps_floor = btrs_floor;\n"
    )


def make_btrs_accept_template(n, p, fp, utail, fast=False):
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

    # k_lo >= 6 so that k+1 >= 7 (Stirling branch valid)
    k_lo = float(max(6,   int(m - 10 * spq)))
    k_hi = float(min(n-1, int(math.ceil(m + 10 * spq))))

    # Build loggam Definitions for k+1 and n-k+1
    defs_k,  name_k  = _loggam_defs("k + 1.0",           "lgk",  rnd)
    defs_nk, name_nk = _loggam_defs(f"{float(n):.1f} - k + 1.0", "lgnk", rnd)

    log_us_term = "" if fast else " - 2.0 * log(us_)"

    return (
        "Variables\n"
        f"  real u in [{-(0.5 - utail):.20e}, {0.5 - utail:.20e}],\n"
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


def make_logv_template(fp, vtail):
    """
    Absolute error of -log(v), v in [vtail, 1.0] — split out from
    btrs_accept because folding it into that expression adds v as an
    extra dimension to FPTaylor's joint branch-and-bound search and
    blows up the runtime of the whole eps_accept query.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real v in [{vtail:.1e}, 1.0];\n\n"
        "Definitions\n"
        f"  logv_step {rnd}= - log(v);\n\n"
        "Expressions\n"
        "  eps_logv = logv_step;\n"
    )


_LOGV_EPS_CACHE = {}


def _eps_logv(fptaylor, fp, vtail, inputs_dir, outputs_dir, env, verbose):
    """Absolute error of -log(v); same for every (n, p), so cache it."""
    key = (fp, vtail)
    if key in _LOGV_EPS_CACHE:
        return _LOGV_EPS_CACHE[key]

    input_path = inputs_dir  / f"binomial_btrs_logv_{fp}.txt"
    out_path   = outputs_dir / f"binomial_btrs_logv_{fp}.out"
    input_path.write_text(make_logv_template(fp, vtail))

    code, output = run_command([fptaylor, str(input_path)], cwd=ROOT, env=env)
    out_path.write_text(output)
    if verbose:
        print(f"--- FPTaylor BTRS log(v) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS log(v) failed; see {out_path}")

    eps_logv = extract_abs_errors_by_problem(output)["eps_logv"]
    _LOGV_EPS_CACHE[key] = eps_logv
    return eps_logv


def make_btrs_logus_template(fp, utail):
    """
    Absolute error of -2*log(us_), with us_ = 0.5 - |u|,
    u in [-(0.5-utail), 0.5-utail] — split out of btrs_accept for the
    --fast path (see make_btrs_accept_template).
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real u in [{-(0.5 - utail):.20e}, {0.5 - utail:.20e}];\n\n"
        "Definitions\n"
        f"  us_       {rnd}= 0.5 - abs(u),\n"
        f"  logus_step {rnd}= - 2.0 * log(us_);\n\n"
        "Expressions\n"
        "  eps_logus = logus_step;\n"
    )


_LOGUS_EPS_CACHE = {}


def _eps_logus(fptaylor, fp, utail, inputs_dir, outputs_dir, env, verbose):
    """Absolute error of -2*log(us_); same for every (n, p), so cache it."""
    key = (fp, utail)
    if key in _LOGUS_EPS_CACHE:
        return _LOGUS_EPS_CACHE[key]

    input_path = inputs_dir  / f"binomial_btrs_logus_{fp}.txt"
    out_path   = outputs_dir / f"binomial_btrs_logus_{fp}.out"
    input_path.write_text(make_btrs_logus_template(fp, utail))

    code, output = run_command([fptaylor, str(input_path)], cwd=ROOT, env=env)
    out_path.write_text(output)
    if verbose:
        print(f"--- FPTaylor BTRS log(us) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS log(us) failed; see {out_path}")

    eps_logus = extract_abs_errors_by_problem(output)["eps_logus"]
    _LOGUS_EPS_CACHE[key] = eps_logus
    return eps_logus


def _run_btrs_fptaylor(fptaylor, n, p, fp, tag, inputs_dir, outputs_dir, env, verbose, fast=False):
    """Run FPTaylor for BTRS and return a partial row dict."""
    q    = 1.0 - p
    spq  = math.sqrt(n * p * q)
    b    = 1.15 + 2.53 * spq
    alpha = (2.83 + 5.1 / b) * spq
    utail = 1e-5
    vtail = 1e-10

    floor_input  = inputs_dir  / f"binomial_btrs_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"binomial_btrs_floor_{fp}_{tag}.out"
    floor_input.write_text(make_btrs_floor_template(n, p, fp, utail))

    code, output = run_command([fptaylor, str(floor_input)], cwd=ROOT, env=env)
    floor_output.write_text(output)
    if verbose:
        print(f"--- FPTaylor BTRS floor (n={n}, p={p}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS floor failed for n={n}, p={p}; see {floor_output}")

    eps_floor = 5 * extract_abs_errors_by_problem(output)["eps_floor"] + 2 * utail

    accept_input  = inputs_dir  / f"binomial_btrs_accept_{fp}_{tag}.txt"
    accept_output = outputs_dir / f"binomial_btrs_accept_{fp}_{tag}.out"
    accept_input.write_text(make_btrs_accept_template(n, p, fp, utail, fast=fast))

    code, output = run_command([fptaylor, str(accept_input)], cwd=ROOT, env=env)
    accept_output.write_text(output)
    if verbose:
        print(f"--- FPTaylor BTRS accept (n={n}, p={p}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS accept failed for n={n}, p={p}; see {accept_output}")

    eps_accept = extract_abs_errors_by_problem(output)["eps_accept"] + _eps_logv(
        fptaylor, fp, vtail, inputs_dir, outputs_dir, env, verbose,
    )
    if fast:
        eps_accept += _eps_logus(
            fptaylor, fp, utail, inputs_dir, outputs_dir, env, verbose,
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
                eps0, eps1, eps2 = _run_cire(fptaylor, n, p, args, inputs_dir, outputs_dir)
                bound = n * p + 10.0 * math.sqrt(n * p * (1.0 - p))
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
                eps0, eps1, eps2 = deltas["eps0"], deltas["eps1"], deltas["eps2"]
                bound = n * p + 10.0 * math.sqrt(n * p * (1.0 - p))
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

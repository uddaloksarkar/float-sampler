"""
Binomial sampler FP-error analysis.
High range (n*p >= _BTRS_SWITCH) follows the BTRS algorithm in
distributions/btrs.c (Hormann transformed rejection) and mirrors the PTRS
analysis in dist_poisson.py (eps_floor / eps_accept split, shared -log(v) and
-2*log(us) helpers, --fast flag). Low range (n*p < _BTRS_SWITCH) follows the
legacy inversion loop in distributions/binomial_legacy_inversion.c. Both
regimes analyse the p the sampler actually runs on, min(p, 1-p) (sampler_p).
"""
import math
import sys
import time
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_deltas_by_problem, extract_abs_errors_by_problem,
    loggam_defs, eps_logv, eps_logus, run_fptaylor_query,
    ulp_rnd_op,
    point_ivar,
    hormann_u_at, hormann_proposal_deviation, acceptance_tv,
    vprint, elapsed_since, format_seconds,
    floor_x_abs_tol_vars, accept_x_abs_tol_vars,
    dist_switch,
)

NAME = "binomial"
CSV_FIELDS = ["n", "p", "fp", "regime", "eps0", "eps1", "eps2",
              "eps_floor", "eps_accept", "tv", "time_s"]

# n*p threshold: inversion below, BTRS above -- overridable via
# fptaylor_settings.toml's [binomial].switch (dist_common.dist_switch).
_BTRS_SWITCH = dist_switch(NAME, 30.0)


# ---------------------------------------------------------------------------
# BTRS FPTaylor templates  (n*p >= _BTRS_SWITCH)
# ---------------------------------------------------------------------------

def btrs_consts(n, p):
    """(spq, a, b, c): the setup constants btrs.c computes once per (n, p)."""
    spq = math.sqrt(n * p * (1.0 - p))
    b   = 1.15 + 2.53 * spq
    a   = -0.0873 + 0.0248 * b + 0.01 * p
    return spq, a, b, n * p + 0.5


def btrs_setup_defs(rnd, n_expr, p_expr, accept=False):
    """
    btrs.c's setup block [lines 48-51, 72-76] as FPTaylor Definitions.
    These are derived from (n, p), not free inputs, so writing them as
    expressions keeps them correlated and charges the rounding of the setup
    arithmetic itself.
    """
    d = [f"  spq_   {rnd}= sqrt({n_expr} * {p_expr} * (1.0 - {p_expr})),",
         f"  b_     {rnd}= 1.15 + 2.53 * spq_,",
         f"  a_     {rnd}= -0.0873 + 0.0248 * b_ + 0.01 * {p_expr},",
         f"  c_     {rnd}= {n_expr} * {p_expr} + 0.5,"]
    if accept:
        d += [f"  alpha_ {rnd}= (2.83 + 5.1 / b_) * spq_,",
              f"  lpq_   {ulp_rnd_op(rnd, 'log')}= log({p_expr} / (1.0 - {p_expr})),"]
    return d


def btrs_accept_k_range(n):
    """
    k window the accept query covers: the sampler's entire support [0, n].
    Unlike PTRS (dist_poisson.ptrs_accept_k_range), no boundary margin is
    needed: k1_ = k+1 and nk1_ = n-k+1 are exact functions of the declared
    k, so they stay in [1, n+1] -- safely > 0 -- across the whole range.
    """
    return 0.0, float(n)


def btrs_u_at(n, p, y, consts=None):
    """The u with (2*a/us + b)*u + c = y (see dist_common.hormann_u_at)."""
    _, a, b, c = consts or btrs_consts(n, p)
    return hormann_u_at(a, b, c, y)


_K_SLACK = 1.0            # floor can disagree by one integer: y in [k-1, k+2]


def y_window(k_lo, k_hi, slack=_K_SLACK):
    """y range covering every u mapping to a k in [k_lo, k_hi] (floor(y)=k
    means y in [k, k+1)), widened by `slack` for floor disagreement."""
    return k_lo - slack, k_hi + 1.0 + slack


def clip_u_trunc(u_lo, u_hi, u_trunc):
    """Clip [u_lo, u_hi] so us=0.5-|u| never dips below u_trunc; return
    (u_lo, u_hi, excess), excess being the trimmed u-probability mass
    (charged to TV by the caller, like v_trunc).  Same role as
    dist_poisson.clip_u_trunc, but BTRS charges the actual trimmed mass
    rather than a flat 2*u_trunc.  A window entirely beyond the boundary
    clips to empty (u_lo > u_hi)."""
    excess = 0.0
    lo_edge = u_trunc - 0.5
    if u_lo < lo_edge:
        excess += min(u_hi, lo_edge) - u_lo
        u_lo = lo_edge
    hi_edge = 0.5 - u_trunc
    if u_hi > hi_edge:
        excess += u_hi - max(u_lo, hi_edge)
        u_hi = hi_edge
    return u_lo, u_hi, excess


def make_btrs_floor_template(n, p, fp, u_lo, u_hi):
    """
    FPTaylor expression for eps_floor: absolute error of
    (2*a/us + b)*u + c, with us = 0.5 - |u|   [btrs.c lines 60-61]

    Only u is free; n and p are declared as Variables bracketing their exact
    values (dist_common.exact_bracket) so a, b, c reference them by name
    instead of re-embedding the literals at every occurrence.  u runs over
    its full signed range in one query: the expression is a smooth function
    of u across u = 0, so nothing here needs splitting by sign.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}],\n"
        + point_ivar("n", n) + ",\n"
        + point_ivar("p", p) + ";\n\n"
        + "Definitions\n"
        + "\n".join(btrs_setup_defs(rnd, "n", "p")) + "\n"
        + f"  us_    {rnd}= 0.5 - abs(u),\n"
        + f"  btrs_floor {rnd}= (2.0 * a_ / us_ + b_) * u + c_;\n\n"
        + "Expressions\n"
          "  eps_floor = btrs_floor;\n"
    )


def make_btrs_accept_template(n, p, fp, u_lo, u_hi, k_lo, k_hi, fast=False):
    """
    FPTaylor expression for eps_accept (excluding -log(v), see
    dist_common.make_logv_template): absolute error of
    h - loggam(k+1) - loggam(n-k+1) + (k-m)*lpq - log(alpha)
    + log(a/us^2 + b), with us = 0.5 - |u|   [btrs.c line 85, rearranged so
    that log(v) is alone on the left-hand side].
    loggam is FPTaylor's native lgamma directly (loggam_defs, dist_common.py).
    log(a/us^2 + b) is rewritten as log(a + b*us^2) - 2*log(us) to avoid
    forming 1/us^2 directly when us is small.

    m is tied to (n, p) via a floor encoding fm (m_ = (n+1)*p - fm, fm in
    [0, 1)), computed inside FPTaylor rather than precomputed in Python,
    which can't guarantee matching the compiled sampler's rounding
    bit-for-bit; h = lgamma(m+1) + lgamma(n-m+1) [btrs.c line 77] is
    likewise derived from m_ inside the query, since each lgamma() call's
    own error can be as large as the whole eps_accept bound for m in the
    thousands.

    k is declared directly as a Variable over [k_lo, k_hi] (from
    btrs_accept_k_range) rather than derived here from u via the y = f(u)
    map: k and u are only jointly reachable through the sampler's *exact*
    floor relationship, and eps_floor already bounds any disagreement about
    which k a given u floors to. Re-deriving k from u inside this template
    would needlessly propagate u's own error through that derivative-large
    map into loggam(k+1)/loggam(n-k+1), inflating eps_accept and
    double-counting what eps_floor already covers. Declaring k directly, and
    letting u range over its own full (both-signs) interval only for
    us_/log_num_, is a sound relaxation -- the same reparametrization used
    for PTRS's k (dist_poisson.make_ptrs_accept_template) and HRUA's Z
    (dist_hypergeometric.hrua_z_defs).

    If fast is True, the -2*log(us_) term is omitted here and its error is
    computed separately (see dist_common.make_logus_template) and summed in
    by the caller. This drops u as a shared variable between the two terms,
    which may yield a more conservative (looser) overall bound.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    defs_hm,  name_hm  = loggam_defs("m_ + 1.0", "hm", rnd)
    defs_hnm, name_hnm = loggam_defs("n - m_ + 1.0", "hnm", rnd)
    defs_k,   name_k   = loggam_defs("k1_",  "lgk",  rnd)
    defs_nk,  name_nk  = loggam_defs("nk1_", "lgnk", rnd)
    log_rnd = ulp_rnd_op(rnd, "log")
    log_us_def  = "" if fast else f"  log_us_ {log_rnd}= log(us_),\n"
    log_us_term = "" if fast else " - 2.0 * log_us_"

    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}],\n"
        f"  real k in [{k_lo:.20e}, {k_hi:.20e}],\n"
        + point_ivar("n", n) + ",\n"
        + point_ivar("p", p) + ",\n"
        + "  real fm in [0.0, 1.0];\n\n"
        + "Definitions\n"
        + "\n".join(btrs_setup_defs(rnd, "n", "p", accept=True)) + "\n"
        + "  m_     = (n + 1.0) * p - fm,\n"
        + "\n".join(defs_hm + defs_hnm) + "\n"
        + f"  h_     {rnd}= {name_hm} + {name_hnm},\n"
        + f"  us_    {rnd}= 0.5 - abs(u),\n"
        + "  k1_    = k + 1.0,\n"
        + "  nk1_   = n - k + 1.0,\n"
        + "\n".join(defs_k + defs_nk) + "\n"
        + f"  us_sq_      {rnd}= us_ * us_,\n"
        + f"  log_num_    {rnd}= a_ + b_ * us_sq_,\n"
        + f"  log_alpha_  {log_rnd}= log(alpha_),\n"
        + f"  log_lognum_ {log_rnd}= log(log_num_),\n"
        + log_us_def
        + f"  btrs_accept {rnd}= h_ - {name_k} - {name_nk}"
          f" + (k - m_) * lpq_ - log_alpha_ + log_lognum_{log_us_term};\n\n"
        + "Expressions\n"
          "  eps_accept = btrs_accept;\n"
    )


def _run_btrs_fptaylor(fptaylor, n, p, args, tag, inputs_dir, outputs_dir, env):
    """(eps_floor, eps_accept, tv) for the BTRS regime at (n, p)."""
    fp, verbose = args.fp, args.verbose
    fast, v_trunc = args.fast, args.v_trunc
    ratio_tol, bb_eval = args.bb_geometric_ratio_tol, args.bb_eval
    x_abs_tol, approx = args.opt_x_abs_tol, args.approx
    u_trunc = args.u_trunc
    floor_tol_vars = floor_x_abs_tol_vars(args)
    accept_tol_vars = accept_x_abs_tol_vars(args)
    if u_trunc is None or not (0.0 <= u_trunc < 0.5):
        raise ValueError("BTRS requires --u-trunc with 0 <= u_trunc < 0.5")
    consts = spq, a, b, c = btrs_consts(n, p)
    if a <= 0.0:
        raise ValueError(f"BTRS shape constant a = {a:.6g} <= 0 "
                         f"(n*p*q = {n * p * (1.0 - p):.6g} too small); "
                         "the reachable u range is not a single interval")
    alpha = (2.83 + 5.1 / b) * spq
    accept_iter = alpha / (math.sqrt(2 * math.pi) * spq)

    # k is declared directly over its own interval (btrs_accept_k_range),
    # decoupled from u -- see make_btrs_accept_template's docstring -- and
    # both queries run over the one u window reaching every k in it (the
    # sampler's whole support), clipped by u_trunc; the clipped-off mass is
    # charged to TV as u_excess.
    k_lo, k_hi = btrs_accept_k_range(n)
    y_lo, y_hi = y_window(k_lo, k_hi)
    u_lo, u_hi = btrs_u_at(n, p, y_lo, consts), btrs_u_at(n, p, y_hi, consts)
    us_min = max(min(0.5 + u_lo, 0.5 - u_hi), u_trunc)   # smallest reachable us
    u_lo, u_hi, u_excess = clip_u_trunc(u_lo, u_hi, u_trunc)
    if u_lo > u_hi:
        raise ValueError(f"n={n} p={p}: u-range emptied by u_trunc={u_trunc}")
    vprint(verbose, f"binomial BTRS n={n} p={p}",
           spq=spq, a=a, b=b, c=c, alpha=alpha,
           u_lo=u_lo, u_hi=u_hi, k_lo=k_lo, k_hi=k_hi, us_min=us_min,
           u_excess=u_excess, v_trunc=v_trunc, u_trunc=u_trunc)

    # ---- floor ----
    floor_input  = inputs_dir  / f"binomial_btrs_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"binomial_btrs_floor_{fp}_{tag}.out"
    floor_input.write_text(make_btrs_floor_template(n, p, fp, u_lo, u_hi))

    code, output = run_fptaylor_query(fptaylor, floor_input, outputs_dir, env,
                                       ratio_tol, bb_eval, x_abs_tol, floor_tol_vars, approx)
    floor_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor BTRS floor (n={n} p={p}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS floor failed for n={n} p={p}; see {floor_output}")

    floor_raw = extract_abs_errors_by_problem(output)["eps_floor"]
    eps_floor = hormann_proposal_deviation(floor_raw, a, b)

    # ---- accept ----
    accept_input  = inputs_dir  / f"binomial_btrs_accept_{fp}_{tag}.txt"
    accept_output = outputs_dir / f"binomial_btrs_accept_{fp}_{tag}.out"
    accept_input.write_text(
        make_btrs_accept_template(n, p, fp, u_lo, u_hi, k_lo, k_hi, fast=fast))

    code, output = run_fptaylor_query(fptaylor, accept_input, outputs_dir,
                                       env, ratio_tol, bb_eval, x_abs_tol,
                                       accept_tol_vars, approx)
    accept_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor BTRS accept (n={n} p={p}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS accept failed for "
                           f"n={n} p={p}; see {accept_output}")
    accept_raw = extract_abs_errors_by_problem(output)["eps_accept"]

    logv, _ = eps_logv(
        fptaylor, fp, v_trunc, inputs_dir, outputs_dir, env, verbose, ratio_tol,
        bb_eval, x_abs_tol, accept_tol_vars, approx=approx,
    )
    eps_accept = accept_raw + logv
    if fast:
        # the -2*log(us) query only sees us, so the smallest reachable us is
        # the right lower bound to pass
        logus, _ = eps_logus(
            fptaylor, fp, us_min, inputs_dir, outputs_dir, env, verbose, ratio_tol,
            bb_eval, x_abs_tol, accept_tol_vars, approx=approx,
        )
        eps_accept += logus

    tv = (2.0 * eps_floor * accept_iter
          + acceptance_tv(eps_accept)
          + u_excess + v_trunc)
    return eps_floor, eps_accept, tv


# ---------------------------------------------------------------------------
# Inversion FPTaylor template  (n*p < _BTRS_SWITCH)
# ---------------------------------------------------------------------------

def inversion_params(n, p):
    """(qn, z_lo, x_hi): the interval bounds the inversion template uses."""
    q = 1.0 - p
    qn_raw = math.exp(n * math.log(q))
    qn = max(qn_raw, sys.float_info.min)
    z_lo = max(min(qn_raw, math.exp(-22) / math.sqrt(2 * math.pi * n * p * q)),
               sys.float_info.min)
    x_hi = min(float(n), n * p + 10.0 * math.sqrt(n * p * q))
    return qn, z_lo, x_hi


def _make_inversion_template(n, p, fp):
    """One query per FP op in legacy_random_binomial_inversion's loop
    (distributions/binomial_legacy_inversion.c): eps0=qn, eps1=px, eps2=sum+prod."""
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


def _compute_inversion_tv(n, p, eps0, eps1, eps2):
    """tv from the inversion loop's per-op relative errors: eps1 is charged
    per step on p, eps2 on up to `bound` summation steps."""
    bound = n * p + 10.0 * math.sqrt(n * p * (1.0 - p))
    return 0.5 * (eps0 + eps1 * p + eps2 * bound)


def _run_inversion_fptaylor(fptaylor, n, p, args, tag, inputs_dir, outputs_dir, env):
    """(eps0, eps1, eps2, tv) for the inversion regime at (n, p)."""
    fp, verbose = args.fp, args.verbose
    qn, z_lo, x_hi = inversion_params(n, p)
    vprint(verbose, f"binomial inversion n={n} p={p}", qn=qn, z_lo=z_lo, x_hi=x_hi)

    inv_input  = inputs_dir  / f"binomial_inversion_{fp}_{tag}.txt"
    inv_output = outputs_dir / f"binomial_inversion_{fp}_{tag}.out"
    inv_input.write_text(_make_inversion_template(n, p, fp))

    code, output = run_command(
        [fptaylor, "--rel-error", "true", str(inv_input)], cwd=ROOT, env=env)
    inv_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor inversion (n={n} p={p}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor inversion failed for n={n} p={p}; see {inv_output}")

    deltas = extract_deltas_by_problem(output, f"n={n} p={p}")
    eps0, eps1, eps2 = deltas["eps0"], deltas["eps1"], deltas["eps2"]
    return eps0, eps1, eps2, _compute_inversion_tv(n, p, eps0, eps1, eps2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sampler_p(p):
    """The p the sampler actually runs on: btrs.c / the inversion loop both
    draw with min(p, 1-p) and reflect the result."""
    return 1.0 - p if p > 0.5 else p


def _use_btrs(n, p):
    """BTRS above the n*p switch, inversion below; p is sampler_p(p)."""
    return n * p >= _BTRS_SWITCH


def _fmt_signed(v):
    return f"{v:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def safe_pair_name(n, p):
    return f"n{n}_p{_fmt_signed(p)}"


def _validate(n, p, loc=""):
    prefix = f"{loc}: " if loc else ""
    if n <= 0:
        raise ValueError(f"{prefix}n must be positive")
    if not (0 < p < 1):
        raise ValueError(f"{prefix}p must be in (0, 1)")


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
        _validate(n, p, f"{path}:{lineno}")
        pairs.append((n, p))
    return pairs


def _empty_row(n, p, fp):
    return {"n": n, "p": f"{p:.17g}", "fp": fp, "regime": "",
            "eps0": "", "eps1": "", "eps2": "",
            "eps_floor": "", "eps_accept": "", "tv": "", "time_s": ""}


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with (n, p) pairs, one per line (format: 'n p')")
    source.add_argument("--n", type=int, default=None,
                        help="Single n value (requires --p)")
    parser.add_argument("--p", type=float, default=None,
                        help="Probability p in (0,1), required with --n")
    parser.add_argument("--fast", action="store_true",
                        help="BTRS only: compute the -2*log(us) term of "
                             "eps_accept in a separate FPTaylor query and "
                             "sum it in, decoupling it from the shared "
                             "variable u. Faster, but may yield a more "
                             "conservative (looser) bound.")


def default_out_dir(args):
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "binomial_runs"
    return ROOT / f"binomial_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    if args.n is not None:
        if args.p is None:
            raise ValueError("--p is required when --n is given")
        _validate(args.n, args.p)
        pairs = [(args.n, args.p)]
    else:
        pairs = read_np_pairs(args.input_file)
    if not pairs:
        raise ValueError("no (n, p) pairs found in input")

    rows = []
    for n, p in pairs:
        start = time.perf_counter()
        analysis_p = sampler_p(p)
        tag = safe_pair_name(n, p)
        label = f"n={n} p={p}"
        if analysis_p != p:
            label += f" (sampler uses p={analysis_p:.17g})"
        try:
            row = _empty_row(n, p, args.fp)

            # ---- low range (inversion) ----
            if not _use_btrs(n, analysis_p):
                eps0, eps1, eps2, tv = _run_inversion_fptaylor(
                    fptaylor, n, analysis_p, args, tag, inputs_dir, outputs_dir, env)

                row.update({
                    "regime": "inversion",
                    "eps0": f"{eps0:.17e}",
                    "eps1": f"{eps1:.17e}",
                    "eps2": f"{eps2:.17e}",
                    "tv": f"{tv:.17e}",
                    "time_s": f"{elapsed_since(start):.6f}",
                })
                rows.append(row)
                print(f"{label} [inversion] eps0={eps0:.6e} eps1={eps1:.6e}"
                      f" eps2={eps2:.6e} TV={tv:.6e}"
                      f" time={format_seconds(float(row['time_s']))}")
                continue

            # ---- high range (BTRS) ----
            eps_floor, eps_accept, tv = _run_btrs_fptaylor(
                fptaylor, n, analysis_p, args, tag, inputs_dir, outputs_dir, env)

            row.update({
                "regime": "btrs",
                "eps_floor": f"{eps_floor:.17e}",
                "eps_accept": f"{eps_accept:.17e}",
                "tv": f"{tv:.17e}",
                "time_s": f"{elapsed_since(start):.6f}",
            })
            rows.append(row)
            print(f"{label} [BTRS] eps_floor={eps_floor:.6e}"
                  f" eps_accept={eps_accept:.6e} TV={tv:.6e}"
                  f" time={format_seconds(float(row['time_s']))}")
        except Exception as exc:
            print(f"WARNING: skipping n={n} p={p}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    import contextlib
    import os
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
            if not r[key]:           # field not produced by this row's regime
                continue
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

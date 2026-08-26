"""
Poisson sampler FP-error analysis.
High range (lambda >= SWITCH) follows the PTRS algorithm in
distributions/random_poisson_ptrs.c and mirrors the BTRS analysis in
dist_binomial.py (eps_floor / eps_accept split, shared -log(v) and
-2*log(us) helpers, --fast flag).
"""
import math
import shutil
import tempfile
import time
from pathlib import Path

from analyticError import FP_BETA, SWITCH, computeDeltaHighRange, computeDeltaLowRange
from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_abs_errors_by_problem,
    save_loglog_plot,
    loggam_defs, eps_logv, eps_logus, fptaylor_cmd,
    hormann_u_at, point_ivar,
    hormann_proposal_deviation, acceptance_tv,
    elapsed_since, format_seconds,
    floor_x_abs_tol_vars, accept_x_abs_tol_vars,
)

NAME = "poisson"
CSV_FIELDS = ["lambda", "fp", "regime", "eps_floor", "eps_accept", "tv",
              "ref_tv", "time_s"]


# ---------------------------------------------------------------------------
# PTRS FPTaylor templates  (lambda >= SWITCH)
# ---------------------------------------------------------------------------

def ptrs_consts(lam):
    """(slam, a, b, c): the setup constants random_poisson_ptrs computes once."""
    slam = math.sqrt(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    return slam, a, b, lam + 0.43


def ptrs_setup_defs(rnd, lam_expr, accept=False):
    """
    random_poisson_ptrs's setup block [lines 77-82] as FPTaylor Definitions.
    These are derived from lambda, not free inputs, so writing them as
    expressions keeps them correlated and charges the rounding of the setup
    arithmetic itself.
    """
    d = [f"  slam_ {rnd}= sqrt({lam_expr}),",
         f"  b_    {rnd}= 0.931 + 2.53 * slam_,",
         f"  a_    {rnd}= -0.059 + 0.02483 * b_,",
         f"  c_    {rnd}= {lam_expr} + 0.43,"]
    if accept:
        d += [f"  ialp_ {rnd}= 1.1239 + 1.1328 / (b_ - 3.4),",
              f"  llam_ {rnd}= log({lam_expr}),"]
    return d


# k1_ = k_ + 1 = (y_ - f) + 1, f in [0, 1], and (unlike the BTRS/PTRS accept
# shell's k-range, which has no extra y-window padding of its own here) y_
# sits exactly at k_lo at the window's low corner, so k1_'s enclosure dips
# to exactly k_lo at f = 1 -- k_lo >= 1 is what keeps k1_ > 0.
_K_BOUNDARY_MARGIN = 1.0
_K_SIGMAS = 10.0

def ptrs_accept_k_range(lam):
    """k window the accept query covers: +-10 sigma bulk, kept >= _K_BOUNDARY_MARGIN."""
    slam = math.sqrt(lam)
    k_lo = max(_K_BOUNDARY_MARGIN, math.ceil(lam - _K_SIGMAS * slam))
    k_hi = math.floor(lam + _K_SIGMAS * slam)
    if k_lo >= k_hi:
        raise ValueError(f"empty accept window k in [{k_lo:.6g}, {k_hi:.6g}] "
                         f"for lambda={lam:.6g}")
    return float(k_lo), float(k_hi)


def ptrs_accept_u_range(lam):
    """The u that map into ptrs_accept_k_range."""
    _, a, b, c = ptrs_consts(lam)
    k_lo, k_hi = ptrs_accept_k_range(lam)
    return hormann_u_at(a, b, c, k_lo), hormann_u_at(a, b, c, k_hi)


def clip_ptrs_u(u_lo, u_hi, u_trunc):
    edge_lo = -0.5 + u_trunc
    edge_hi = 0.5 - u_trunc
    return max(u_lo, edge_lo), min(u_hi, edge_hi)


def poisson_cdf_below(lam, k_lo):
    """
    P(K < k_lo) for K ~ Poisson(lam), summed term by term in log space.

    Only called with k_lo = ptrs_accept_k_range's low end (see
    _K_BOUNDARY_MARGIN).  Nudged up by a few ulp to
    stay an upper bound on the exact tail despite lgamma/exp rounding: it is
    charged as probability mass, so erring high is the safe direction.
    """
    m = int(math.ceil(k_lo)) - 1          # largest integer k with k < k_lo
    if m < 0:
        return 0.0
    loglam = math.log(lam)
    total = math.fsum(
        math.exp(-lam + k * loglam - math.lgamma(k + 1.0))
        for k in range(m + 1)
    )
    return total * (1.0 + 1e-12)


def ptrs_k_tail_prob(lam):
    """
    P(K outside the accept window): the output mass that query does not cover.

    The lower tail is summed exactly -- for lambda close to SWITCH, k_lo is
    pinned near _K_BOUNDARY_MARGIN, so it is O(1) terms, and the Chernoff
    bound is badly loose that close to
    the mode (at lambda=30 it gives 1.1e-5 for a tail that is really 1.2e-6),
    which made it the dominant term of the whole TV bound.

    The upper tail stays on the Chernoff bound, using the Poisson rate
    function h(t) = t*log(t) - t + 1, i.e. P(K >= x) <= exp(-lam*h(x/lam)):
    it sits ~10 sigma out where the bound is already negligible and summing
    it would cost O(lam) terms.  Bernstein is far too crude this far out --
    at lambda=50 it gives 2e-6 for a tail that is really ~1e-12, which would
    then dominate the whole bound.
    """
    k_lo, k_hi = ptrs_accept_k_range(lam)
    if not (k_lo < lam < k_hi):
        raise ValueError(f"accept window [{k_lo:.6g}, {k_hi:.6g}] does not "
                         f"contain lambda={lam:.6g}")

    def h(t):
        return t * math.log(t) - t + 1.0 if t > 0.0 else 1.0

    return poisson_cdf_below(lam, k_lo) + math.exp(-lam * h(k_hi / lam))


def make_ptrs_floor_template(lam, fp, utail):
    """
    FPTaylor expression for eps_floor: absolute error of
    (2*a/us + b)*u + (lambda + 0.43), with us = 0.5 - |u|
    [random_poisson_ptrs.c line 89]

    Only u is free; lambda is declared as a Variable bracketing its exact
    value (dist_common.exact_bracket) so a, b, c reference it by name
    instead of re-embedding the literal at every occurrence.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real u in [{-(0.5 - utail):.20e}, {0.5 - utail:.20e}],\n"
        + point_ivar("lam", lam) + ";\n\n"
        + "Definitions\n"
        + "\n".join(ptrs_setup_defs(rnd, "lam")) + "\n"
        + f"  us_   {rnd}= 0.5 - abs(u),\n"
        + f"  ptrs_floor {rnd}= (2.0 * a_ / us_ + b_) * u + c_;\n\n"
        + "Expressions\n"
          "  eps_floor = ptrs_floor;\n"
    )


def make_ptrs_accept_template(lam, fp, u_lo, u_hi, k_lo, k_hi, fast=False):
    """
    FPTaylor expression for eps_accept (excluding -log(v), see
    dist_common.make_logv_template): absolute error of
    -lambda + k*log(lambda) - loggam(k+1) - log(invalpha) + log(a/us^2 + b),
    with us = 0.5 - |u|   [random_poisson_ptrs.c lines 98-99, rearranged so
    that log(v) is alone on the left-hand side].
    loggam(k+1) is FPTaylor's native lgamma(k1_) directly (loggam_defs,
    dist_common.py).
    log(a/us^2 + b) is rewritten as log(a + b*us^2) - 2*log(us) to avoid
    forming 1/us^2 directly when us is small (see dist_binomial.py).

    k is declared directly as a Variable over [k_lo, k_hi] (from
    ptrs_accept_k_range) rather than derived here from u via the y = f(u)
    map: k and u are only jointly reachable through the *exact* (unrounded)
    floor relationship the sampler enforces, and eps_floor already bounds
    any disagreement about which k a given u floors to. Re-deriving k from
    u inside this template would needlessly propagate u's own error through
    that derivative-~1e6 map into loggam(k+1), inflating eps_accept and
    double-counting what eps_floor already covers (see the old
    hormann_k_defs docstring in dist_common.py). Declaring k directly, and
    letting u range over its own full (both-signs) interval only for
    us_/log_num_, is a sound relaxation -- exactly the reparametrization
    used for hypergeometric's W (see dist_hypergeometric.hrua_z_defs) --
    and confirmed by direct comparison to give a *tighter* bound here too
    (single query, no domain errors, ~28% smaller eps_accept at
    lambda=40 than the old per-sign derivation).

    If fast is True, the -2*log(us_) term is omitted here and its error is
    computed separately (see dist_common.make_logus_template) and summed in
    by the caller. This drops u as a shared variable between the two terms,
    which may yield a more conservative (looser) overall bound.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    defs_k, name_k = loggam_defs("k1_", "lgk", rnd)
    log_us_term = "" if fast else " - 2.0 * log(us_)"

    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}],\n"
        f"  real k in [{k_lo:.20e}, {k_hi:.20e}],\n"
        + point_ivar("lam", lam) + ";\n\n"
        + "Definitions\n"
        + "\n".join(ptrs_setup_defs(rnd, "lam", accept=True)) + "\n"
        + f"  us_    {rnd}= 0.5 - abs(u),\n"
        + f"  k1_    = k + 1.0,\n"
        + "\n".join(defs_k) + "\n"
        + f"  us_sq_      {rnd}= us_ * us_,\n"
        + f"  log_num_    {rnd}= a_ + b_ * us_sq_,\n"
        + f"  ptrs_accept {rnd}= -lam + k * llam_ - {name_k}"
          f" - log(ialp_) + log(log_num_){log_us_term};\n\n"
        + "Expressions\n"
          "  eps_accept = ptrs_accept;\n"
    )


def _run_fptaylor_query(fptaylor, input_path, outputs_dir, env, ratio_tol,
                        bb_eval=False, x_abs_tol=None, x_abs_tol_vars=None,
                        approx=True):
    """Run one FPTaylor query and return output.

    PTRS queries run one per call site, not concurrently, so the compile race
    documented in dist_common.fptaylor_cmd does not apply here even with the
    default --opt bb backend.
    """
    work = Path(tempfile.mkdtemp(prefix="fpt_", dir=outputs_dir))
    try:
        return run_command(
            fptaylor_cmd(fptaylor, input_path, work, ratio_tol, bb_eval,
                        x_abs_tol, x_abs_tol_vars, approx),
            cwd=ROOT, env=env)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_ptrs_fptaylor(fptaylor, lam, fp, tag, inputs_dir, outputs_dir, env, verbose,
                       fast=False, v_trunc=2.0 ** -53, ratio_tol=2.0,
                       bb_eval=False, x_abs_tol=None, approx=True,
                       u_trunc=None, floor_tol_vars=None,
                       accept_tol_vars=None):
    """Run PTRS queries and compose the theorem 2.7 bound."""
    if u_trunc is None or not (0.0 < u_trunc < 0.5):
        raise ValueError("PTRS requires --u-trunc with 0 < u_trunc < 0.5")
    _, a, b, _ = ptrs_consts(lam)
    invalpha = 1.1239 + 1.1328 / (b - 3.4)

    floor_input  = inputs_dir  / f"poisson_ptrs_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"poisson_ptrs_floor_{fp}_{tag}.out"
    floor_input.write_text(make_ptrs_floor_template(lam, fp, u_trunc))

    code, output = _run_fptaylor_query(fptaylor, floor_input, outputs_dir, env,
                                       ratio_tol, bb_eval, x_abs_tol, floor_tol_vars, approx)
    floor_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor PTRS floor (lambda={lam}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor PTRS floor failed for lambda={lam}; see {floor_output}")

    floor_raw = extract_abs_errors_by_problem(output)["eps_floor"]
    eps_floor = hormann_proposal_deviation(floor_raw, a, b)

    # k is now declared directly over its own interval (ptrs_accept_k_range),
    # decoupled from u -- see make_ptrs_accept_template's docstring -- so u
    # no longer needs a sign-specific derivation and both sides run as one
    # query over u's full (both-signs) range.
    au_lo, au_hi = ptrs_accept_u_range(lam)
    k_lo, k_hi = ptrs_accept_k_range(lam)
    k_tail = ptrs_k_tail_prob(lam)
    lo, hi = clip_ptrs_u(au_lo, au_hi, u_trunc)
    if lo > hi:
        raise ValueError(f"lambda={lam}: u-range emptied by u_trunc={u_trunc}")

    accept_input  = inputs_dir  / f"poisson_ptrs_accept_{fp}_{tag}.txt"
    accept_output = outputs_dir / f"poisson_ptrs_accept_{fp}_{tag}.out"
    accept_input.write_text(
        make_ptrs_accept_template(lam, fp, lo, hi, k_lo, k_hi, fast=fast))

    code, output = _run_fptaylor_query(fptaylor, accept_input, outputs_dir,
                                       env, ratio_tol, bb_eval, x_abs_tol,
                                       accept_tol_vars, approx)
    accept_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor PTRS accept (lambda={lam}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor PTRS accept failed for "
                           f"lambda={lam}; see {accept_output}")
    accept_raw = extract_abs_errors_by_problem(output)["eps_accept"]

    logv, _ = eps_logv(
        fptaylor, fp, v_trunc, inputs_dir, outputs_dir, env, verbose, ratio_tol,
        bb_eval, x_abs_tol, accept_tol_vars, approx=approx,
    )
    eps_accept = accept_raw + logv
    if fast:
        logus, _ = eps_logus(
            fptaylor, fp, u_trunc, inputs_dir, outputs_dir, env, verbose, ratio_tol,
            bb_eval, x_abs_tol, accept_tol_vars, approx=approx,
        )
        eps_accept += logus

    tv = (2.0 * u_trunc + 2.0 * k_tail + v_trunc
          + 2.0 * invalpha * eps_floor
          + acceptance_tv(eps_accept))
    return eps_floor, eps_accept, tv


# ---------------------------------------------------------------------------
# Low-range FPTaylor templates  (lambda < SWITCH)
# ---------------------------------------------------------------------------

def _make_low_range_template(lam_str, fp):
    lam = float(lam_str)
    k_star = int(lam + 10 * math.sqrt(lam))
    rnd = FP_TO_FPTAYLOR_RND[fp]
    var_lines = [f"  real u_{i} in [0, 1]" for i in range(1, k_star + 1)]
    def_lines = (
        [f"  lambda = {lam_str}", f"  L = {rnd}(exp(-lambda))", f"  p_1 = {rnd}(u_1)"]
        + [f"  p_{i} = {rnd}(p_{i-1} * u_{i})" for i in range(2, k_star + 1)]
    )
    return (
        "Variables\n" + ",\n".join(var_lines) + ";\n\n"
        + "Definitions\n" + ",\n".join(def_lines) + ";\n\n"
        + "Expressions\n"
        + f"  L_compute = L;\n"
        + f"  prod_compute = p_{k_star};\n"
    )


def _make_log_low_range_template(lam_str, fp):
    lam = float(lam_str)
    k_star = int(lam + 10 * math.sqrt(lam))
    rnd = FP_TO_FPTAYLOR_RND[fp]
    var_lines = [f"  real u_{i} in [1e-300, 1]" for i in range(1, k_star + 1)]
    def_lines = (
        [f"  lambda = {lam_str}",
         f"  lambda_fp {rnd}= {lam_str}",
         f"  logp_1 = {rnd}(log(u_1))"]
        + [f"  logp_{i} = {rnd}(logp_{i-1} + {rnd}(log(u_{i})))"
           for i in range(2, k_star + 1)]
    )
    return (
        "Variables\n" + ",\n".join(var_lines) + ";\n\n"
        + "Definitions\n" + ",\n".join(def_lines) + ";\n\n"
        + "Expressions\n"
        + f"  log_prod_compute = logp_{k_star};\n"
        + f"  lambda_fp_compute = lambda_fp;\n"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_lambda_name(lam):
    return lam.replace("+", "").replace("-", "m").replace(".", "p").replace("E", "e")


def read_lambdas(path):
    lambdas = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        for token in line.split():
            try:
                lam = float(token)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid lambda {token!r}") from exc
            if lam <= 0:
                raise ValueError(f"{path}:{lineno}: lambda must be positive")
            lambdas.append(token)
    return lambdas


def _compute_low_range_delta(lam, l_compute_error, prod_compute_error):
    l_value = math.exp(-lam)
    E = prod_compute_error + l_compute_error
    delta = math.inf if E >= l_value else 2 * E / (l_value - E)
    return l_value, E, delta


def _compute_log_low_range_delta(lambda_fp_error, log_prod_error):
    E = lambda_fp_error + log_prod_error
    return E, 2 * E


def _empty_row(lam, fp):
    return {"lambda": lam, "fp": fp, "regime": "",
            "eps_floor": "", "eps_accept": "", "tv": "", "ref_tv": "",
            "time_s": ""}


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("lambda_file", nargs="?", type=Path,
                        help="File with lambda values, one or more per line")
    source.add_argument("--lam", type=float, default=None,
                        help="Single lambda value")
    parser.add_argument("--use-log", action="store_true",
                        help="Use log-space template for low-range lambdas")
    parser.add_argument("--fast", action="store_true",
                        help="PTRS only: compute the -2*log(us) term of "
                             "eps_accept in a separate FPTaylor query and "
                             "sum it in, decoupling it from the shared "
                             "variable u. Faster, but may yield a more "
                             "conservative (looser) bound.")


def default_out_dir(args):
    lf = getattr(args, "lambda_file", None)
    if lf is None:
        return ROOT / "poisson_runs"
    return ROOT / f"poisson_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    lambdas = [str(args.lam)] if args.lam is not None else read_lambdas(args.lambda_file)

    rows = []
    for lam in lambdas:
        start = time.perf_counter()
        lam_float = float(lam)
        tag = safe_lambda_name(lam)
        try:
            # ---- low range ----
            if lam_float < SWITCH:
                lr_input = inputs_dir / f"low_range_{args.fp}_lam_{tag}.txt"
                if args.use_log:
                    lr_input.write_text(_make_log_low_range_template(lam, args.fp))
                else:
                    lr_input.write_text(_make_low_range_template(lam, args.fp))

                code, output = run_command([fptaylor, str(lr_input)], cwd=ROOT, env=env)
                out_path = outputs_dir / f"low_range_{args.fp}_lam_{tag}.out"
                out_path.write_text(output)
                if args.verbose >= 2:
                    print(f"--- FPTaylor low range (lambda={lam}) ---\n{output}")
                if code != 0:
                    raise RuntimeError(f"FPTaylor low range failed for lambda={lam}; see {out_path}")

                errs = extract_abs_errors_by_problem(output)
                if args.use_log:
                    missing = {"log_prod_compute", "lambda_fp_compute"} - errs.keys()
                    if missing:
                        raise RuntimeError(f"could not parse low-range errors for {', '.join(sorted(missing))}")
                    _, low_tv = _compute_log_low_range_delta(
                        errs["lambda_fp_compute"], errs["log_prod_compute"]
                    )
                else:
                    missing = {"L_compute", "prod_compute"} - errs.keys()
                    if missing:
                        raise RuntimeError(f"could not parse low-range errors for {', '.join(sorted(missing))}")
                    _, _, low_tv = _compute_low_range_delta(
                        lam_float, errs["L_compute"], errs["prod_compute"]
                    )

                ref_tv = computeDeltaLowRange(lam_float, FP_BETA[args.fp])
                row = _empty_row(lam, args.fp)
                row.update({"regime": "low", "tv": f"{low_tv:.17e}", "ref_tv": f"{ref_tv:.17e}"})
                row["time_s"] = f"{elapsed_since(start):.6f}"
                rows.append(row)
                print(f"lambda={lam} [low] TV={row['tv']} ref_TV={row['ref_tv']}"
                      f" time={format_seconds(float(row['time_s']))}")
                continue

            # ---- high range (PTRS) ----
            eps_floor, eps_accept, tv = _run_ptrs_fptaylor(
                fptaylor, lam_float, args.fp, tag, inputs_dir, outputs_dir,
                env, args.verbose, fast=args.fast, v_trunc=args.v_trunc,
                ratio_tol=args.bb_geometric_ratio_tol,
                bb_eval=args.bb_eval, x_abs_tol=args.opt_x_abs_tol,
                approx=args.approx,
                u_trunc=args.u_trunc,
                floor_tol_vars=floor_x_abs_tol_vars(args),
                accept_tol_vars=accept_x_abs_tol_vars(args),
            )
            ref_tv = computeDeltaHighRange(lam_float, FP_BETA[args.fp])[0]

            row = _empty_row(lam, args.fp)
            row.update({
                "regime": "ptrs",
                "eps_floor": f"{eps_floor:.17e}",
                "eps_accept": f"{eps_accept:.17e}",
                "tv": f"{tv:.17e}",
                "ref_tv": f"{ref_tv:.17e}",
                "time_s": f"{elapsed_since(start):.6f}",
            })
            rows.append(row)
            print(f"lambda={lam} [PTRS] eps_floor={eps_floor:.6e}"
                  f" eps_accept={eps_accept:.6e} TV={tv:.6e} ref_TV={ref_tv:.6e}"
                  f" time={format_seconds(float(row['time_s']))}")
        except Exception as exc:
            print(f"WARNING: skipping lambda={lam}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    points = []
    for row in rows:
        points.append((float(row["lambda"]), float(row["tv"]), float(row["ref_tv"])))
    points.sort(key=lambda r: r[0])
    xs = [r[0] for r in points]
    series = [
        ("TV (computed)", [r[1] for r in points], "^"),
        ("TV (analyticError)", [r[2] for r in points], "x"),
    ]
    save_loglog_plot(xs, series, xlabel="lambda", ylabel="error",
                     plot_path=plot_path, plot_pgf=plot_pgf, ylim_top=0.9)

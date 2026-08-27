"""
Poisson PTRS FP-error analysis against a numerically stable reformulation of
the sampler (distributions/random_poisson_ptrs_stable.c).

Same algorithm as dist_poisson.py -- same hat, same acceptance region, same
RNG stream -- evaluated so that FPTaylor's bound stops being dominated by
cancellation and by rounding nodes that do not actually exist.  Three changes,
each measured independently on this repo's FPTaylor build:

1. The sampled inputs are exact.  rk_double() returns a multiple of 2^-53, so
   U = rk_double() - 0.5 and us = 0.5 - |U| are multiples of 2^-53 in [-1/2,
   1/2] and [0, 1/2] -- exactly representable, as is V.  dist_poisson.py
   declares them `real`, so FPTaylor inserts a rounding node on the input and
   amplifies it by dt/du ~ a/us^2 (coefficient 1.63e7 at lambda=1e6).
   Declaring them float64 is sound, not an approximation.
       eps_logv  8.674e-9 -> 1.776e-15;  eps_floor 1.92e-9 -> 1.82e-10 at 1e6.
   With v exact the bound stays tight arbitrarily close to 0 (5.68e-14 at
   v in [1e-300, 1]), but FPTaylor's compiled --opt bb backend fails to
   compile the generated OCaml for a literal that extreme (the exact
   rational it emits for a double that close to the subnormal range blows
   up), so v_trunc is left at the same CLI/toml-driven value as
   dist_poisson.py rather than pushed to the edge of what the bound itself
   would tolerate.
   This does *not* help eps_accept (7.2220e-9 vs 7.2233e-9): that error is
   genuine cancellation, addressed by (2).

2. Loader's saddle-point form for the log-pmf.  -lam + k*log(lam) -
   loggam(k+1) sums terms of size lam*log(lam) (3e14 at lambda=1e13) to
   produce a result of size ~10.  Instead

       log p(k) = -bd0(k, lam) - stirlerr(k) - 0.5*log(2*pi*k)

   with bd0 evaluated through the identity
       k*log(k/lam) - k + lam = (k-lam)^2/(k+lam) + 2k*(atanh(v) - v),
       v = (k-lam)/(k+lam),  atanh(v) - v = v^3*(1/3 + v^2/5 + v^4/7 + ...)
   so every intermediate is O(50) instead of O(lam*log lam).  The series is
   used only where |v| <= 0.1; outside that k*log(k/lam) - k + lam is already
   well conditioned.  The k-range is partitioned and the max taken, which is
   sound for the same reason as logbb.log_bb: max over a partition of per-box
   bounds bounds the union.
       eps_accept 7.22e-9 -> 6.91e-13 (1e6);  2.32e-1 -> 3.09e-9 (1e13).

3. lambda never enters the rounded floor expression.  With lam = m + f,
   m = floor(lam) and f = lam - m exact by Sterbenz,
       k = m + floor((2a/us + b)*U + f + 0.43)
   and the outer sum is exact (both integers, lam < 2^53).  Writing the
   proposal in us rather than U,
       (2a/us + b)*(0.5 - us) = a/us - 2a + 0.5b - b*us,
   drops the argument magnitude from ~lam to ~a/us, i.e. eps_floor from
   O(lam*2^-53) to O(sqrt(lam)*2^-53).
       eps_floor 1.17e-7 -> 4.69e-10 (1e9);  9.88e-4 -> 5.98e-8 (1e13).

Measured tv against dist_poisson.py: 1.8e-8 -> 6.1e-13 at lambda=50, and
0.467 -> 1.4e-7 at lambda=1e13, where the old bound is vacuous.

Everything outside the PTRS expressions -- the low-range path, eps_floor's
Hormann-deviation wrapper, the u_trunc-derived k-range and TV formula, and
the FPTaylor invocation flags (--bb-eval, --approx, --opt-x-abs-tol) -- is
carried over from dist_poisson.py unchanged, so the two modules are directly
comparable.
"""
import math
import time
from fractions import Fraction
from pathlib import Path

from analyticError import FP_BETA, SWITCH, computeDeltaHighRange, computeDeltaLowRange
from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND, run_command, extract_abs_errors_by_problem,
    save_loglog_plot, run_fptaylor_query, ulp_rnd_op,
    hormann_proposal_deviation, acceptance_tv,
    elapsed_since, format_seconds,
    floor_x_abs_tol_vars, accept_x_abs_tol_vars,
)
import dist_poisson as base

NAME = "poisson-stable"
CSV_FIELDS = base.CSV_FIELDS

NSERIES = 10            # atanh-series terms; truncation is checked per box
VSPLIT = 0.1            # |v| below which the series is used (Loader's cutoff)
STIRLERR_KMIN = 16      # series valid above this; exact table below
EPS = 2.0 ** -53
LS2PI = 0.5 * math.log(2.0 * math.pi)

# stirlerr(n) = log(n!) - (0.5*log(2*pi*n) + n*log(n) - n), computed at 60
# decimal digits and rounded once.  Index 0 is unused (log p(0) = -lam is
# exact and handled separately).
STIRLERR_TAB = [
    0.0,
    8.10614667953272611e-02, 4.13406959554092970e-02, 2.76779256849983384e-02,
    2.07906721037650934e-02, 1.66446911898211931e-02, 1.38761288230707484e-02,
    1.18967099458917695e-02, 1.04112652619720962e-02, 9.25546218271273285e-03,
    8.33056343336287079e-03, 7.57367548795184059e-03, 6.94284010720952992e-03,
    6.40899418800420714e-03, 5.95137011275884750e-03, 5.55473355196280105e-03,
]
# |series - exact| at n = STIRLERR_KMIN, the worst point of the series branch.
STIRLERR_SERIES_TRUNC = 2.0e-18
# stirlerr(n) = S0/n - S1/n^3 + S2/n^5 - S3/n^7 + S4/n^9 - S5/n^11
SERR = [1 / 12.0, -1 / 360.0, 1 / 1260.0, -1 / 1680.0, 1 / 1188.0, -691 / 360360.0]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def ptrs_params(lam):
    """a, b, invalpha exactly as random_poisson_ptrs_stable.c computes them."""
    b = 0.931 + 2.53 * math.sqrt(lam)
    a = -0.059 + 0.02483 * b
    return a, b, 1.1239 + 1.1328 / (b - 3.4)


def _const_err(computed, exact):
    """|fl(expr) - expr| for a constant the C code evaluates once, outside the
    loop.  FPTaylor sees only the rounded literal, so this offset has to be
    added by hand; `exact` is a Fraction over the double inputs."""
    return float(abs(Fraction(computed) - exact))


def floor_constants(lam):
    """(m, A2, c0, const_err) for  t = s*(a/us + A2 - b*us) + c0,  k = m + floor(t)."""
    a, b, _ = ptrs_params(lam)
    m = math.floor(lam)
    frac = lam - m                       # exact (Sterbenz, lam >= 1)
    A2 = 0.5 * b - 2.0 * a
    c0 = frac + 0.43
    err = (_const_err(A2, Fraction(b) / 2 - 2 * Fraction(a))
           + _const_err(c0, Fraction(frac) + Fraction(0.43)))
    return m, A2, c0, err


# ---------------------------------------------------------------------------
# FPTaylor templates
# ---------------------------------------------------------------------------

def _horner(prefix, coeffs, xvar, rnd):
    """h0 = (..((c_n*x + c_{n-1})*x + ..)*x + c_0.  Returns (lines, name)."""
    n = len(coeffs) - 1
    lines = [f"  {prefix}{n} = {coeffs[n]:.20e},"]
    for i in range(n - 1, -1, -1):
        lines.append(f"  {prefix}{i} {rnd}= {prefix}{i+1} * {xvar} + {coeffs[i]:.20e},")
    return lines, f"{prefix}0"


def make_floor_template(lam, fp, utail, sign):
    """Absolute error of  t = s*(a/us + A2 - b*us) + c0,  us in [utail, 1/2].

    us is the primitive draw and is exactly representable, hence float64.
    """
    a, b, _ = ptrs_params(lam)
    _, A2, c0, _ = floor_constants(lam)
    rnd = FP_TO_FPTAYLOR_RND[fp]
    core = "(a_ / us + A2_) - b_ * us"
    body = f"({core}) + c0_" if sign > 0 else f"c0_ - ({core})"
    return (
        "Variables\n"
        f"  float64 us in [{utail:.20e}, 5.0e-1];\n\n"
        "Definitions\n"
        f"  a_  = {a:.20e},\n"
        f"  b_  = {b:.20e},\n"
        f"  A2_ = {A2:.20e},\n"
        f"  c0_ = {c0:.20e},\n"
        f"  t_  {rnd}= {body};\n\n"
        "Expressions\n"
        "  eps_floor = t_;\n"
    )


def _accept_tail(rnd):
    """The us-dependent part, shared by every accept template:
    log(a + b*us^2) - 2*log(us), with us exact.  The two log() calls get
    the ULP ledger's log scale (ulp_rnd_op) rather than the plain rnd used
    for the surrounding arithmetic -- see dist_common.ulp_rnd_op."""
    log_rnd = ulp_rnd_op(rnd, "log")
    return [f"  us2_     {rnd}= us * us,",
            f"  num_     {rnd}= a_ + b_ * us2_,",
            f"  log_num_ {log_rnd}= log(num_),",
            f"  log_us_  {log_rnd}= log(us),"]


def make_accept_template(lam, fp, utail, k_lo, k_hi, mode):
    """Absolute error of

        C - bd0(k, lam) - stirlerr(k) - 0.5*log(k) + log(a + b*us^2) - 2*log(us)

    with C = -log(invalpha) - 0.5*log(2*pi) folded into one constant.
    mode 'series' uses the atanh expansion of bd0, 'direct' uses
    k*log(k/lam) - k + lam.  Requires k_lo >= STIRLERR_KMIN.
    """
    a, b, invalpha = ptrs_params(lam)
    rnd = FP_TO_FPTAYLOR_RND[fp]
    log_rnd = ulp_rnd_op(rnd, "log")
    C = -math.log(invalpha) - LS2PI

    d = [f"  a_    = {a:.20e},",
         f"  b_    = {b:.20e},",
         f"  lam_  = {lam:.20e},",
         f"  C_    = {C:.20e},"]

    if mode == "series":
        P = [1.0 / (2 * j + 1) for j in range(1, NSERIES + 1)]
        hl, hn = _horner("bp_", P, "v2_", rnd)
        d += [f"  dk_   {rnd}= k - lam_,",
              f"  sk_   {rnd}= k + lam_,",
              f"  v_    {rnd}= dk_ / sk_,",
              f"  v2_   {rnd}= v_ * v_,"] + hl + [
              f"  bd0_  {rnd}= dk_ * v_ + ((2.0 * k) * (v_ * v2_)) * {hn},"]
    else:
        d += [f"  log_k_lam_ {log_rnd}= log(k / lam_),",
              f"  bd0_       {rnd}= (k * log_k_lam_ - k) + lam_,"]

    sl, sn = _horner("se_", SERR, "ki2_", rnd)
    d += [f"  ki_   {rnd}= 1.0 / k,",
          f"  ki2_  {rnd}= ki_ * ki_,"] + sl + [f"  st_   {rnd}= {sn} * ki_,"]
    d += [f"  log_k_ {log_rnd}= log(k),"]
    d += _accept_tail(rnd)
    d += [f"  acc_  {rnd}= (((C_ - bd0_) - st_) - 0.5 * log_k_)"
          f" + log_num_ - 2.0 * log_us_;"]

    return ("Variables\n"
            f"  float64 us in [{utail:.20e}, 5.0e-1],\n"
            f"  real k in [{k_lo:.1f}, {k_hi:.1f}];\n\n"
            "Definitions\n" + "\n".join(d) + "\n\n"
            "Expressions\n  eps_accept = acc_;\n")


def make_accept_point_template(lam, fp, utail, k):
    """Accept expression for one integer k < STIRLERR_KMIN, where stirlerr
    comes from the exact table rather than the series.  k is a literal, so us
    is the only variable.  k = 0 uses log p(0) = -lam exactly."""
    a, b, invalpha = ptrs_params(lam)
    rnd = FP_TO_FPTAYLOR_RND[fp]
    log_rnd = ulp_rnd_op(rnd, "log")
    d = [f"  a_    = {a:.20e},",
         f"  b_    = {b:.20e},"]

    if k == 0:
        d += [f"  C_    = {-math.log(invalpha):.20e},",
              f"  lam_  = {lam:.20e},",
              f"  lp_   {rnd}= C_ - lam_,"]
    else:
        # -bd0(k,lam) - stirlerr(k) - 0.5*log(k) - LS2PI - log(invalpha)
        C = -math.log(invalpha) - LS2PI - STIRLERR_TAB[k]
        d += [f"  C_         = {C:.20e},",
              f"  lam_       = {lam:.20e},",
              f"  kk_        = {float(k):.20e},",
              f"  log_kk_lam_ {log_rnd}= log(kk_ / lam_),",
              f"  bd0_        {rnd}= (kk_ * log_kk_lam_ - kk_) + lam_,",
              f"  log_kk_    {log_rnd}= log(kk_),",
              f"  lp_        {rnd}= (C_ - bd0_) - 0.5 * log_kk_,"]

    d += _accept_tail(rnd)
    d += [f"  acc_  {rnd}= lp_ + log_num_ - 2.0 * log_us_;"]

    return ("Variables\n"
            f"  float64 us in [{utail:.20e}, 5.0e-1];\n\n"
            "Definitions\n" + "\n".join(d) + "\n\n"
            "Expressions\n  eps_accept = acc_;\n")


def make_logv_template(fp, vtail):
    """-log(v) with v exact.  Compare dist_common.make_logv_template, which
    declares v `real`: that costs 8.67e-9 instead of 5.68e-14."""
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return ("Variables\n"
            f"  float64 v in [{vtail:.20e}, 1.0];\n\n"
            "Definitions\n"
            f"  logv_step {ulp_rnd_op(rnd, 'log')}= - log(v);\n\n"
            "Expressions\n  eps_logv = logv_step;\n")


# ---------------------------------------------------------------------------
# k partition and non-FPTaylor (method) error
# ---------------------------------------------------------------------------

def accept_partition(lam, u_trunc):
    """[(k_lo, k_hi, mode)] over k >= STIRLERR_KMIN, series only where
    |v| <= VSPLIT.  Integer k below STIRLERR_KMIN are handled point-wise.

    The k window itself (dist_poisson.ptrs_accept_k_range) is the same real
    quantity here as in dist_poisson.py: y = (2*a/us + b)*u + c is the same
    number whether evaluated in that form or in this module's
    cancellation-avoiding us-only form, so k_lo/k_hi carry over unchanged.
    """
    lo, hi = base.ptrs_accept_k_range(lam, u_trunc)
    lo = max(lo, STIRLERR_KMIN)
    if lo >= hi:
        return []
    r = (1.0 + VSPLIT) / (1.0 - VSPLIT)          # |v| <= VSPLIT  <=>  k/lam in [1/r, r]
    n_lo, n_hi = lam / r, lam * r
    parts = []
    if lo < min(hi, n_lo):
        parts.append((float(lo), float(min(hi, n_lo)), "direct"))
    if max(lo, n_lo) < min(hi, n_hi):
        parts.append((float(max(lo, n_lo)), float(min(hi, n_hi)), "series"))
    if max(lo, n_hi) < hi:
        parts.append((float(max(lo, n_hi)), float(hi), "direct"))
    return parts


def method_error(lam, k_lo, k_hi, mode):
    """Bound on |value the template encodes in exact arithmetic - log p(k)|.

    FPTaylor bounds rounding of the written expression; it cannot see that the
    expression is itself an approximation.  Two sources:
      * the atanh tail beyond NSERIES terms;
      * the series coefficients are stored as doubles, so the encoded real
        expression uses fl(1/(2j+1)) and fl(S_j).  Both series have the sign
        pattern needed for a term-wise relative bound of EPS.
    """
    err = STIRLERR_SERIES_TRUNC + EPS * abs(SERR[0]) / k_lo
    if mode == "series":
        v = max(abs((k - lam) / (k + lam)) for k in (k_lo, k_hi))
        J = NSERIES + 1
        tail = 2 * k_hi * v ** (2 * J + 1) / ((2 * J + 1) * (1 - v * v))
        head = 2 * k_hi * v ** 3 / 3.0 / (1 - v * v)
        err += tail + EPS * head
    return err


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _fpt(fptaylor, name, text, problem, inputs_dir, outputs_dir, env, verbose,
        label, ratio_tol, bb_eval, x_abs_tol, x_abs_tol_vars, approx):
    in_path = inputs_dir / f"{name}.txt"
    out_path = outputs_dir / f"{name}.out"
    in_path.write_text(text)
    code, output = run_fptaylor_query(fptaylor, in_path, outputs_dir, env,
                                      ratio_tol, bb_eval, x_abs_tol,
                                      x_abs_tol_vars, approx)
    out_path.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor {label} ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor {label} failed; see {out_path}")
    return extract_abs_errors_by_problem(output)[problem]


_LOGV_CACHE = {}


def eps_logv(fptaylor, fp, vtail, inputs_dir, outputs_dir, env, verbose,
            ratio_tol, bb_eval, x_abs_tol, x_abs_tol_vars, approx):
    key = (fp, vtail, ratio_tol, bb_eval, x_abs_tol, x_abs_tol_vars, approx)
    if key not in _LOGV_CACHE:
        _LOGV_CACHE[key] = _fpt(
            fptaylor, f"stable_logv_{fp}", make_logv_template(fp, vtail),
            "eps_logv", inputs_dir, outputs_dir, env, verbose, "log(v)",
            ratio_tol, bb_eval, x_abs_tol, x_abs_tol_vars, approx)
    return _LOGV_CACHE[key]


def run_ptrs_stable(fptaylor, lam, args, tag, inputs_dir, outputs_dir, env):
    """(eps_floor, eps_accept, tv) for the stable PTRS formulation; composes
    the same bound as dist_poisson._run_ptrs_fptaylor (Hormann deviation on
    eps_floor, u_trunc/v_trunc flat TV charges, acceptance_tv on eps_accept)
    over this module's cancellation-avoiding templates."""
    fp, verbose = args.fp, args.verbose
    ratio_tol, bb_eval = args.bb_geometric_ratio_tol, args.bb_eval
    x_abs_tol, approx = args.opt_x_abs_tol, args.approx
    u_trunc, v_trunc = args.u_trunc, args.v_trunc
    if u_trunc is None or not (0.0 < u_trunc < 0.5):
        raise ValueError("PTRS requires --u-trunc with 0 < u_trunc < 0.5")
    floor_tol_vars = floor_x_abs_tol_vars(args)
    accept_tol_vars = accept_x_abs_tol_vars(args)
    a, b, invalpha = ptrs_params(lam)

    # ---- floor: max over the two signs of U ----
    _, _, _, c_err = floor_constants(lam)
    floor_raw = max(
        _fpt(fptaylor, f"stable_floor_{fp}_{tag}_{'p' if s > 0 else 'm'}",
             make_floor_template(lam, fp, u_trunc, s), "eps_floor",
             inputs_dir, outputs_dir, env, verbose,
             f"floor s={s:+d} (lambda={lam})",
             ratio_tol, bb_eval, x_abs_tol, floor_tol_vars, approx)
        for s in (+1, -1))
    eps_floor = hormann_proposal_deviation(floor_raw + c_err, a, b)

    # ---- accept: max over the k partition, plus the small-k point queries ----
    acc = 0.0
    for (k_lo, k_hi, mode) in accept_partition(lam, u_trunc):
        e = _fpt(fptaylor, f"stable_accept_{fp}_{tag}_{mode}_{int(k_lo)}",
                 make_accept_template(lam, fp, u_trunc, k_lo, k_hi, mode),
                 "eps_accept", inputs_dir, outputs_dir, env, verbose,
                 f"accept {mode} k in [{k_lo:.0f},{k_hi:.0f}] (lambda={lam})",
                 ratio_tol, bb_eval, x_abs_tol, accept_tol_vars, approx)
        acc = max(acc, e + method_error(lam, k_lo, k_hi, mode))

    # k=0 is outside this window (dist_poisson.ptrs_accept_k_range clamps
    # k_lo to 1: k1_ = k + 1 must stay > 0), same as dist_poisson.py.
    lo, hi = base.ptrs_accept_k_range(lam, u_trunc)
    for k in range(int(lo), min(int(hi), STIRLERR_KMIN - 1) + 1):
        e = _fpt(fptaylor, f"stable_accept_{fp}_{tag}_k{k}",
                 make_accept_point_template(lam, fp, u_trunc, k), "eps_accept",
                 inputs_dir, outputs_dir, env, verbose,
                 f"accept k={k} (lambda={lam})",
                 ratio_tol, bb_eval, x_abs_tol, accept_tol_vars, approx)
        acc = max(acc, e)          # table stirlerr and exact log p(0): no method error

    eps_accept = acc + eps_logv(fptaylor, fp, v_trunc, inputs_dir, outputs_dir,
                                env, verbose, ratio_tol, bb_eval, x_abs_tol,
                                accept_tol_vars, approx)

    tv = (2.0 * u_trunc + v_trunc + 2.0 * invalpha * eps_floor
          + acceptance_tv(eps_accept))
    return eps_floor, eps_accept, tv


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


def default_out_dir(args):
    lf = getattr(args, "lambda_file", None)
    if lf is None:
        return ROOT / "poisson_stable_runs"
    return ROOT / f"poisson_stable_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    lambdas = [str(args.lam)] if args.lam is not None else base.read_lambdas(args.lambda_file)

    rows = []
    for lam in lambdas:
        start = time.perf_counter()
        lam_float = float(lam)
        tag = base.safe_lambda_name(lam)
        try:
            # ---- low range: unchanged, delegated to dist_poisson ----
            if lam_float < SWITCH:
                lr_input = inputs_dir / f"low_range_{args.fp}_lam_{tag}.txt"
                lr_input.write_text(
                    base._make_log_low_range_template(lam, args.fp) if args.use_log
                    else base._make_low_range_template(lam, args.fp))

                code, output = run_command([fptaylor, str(lr_input)], cwd=ROOT, env=env)
                out_path = outputs_dir / f"low_range_{args.fp}_lam_{tag}.out"
                out_path.write_text(output)
                if args.verbose >= 2:
                    print(f"--- FPTaylor low range (lambda={lam}) ---\n{output}")
                if code != 0:
                    raise RuntimeError(f"FPTaylor low range failed for lambda={lam}; see {out_path}")

                errs = extract_abs_errors_by_problem(output)
                if args.use_log:
                    _, low_tv = base._compute_log_low_range_delta(
                        errs["lambda_fp_compute"], errs["log_prod_compute"])
                else:
                    _, _, low_tv = base._compute_low_range_delta(
                        lam_float, errs["L_compute"], errs["prod_compute"])

                ref_tv = computeDeltaLowRange(lam_float, FP_BETA[args.fp])
                row = base._empty_row(lam, args.fp)
                row.update({"regime": "low", "tv": f"{low_tv:.17e}", "ref_tv": f"{ref_tv:.17e}"})
                row["time_s"] = f"{elapsed_since(start):.6f}"
                rows.append(row)
                print(f"lambda={lam} [low] TV={row['tv']} ref_TV={row['ref_tv']}"
                      f" time={format_seconds(float(row['time_s']))}")
                continue

            # ---- high range (stable PTRS) ----
            eps_floor, eps_accept, tv = run_ptrs_stable(
                fptaylor, lam_float, args, tag, inputs_dir, outputs_dir, env)
            ref_tv = computeDeltaHighRange(lam_float, FP_BETA[args.fp])[0]

            row = base._empty_row(lam, args.fp)
            row.update({
                "regime": "ptrs-stable",
                "eps_floor": f"{eps_floor:.17e}",
                "eps_accept": f"{eps_accept:.17e}",
                "tv": f"{tv:.17e}",
                "ref_tv": f"{ref_tv:.17e}",
                "time_s": f"{elapsed_since(start):.6f}",
            })
            rows.append(row)
            print(f"lambda={lam} [PTRS-stable] eps_floor={eps_floor:.6e}"
                  f" eps_accept={eps_accept:.6e} TV={tv:.6e} ref_TV={ref_tv:.6e}"
                  f" time={format_seconds(float(row['time_s']))}")
        except Exception as exc:
            print(f"WARNING: skipping lambda={lam}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    base.write_plot(rows, plot_path, plot_components=plot_components,
                    plot_pgf=plot_pgf)

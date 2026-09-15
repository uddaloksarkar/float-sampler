"""
Zipf sampler FP-error analysis (distributions/zipf.c, legacy_random_zipf).
Single regime: Devroye's rejection sampler, analysed like dist_poisson.py's
PTRS (eps_floor / eps_accept split).

First-cut point-mode analysis: eps_floor and eps_accept are genuine rigorous
FPTaylor bounds on the sampler's two computed quantities, but the TV
combination below has NOT been derived from the algorithm's acceptance
probability the way BTRS/PTRS's has (see dist_binomial._run_btrs_fptaylor) --
it uses the simplest universally-sound (PMF <= 1) bound instead, which is
correct but likely far looser than a distribution-specific derivation would
give. Treat `tv` here as a provisional upper bound, not a validated result.

Algorithm (rejection sampling, one iteration):
  am1 = a - 1;  b = 2^am1                          [setup, once per a]
  U = 1 - rk_double() in (0, 1];  V = rk_double() in [0, 1)
  X = floor(U^(-1/am1))                             -- floor step
  reject if X < 1 (or X too large; not modeled here, see _X_MAX)
  T = (1 + 1/X)^am1
  accept iff V*X*(T-1)/(b-1) <= T/b                  -- accept step

pow(x, y) for real y has no rigorous FPTaylor primitive (Op_nat_pow only
takes natural exponents), so it is modeled as exp(y*log(x)) throughout,
mirroring dist_binomial._make_inversion_template's qn_step = exp(n*log(q)).

Every runner takes a either as a point or as a (lo, hi) interval (interval
mode, --s-range): see dist_common's "Interval (box) mode" section.
"""
import math
import time
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    extract_abs_errors_by_problem,
    run_fptaylor_query,
    iv, param_ivar, fp_var_type,
    vprint, elapsed_since, format_seconds,
    floor_x_abs_tol_vars, accept_x_abs_tol_vars,
    analyse_param_box, max_fields, parse_range, safe_box_name, box_label,
    csv_num, fmt_num, with_param_tols,
)

NAME = "zipf"
CSV_FIELDS = ["a", "a_lo", "a_hi", "fp", "regime", "eps_floor", "eps_accept",
              "tv", "n_boxes", "time_s"]

# Representative literal X values swept for eps_accept (a spot check, not a
# rigorous cover of the whole unbounded support -- see module docstring).
# Also caps the eps_floor domain (U restricted to reach only X <= _X_MAX;
# see make_zipf_floor_template) so its range stays away from U^(-1/am1)'s
# genuine singularity at U = 0.
_X_SWEEP = [1, 2, 3, 5, 10, 20, 50, 100, 1000]
_X_MAX = max(_X_SWEEP)


# ---------------------------------------------------------------------------
# Zipf FPTaylor templates
# ---------------------------------------------------------------------------

def zipf_tail_prob(a, x_max=_X_MAX):
    """P(X > x_max) <= integral_{x_max}^inf t^-a dt = x_max^-(a-1) / (a-1),
    dropping the true PMF's 1/zeta(a) <= 1 normalizing factor (safe: that
    only makes this bound larger, never violated).  Decreasing in a, so an
    interval takes its lo."""
    am1 = iv(a)[0] - 1.0
    return x_max ** (-am1) / am1


def make_zipf_floor_template(a, fp):
    """
    FPTaylor expression for eps_floor: absolute error of
    Y = U^(-1/am1)   [zipf.c: X = floor(pow(U, -1/am1))]

    U is restricted to [U_min, 1], U_min = _X_MAX^-am1, so Y only ranges up
    to _X_MAX -- the X > _X_MAX tail is charged directly via zipf_tail_prob
    instead of asking FPTaylor to bound Y near its true U=0 singularity.
    For an interval a, U_min is taken at a's hi (the smallest U_min): that
    U range contains every a's own, so it is a superset of every point's
    domain (Y then reaches past _X_MAX for smaller a -- sound, if looser).
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    am1 = iv(a)[1] - 1.0
    u_min = _X_MAX ** (-am1)
    v0_max = 1.0 - u_min
    return (
        "Variables\n"
        f"  {fp_var_type(fp)} V0 in [0.0, {v0_max:.20e}],\n"
        + param_ivar("a", a) + ";\n\n"
        + "Definitions\n"
        f"  U      {rnd}= 1.0 - V0,\n"
        f"  am1_   {rnd}= a - 1.0,\n"
        f"  ninv_  {rnd}= -1.0 / am1_,\n"
        f"  zipf_floor {rnd}= exp(ninv_ * log(U));\n\n"
        + "Expressions\n"
        f"  eps_floor = zipf_floor;\n"
    )


def make_zipf_accept_template(a, x, fp):
    """
    FPTaylor expression for eps_accept: absolute error of
    R = T*(b-1) / (b*x*(T-1)), the threshold V is compared against
    [zipf.c: V*X*(T-1)/(b-1) <= T/b, rearranged so the only free input left
    is V's comparison target], at one literal X = x.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        + param_ivar("a", a) + ";\n\n"
        + "Definitions\n"
        f"  am1_   {rnd}= a - 1.0,\n"
        f"  b_     {rnd}= exp(am1_ * log(2.0)),\n"
        f"  ix1_   {rnd}= 1.0 + 1.0 / {x:.1f},\n"
        f"  T_     {rnd}= exp(am1_ * log(ix1_)),\n"
        f"  num_   {rnd}= T_ * (b_ - 1.0),\n"
        f"  den_   {rnd}= b_ * {x:.1f} * (T_ - 1.0),\n"
        f"  zipf_accept {rnd}= num_ / den_;\n\n"
        + "Expressions\n"
        f"  eps_accept = zipf_accept;\n"
    )


def _run_zipf_fptaylor(fptaylor, a, args, tag, inputs_dir, outputs_dir, env):
    """(eps_floor, eps_accept, tv) for one a (point or interval); see the
    module docstring for what tv does and doesn't yet account for."""
    label = _a_label(a)
    fp, verbose = args.fp, args.verbose
    ratio_tol, bb_eval = args.bb_geometric_ratio_tol, args.bb_eval
    x_abs_tol, approx = args.opt_x_abs_tol, args.approx
    floor_tol_vars = with_param_tols(floor_x_abs_tol_vars(args), {"a": a})
    accept_tol_vars = with_param_tols(accept_x_abs_tol_vars(args), {"a": a})
    tail_prob = zipf_tail_prob(a)
    a_hi = iv(a)[1]
    vprint(verbose, f"zipf {label}",
           am1=a_hi - 1.0, u_min=_X_MAX ** (-(a_hi - 1.0)), x_max=_X_MAX,
           x_sweep=_X_SWEEP, tail_prob=tail_prob)

    # ---- floor ----
    floor_input  = inputs_dir  / f"zipf_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"zipf_floor_{fp}_{tag}.out"
    floor_input.write_text(make_zipf_floor_template(a, fp))

    code, output = run_fptaylor_query(fptaylor, floor_input, outputs_dir, env,
                                       ratio_tol, bb_eval, x_abs_tol, floor_tol_vars, approx)
    floor_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor zipf floor ({label}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor zipf floor failed for {label}; see {floor_output}")

    eps_floor = extract_abs_errors_by_problem(output)["eps_floor"]

    # ---- accept: max over the _X_SWEEP spot checks ----
    eps_accept = 0.0
    for x in _X_SWEEP:
        accept_input  = inputs_dir  / f"zipf_accept_{fp}_{tag}_x{x}.txt"
        accept_output = outputs_dir / f"zipf_accept_{fp}_{tag}_x{x}.out"
        accept_input.write_text(make_zipf_accept_template(a, x, fp))

        code, output = run_fptaylor_query(fptaylor, accept_input, outputs_dir,
                                           env, ratio_tol, bb_eval, x_abs_tol,
                                           accept_tol_vars, approx)
        accept_output.write_text(output)
        if verbose >= 2:
            print(f"--- FPTaylor zipf accept ({label}, X={x}) ---\n{output}")
        if code != 0:
            raise RuntimeError(f"FPTaylor zipf accept failed for "
                               f"{label} X={x}; see {accept_output}")
        eps_accept = max(eps_accept,
                         extract_abs_errors_by_problem(output)["eps_accept"])

    # PMF(X) <= PMF(1) = 1/zeta(a) <= 1 always, so a shift of eps_floor in Y
    # reassigns at most 2*eps_floor of probability mass in the worst case
    # (the same "floor can disagree either way" argument as elsewhere, using
    # the universal density bound instead of a sampler-specific one).
    # zipf_tail_prob charges the X > _X_MAX region eps_floor/eps_accept never
    # examine, the same way v_trunc/u_trunc charge their excluded regions.
    tv = 2.0 * eps_floor + 2.0 * eps_accept + tail_prob
    return eps_floor, eps_accept, tv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _a_label(a):
    return box_label({"a": a}) if isinstance(a, tuple) else f"a={a}"


def safe_a_name(a):
    return "a" + f"{a:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def _validate(a, loc=""):
    prefix = f"{loc}: " if loc else ""
    if a <= 1.0:
        raise ValueError(f"{prefix}a must be > 1")


def read_as(path):
    values = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        for token in line.split():
            try:
                a = float(token)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid a {token!r}") from exc
            _validate(a, f"{path}:{lineno}")
            values.append(a)
    return values


def _empty_row(a, fp):
    return {"a": "" if a == "" else f"{a:.17g}", "a_lo": "", "a_hi": "",
            "fp": fp, "regime": "", "eps_floor": "", "eps_accept": "", "tv": "",
            "n_boxes": "", "time_s": ""}


# ---------------------------------------------------------------------------
# Interval mode  (--s-range)
# ---------------------------------------------------------------------------

def run_box(args, fptaylor, inputs_dir, outputs_dir, env, a_iv):
    """One row bounding TV over every a in a_iv (a single regime, so the
    only splitting is --split-depth's)."""
    start = time.perf_counter()

    def analyse(sub, regime):
        eps_floor, eps_accept, tv = _run_zipf_fptaylor(
            fptaylor, sub["a"], args, safe_box_name(sub), inputs_dir, outputs_dir, env)
        return {"eps_floor": eps_floor, "eps_accept": eps_accept, "tv": tv}

    fields = ("eps_floor", "eps_accept", "tv")
    results = analyse_param_box({"a": a_iv}, lambda box: {"rejection"}, None, analyse,
                                args.split_depth, verbose=args.verbose)
    worst = max_fields(results, fields)

    row = _empty_row("", args.fp)
    row.update({"a_lo": f"{a_iv[0]:.17g}", "a_hi": f"{a_iv[1]:.17g}",
                "regime": "rejection", "n_boxes": len(results),
                "time_s": f"{elapsed_since(start):.6f}"})
    row.update({f: csv_num(worst[f]) for f in fields})
    print(f"{box_label({'a': a_iv})} [rejection] boxes={len(results)}"
          f" eps_floor={fmt_num(worst['eps_floor'])}"
          f" eps_accept={fmt_num(worst['eps_accept'])} TV={fmt_num(worst['tv'])}"
          f" time={format_seconds(float(row['time_s']))}")
    return row


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with a values, one or more per line")
    source.add_argument("--s", type=float, default=None,
                        help="Single exponent a > 1 (zipf.c's `a`)")
    source.add_argument("--s-range", nargs=2, type=float, default=None,
                        metavar=("AMIN", "AMAX"),
                        help="Interval mode: one TV bound valid for every a "
                             "in [AMIN, AMAX] (see --split-depth)")


def default_out_dir(args):
    if getattr(args, "s_range", None) is not None:
        return ROOT / "zipf_runs_interval"
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "zipf_runs"
    return ROOT / f"zipf_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    if args.s_range is not None:
        a_iv = parse_range(args.s_range, "--s-range")
        _validate(a_iv[0], "--s-range")
        return [run_box(args, fptaylor, inputs_dir, outputs_dir, env, a_iv)]
    if args.s is not None:
        _validate(args.s)
        values = [args.s]
    else:
        values = read_as(args.input_file)
    if not values:
        raise ValueError("no a values found in input")

    rows = []
    for a in values:
        start = time.perf_counter()
        tag = safe_a_name(a)
        try:
            row = _empty_row(a, args.fp)

            # ---- rejection (only regime) ----
            eps_floor, eps_accept, tv = _run_zipf_fptaylor(
                fptaylor, a, args, tag, inputs_dir, outputs_dir, env)

            row.update({
                "regime": "rejection",
                "eps_floor": f"{eps_floor:.17e}",
                "eps_accept": f"{eps_accept:.17e}",
                "tv": f"{tv:.17e}",
                "time_s": f"{elapsed_since(start):.6f}",
            })
            rows.append(row)
            print(f"a={a} [rejection] eps_floor={eps_floor:.6e}"
                  f" eps_accept={eps_accept:.6e} TV={tv:.6e}"
                  f" time={format_seconds(float(row['time_s']))}")
        except Exception as exc:
            print(f"WARNING: skipping a={a}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    import contextlib
    import os

    rows = sorted((r for r in rows if r["a"] and math.isfinite(float(r["tv"]))
                   and float(r["tv"]) > 0),
                  key=lambda r: float(r["a"]))
    if not rows:
        print("Nothing to plot")
        return False

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        a_vals = [float(r["a"]) for r in rows]
        tv_vals = [float(r["tv"]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.semilogy(a_vals, tv_vals, marker="o", label="TV")
        if plot_components:
            ax.semilogy(a_vals, [float(r["eps_floor"]) for r in rows],
                       marker="s", alpha=0.6, label="eps_floor")
            ax.semilogy(a_vals, [float(r["eps_accept"]) for r in rows],
                       marker="d", alpha=0.6, label="eps_accept")
        ax.legend()
        ax.set_xlabel("a")
        ax.set_ylabel("TV")
        ax.set_title("Zipf FP error (provisional TV bound)")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        if plot_pgf:
            plt.savefig(plot_path.with_suffix(".pgf"), backend="pgf")
        plt.close()

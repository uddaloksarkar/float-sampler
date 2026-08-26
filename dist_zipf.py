"""
Zipf sampler FP-error analysis (distributions/zipf.c, legacy_random_zipf).

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
mirroring dist_binomial.make_template's qn_step = exp(n*log(q)).
"""
import math
import shutil
import tempfile
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_abs_errors_by_problem,
    vprint, fptaylor_cmd, point_ivar,
    floor_x_abs_tol_vars, accept_x_abs_tol_vars,
)

_FP_VAR_TYPE = {"fp32": "float32", "fp64": "float64", "fp128": "float128"}

NAME = "zipf"
CSV_FIELDS = ["a", "eps_floor", "eps_accept", "tv", "n_boxes"]

# Representative literal X values swept for eps_accept (a spot check, not a
# rigorous cover of the whole unbounded support -- see module docstring).
# Also caps the eps_floor domain (U restricted to reach only X <= _X_MAX;
# see make_zipf_floor_template) so its range stays away from U^(-1/am1)'s
# genuine singularity at U = 0.
_X_SWEEP = [1, 2, 3, 5, 10, 20, 50, 100, 1000]
_X_MAX = max(_X_SWEEP)


def _rnd(fp):
    return FP_TO_FPTAYLOR_RND[fp]


def zipf_tail_prob(a, x_max=_X_MAX):
    """P(X > x_max) <= integral_{x_max}^inf t^-a dt = x_max^-(a-1) / (a-1),
    dropping the true PMF's 1/zeta(a) <= 1 normalizing factor (safe: that
    only makes this bound larger, never violated)."""
    am1 = a - 1.0
    return x_max ** (-am1) / am1


def make_zipf_floor_template(a, fp):
    """eps_floor: absolute error of Y = U^(-1/am1) [zipf.c: X = floor(pow(U, -1/am1))].
    U is restricted to [U_min, 1], U_min = _X_MAX^-am1, so Y only ranges up
    to _X_MAX -- the X > _X_MAX tail is charged directly via zipf_tail_prob
    instead of asking FPTaylor to bound Y near its true U=0 singularity."""
    rnd = _rnd(fp)
    am1 = a - 1.0
    u_min = _X_MAX ** (-am1)
    v0_max = 1.0 - u_min
    return (
        "Variables\n"
        f"  {_FP_VAR_TYPE[fp]} V0 in [0.0, {v0_max:.20e}],\n"
        + point_ivar("a", a) + ";\n\n"
        + "Definitions\n"
        f"  U      {rnd}= 1.0 - V0,\n"
        f"  am1_   {rnd}= a - 1.0,\n"
        f"  ninv_  {rnd}= -1.0 / am1_,\n"
        f"  Y_     {rnd}= exp(ninv_ * log(U));\n\n"
        + "Expressions\n"
        f"  eps_floor = Y_;\n"
    )


def make_zipf_accept_template(a, x, fp):
    """eps_accept: absolute error of R = T*(b-1) / (b*x*(T-1)), the threshold
    V is compared against [zipf.c: V*X*(T-1)/(b-1) <= T/b, rearranged so the
    only free input left is V's comparison target]."""
    rnd = _rnd(fp)
    return (
        "Variables\n"
        + point_ivar("a", a) + ";\n\n"
        + "Definitions\n"
        f"  am1_   {rnd}= a - 1.0,\n"
        f"  b_     {rnd}= exp(am1_ * log(2.0)),\n"
        f"  ix1_   {rnd}= 1.0 + 1.0 / {x:.1f},\n"
        f"  T_     {rnd}= exp(am1_ * log(ix1_)),\n"
        f"  num_   {rnd}= T_ * (b_ - 1.0),\n"
        f"  den_   {rnd}= b_ * {x:.1f} * (T_ - 1.0),\n"
        f"  R_     {rnd}= num_ / den_;\n\n"
        + "Expressions\n"
        f"  eps_accept = R_;\n"
    )


def _run_query(fptaylor, text, stem, expr, args, inputs_dir, outputs_dir, env,
               x_abs_tol_vars):
    input_path = inputs_dir / f"{stem}.txt"
    input_path.write_text(text)
    work = Path(tempfile.mkdtemp(prefix="fpt_", dir=outputs_dir))
    try:
        code, output = run_command(
            fptaylor_cmd(fptaylor, input_path, work, args.bb_geometric_ratio_tol,
                        args.bb_eval, args.opt_x_abs_tol, x_abs_tol_vars),
            cwd=ROOT, env=env)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out_path = outputs_dir / f"{stem}.out"
    out_path.write_text(output)
    if args.verbose >= 2:
        print(f"--- FPTaylor {stem} ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor failed on {stem}; see {out_path}")
    errors = extract_abs_errors_by_problem(output)
    if expr not in errors:
        raise RuntimeError(f"FPTaylor reported no {expr} bound on {stem}; see {out_path}")
    return errors[expr]


def _run_zipf_fptaylor(fptaylor, a, tag, args, inputs_dir, outputs_dir, env):
    """(eps_floor, eps_accept, tv, n_boxes) for one a; see module docstring
    for what tv does and doesn't yet account for."""
    eps_floor = _run_query(
        fptaylor, make_zipf_floor_template(a, args.fp), f"zipf_floor_{args.fp}_{tag}",
        "eps_floor", args, inputs_dir, outputs_dir, env, floor_x_abs_tol_vars(args))

    eps_accept = 0.0
    for x in _X_SWEEP:
        e = _run_query(
            fptaylor, make_zipf_accept_template(a, x, args.fp),
            f"zipf_accept_{args.fp}_{tag}_x{x}", "eps_accept", args,
            inputs_dir, outputs_dir, env, accept_x_abs_tol_vars(args))
        eps_accept = max(eps_accept, e)

    # PMF(X) <= PMF(1) = 1/zeta(a) <= 1 always, so a shift of eps_floor in Y
    # reassigns at most 2*eps_floor of probability mass in the worst case
    # (the same "floor can disagree either way" argument as elsewhere, using
    # the universal density bound instead of a, sampler-specific one).
    # zipf_tail_prob charges the X > _X_MAX region eps_floor/eps_accept never
    # examine, the same way v_trunc/u_trunc charge their excluded regions.
    tv = 2.0 * eps_floor + 2.0 * eps_accept + zipf_tail_prob(a)
    n_boxes = 1 + len(_X_SWEEP)
    return eps_floor, eps_accept, tv, n_boxes


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with a values, one per line")
    source.add_argument("--s", type=float, default=None,
                        help="Exponent parameter a > 1 (zipf.c's `a`)")


def default_out_dir(args):
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "zipf_runs"
    return ROOT / f"zipf_runs_{lf.stem}"


def safe_a_name(a):
    return "a" + f"{a:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def read_as(path):
    values = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            a = float(line.split()[0])
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid a value") from exc
        if a <= 1.0:
            raise ValueError(f"{path}:{lineno}: a must be > 1")
        values.append(a)
    return values


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    if getattr(args, "s", None) is not None:
        if args.s <= 1.0:
            raise ValueError("--s must be > 1")
        values = [args.s]
    else:
        values = read_as(args.input_file)
    if not values:
        raise ValueError("no a values found in input")

    rows = []
    for a in values:
        tag = safe_a_name(a)
        try:
            eps_floor, eps_accept, tv, n_boxes = _run_zipf_fptaylor(
                fptaylor, a, tag, args, inputs_dir, outputs_dir, env)
            vprint(args.verbose, f"zipf a={a}", eps_floor=eps_floor,
                   eps_accept=eps_accept, tv=tv)
            rows.append({"a": f"{a:.17g}", "eps_floor": f"{eps_floor:.17e}",
                        "eps_accept": f"{eps_accept:.17e}", "tv": f"{tv:.17e}",
                        "n_boxes": n_boxes})
            print(f"a={a} eps_floor={eps_floor:.6e} eps_accept={eps_accept:.6e} TV={tv:.6e}")
        except Exception as exc:
            print(f"WARNING: skipping a={a}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    import os, contextlib

    rows = [r for r in rows if math.isfinite(float(r["tv"])) and float(r["tv"]) > 0]
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
        ax.semilogy(a_vals, tv_vals, marker="o")
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

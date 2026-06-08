"""
Geometric sampler FP-error analysis.
Follows the pattern in dist_binom.py; called by main.py.

Switch at p = 1/3:
  p <  1/3 : inversion analysis  (delta via H(u) = log(1-u)/log(1-p))
  p >= 1/3 : search analysis     (eps0/eps1/eps2 via sequential-search template)
"""
import math
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_deltas_by_problem, extract_abs_errors_by_problem,
    save_loglog_plot,
)

NAME = "geometric"
CSV_FIELDS = ["p", "eps0", "eps1", "eps2", "delta", "tv"]

_SWITCH = 1.0 / 3.0


def _use_inversion(p):
    return p < _SWITCH


# ---------------------------------------------------------------------------
# FPTaylor templates
# ---------------------------------------------------------------------------

def _make_search_template(p, fp):
    """
    Sequential-search sampler template (p >= 1/3).

        q     = 1.0 - p
        prod *= q            i.e. prod = z * q,    z   in [p*e^-22, 1]
        sum  += prod         i.e. sum  = sum + prod, sum in [p, 1], prod in [0, 1]

      eps0 : rel. error of q    = 1.0 - p
      eps1 : rel. error of prod = z * q
      eps2 : rel. error of sum  = sum + prod
    """
    z_lo = p * math.exp(-22)
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real z in [{z_lo:.20e}, 1.0],\n"
        f"  real sum in [{p:.20e}, 1.0];\n\n"
        + "Definitions\n"
        f"  p = {p:.20e},\n"
        f"  q         {rnd}= 1.0 - p,\n"
        f"  prod      {rnd}= z * q,\n"
        f"  sum_step  {rnd}= sum + prod;\n\n"
        + "Expressions\n"
        f"  eps0 = q;\n"
        f"  eps1 = prod;\n"
        f"  eps2 = sum_step;\n"
    )


def _make_inversion_template(p, fp):
    """
    Inversion / log-formula template (p < 1/3).

        H(u) = log(1-u) / log(1-p),  u in [0, 0.9999999]

      delta : abs error of rnd64(rnd64(log(1-u)) / rnd64(log(1-p)))
                        vs  exact   log(1-u) / log(1-p)
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real u in [0.0, 9.99999900000000000000e-01];\n\n"
        + "Definitions\n"
        f"  p = {p:.20e},\n"
        f"  log_q   {rnd}= log(1.0 - p),\n"
        f"  log_1mu {rnd}= log(1.0 - u),\n"
        f"  H       {rnd}= log_1mu / log_q;\n\n"
        + "Expressions\n"
        f"  delta = H;\n"
    )


def make_template(p, fp):
    if _use_inversion(p):
        return _make_inversion_template(p, fp)
    return _make_search_template(p, fp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_p_name(p):
    return "p" + f"{p:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def read_ps(path):
    ps = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        for token in line.split():
            try:
                p = float(token)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid p {token!r}") from exc
            if not (0 < p < 1):
                raise ValueError(f"{path}:{lineno}: p must be in (0, 1)")
            ps.append(p)
    return ps


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with p values, one per line")
    source.add_argument("--p", type=float, default=None,
                        help="Single probability p in (0,1)")


def default_out_dir(args):
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "geometric_runs"
    return ROOT / f"geometric_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    if getattr(args, "p", None) is not None:
        if not (0 < args.p < 1):
            raise ValueError("--p must be in (0, 1)")
        ps = [args.p]
    else:
        ps = read_ps(args.input_file)
    if not ps:
        raise ValueError("no p values found in input")

    rows = []
    for p in ps:
        tag = safe_p_name(p)

        input_path = inputs_dir / f"geometric_{args.fp}_{tag}.txt"
        input_path.write_text(make_template(p, args.fp))

        code, output = run_command(
            [fptaylor, "--rel-error", "true", str(input_path)],
            cwd=ROOT, env=env,
        )
        out_path = outputs_dir / f"geometric_{args.fp}_{tag}.out"
        out_path.write_text(output)
        if args.verbose:
            print(f"--- FPTaylor geometric (p={p}) ---\n{output}")
        if code != 0:
            raise RuntimeError(f"FPTaylor failed for p={p}; see {out_path}")

        if _use_inversion(p):
            abs_errors = extract_abs_errors_by_problem(output)
            if "delta" not in abs_errors:
                raise RuntimeError(f"p={p}: could not parse absolute error for 'delta'")
            delta = abs_errors["delta"]
            log_inv_q = math.log(1.0 / (1.0 - p))
            tv = 2.0 * (1.0 - p) / p * math.sinh(delta * log_inv_q) + 1e-7
            row = {
                "p":    f"{p:.17g}",
                "eps0": "nan", "eps1": "nan", "eps2": "nan",
                "delta": f"{delta:.17e}",
                "tv":    f"{tv:.17e}",
            }
            print(f"p={p} delta={delta:.6e} TV={tv:.6e}")
        else:
            deltas = extract_deltas_by_problem(output, f"p={p}")
            eps0 = deltas["eps0"]
            eps1 = deltas["eps1"]
            eps2 = deltas["eps2"]
            tv = 0.5 * (eps0 + eps1 * p + eps2 * 7.0 / math.log(1.0 / (1.0 - p)))
            row = {
                "p":    f"{p:.17g}",
                "eps0": f"{eps0:.17e}",
                "eps1": f"{eps1:.17e}",
                "eps2": f"{eps2:.17e}",
                "delta": "nan",
                "tv":    f"{tv:.17e}",
            }
            print(f"p={p} eps0={eps0:.6e} eps1={eps1:.6e} eps2={eps2:.6e} TV={tv:.6e}")

        rows.append(row)

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    xs = [float(r["p"]) for r in rows]
    series = []
    if plot_components:
        inv_rows = [r for r in rows if r["eps0"] != "nan"]
        log_rows = [r for r in rows if r["delta"] != "nan"]
        if inv_rows:
            series += [
                ("eps0", [float(r["eps0"]) for r in inv_rows], "o"),
                ("eps1", [float(r["eps1"]) for r in inv_rows], "s"),
                ("eps2", [float(r["eps2"]) for r in inv_rows], "d"),
            ]
        if log_rows:
            series += [("delta", [float(r["delta"]) for r in log_rows], "x")]
    series.append(("TV", [float(r["tv"]) for r in rows], "^"))
    save_loglog_plot(xs, series, xlabel="p", ylabel="error",
                     plot_path=plot_path, plot_pgf=plot_pgf)

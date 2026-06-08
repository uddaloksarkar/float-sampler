"""
Geometric sampler FP-error analysis.
Follows the pattern in dist_binom.py; called by main.py.
"""
import math
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_deltas_by_problem,
    save_loglog_plot,
)

NAME = "geometric"
CSV_FIELDS = ["p", "eps0", "eps1", "eps2", "tv"]


# ---------------------------------------------------------------------------
# FPTaylor template
# ---------------------------------------------------------------------------

def make_template(p, fp):
    """
    Single FPTaylor input for p with three expressions, one per elementary
    FP operation in random_geometric_search's sampling loop
    (distributions/geometric_search.c):

        q     = 1.0 - p
        prod *= q            i.e. prod = z * q,    z   in [1e-10, 1]
        sum  += prod         i.e. sum  = sum + prod, sum in [1e-10, 1], prod in [0, 1]

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

        deltas = extract_deltas_by_problem(output, f"p={p}")
        eps0 = deltas["eps0"]
        eps1 = deltas["eps1"]
        eps2 = deltas["eps2"]
        tv = 0.5 * (eps0 + eps1 * p + eps2 * 7.0 / math.log(1.0 / (1.0 - p)))

        rows.append({
            "p": f"{p:.17g}",
            "eps0": f"{eps0:.17e}",
            "eps1": f"{eps1:.17e}",
            "eps2": f"{eps2:.17e}",
            "tv": f"{tv:.17e}",
        })
        print(f"p={p} eps0={eps0:.6e} eps1={eps1:.6e} eps2={eps2:.6e} TV={tv:.6e}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    xs = [float(r["p"]) for r in rows]
    series = []
    if plot_components:
        series += [
            ("eps0", [float(r["eps0"]) for r in rows], "o"),
            ("eps1", [float(r["eps1"]) for r in rows], "s"),
            ("eps2", [float(r["eps2"]) for r in rows], "d"),
        ]
    series.append(("TV", [float(r["tv"]) for r in rows], "^"))
    save_loglog_plot(xs, series, xlabel="p", ylabel="error",
                     plot_path=plot_path, plot_pgf=plot_pgf)

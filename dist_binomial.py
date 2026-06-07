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
    save_loglog_plot,
)

NAME = "binomial_inversion"
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
    qn = max(math.exp(n * math.log(q)), sys.float_info.min)
    rnd = FP_TO_FPTAYLOR_RND[fp]

    return (
        "Variables\n"
        f"  real z in [1.0e-6, 1.0],\n"
        f"  real X in [1.0, {float(n):.1f}],\n"
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
    if getattr(args, "n", None) is not None:
        return ROOT / "binomial_inversion_runs"
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "binomial_inversion_runs"
    return ROOT / f"binomial_inversion_runs_{lf.stem}"


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

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    xs = [float(r["n"]) * float(r["p"]) for r in rows]
    series = []
    if plot_components:
        series += [
            ("eps0", [float(r["eps0"]) for r in rows], "o"),
            ("eps1", [float(r["eps1"]) for r in rows], "s"),
            ("eps2", [float(r["eps2"]) for r in rows], "d"),
        ]
    series.append(("TV", [float(r["tv"]) for r in rows], "^"))
    save_loglog_plot(xs, series, xlabel="np  (mean)", ylabel="error",
                     plot_path=plot_path, plot_pgf=plot_pgf)

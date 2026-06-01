"""
Binomial sampler FP-error analysis.
Refactored from fpsampler_binom.py; called by main.py.
"""
import sys
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_deltas_by_problem,
    save_loglog_plot,
)

NAME = "binomial"
CSV_FIELDS = ["n", "p", "delta_1", "eps_exp", "tv"]


# ---------------------------------------------------------------------------
# FPTaylor template
# ---------------------------------------------------------------------------

def make_template(n, p, fp):
    """
    Single FPTaylor input for (n, p) with two expressions:
      delta_1 : max_{z in [q^n,1], k in [1,n]} rel. error of z*(n-k+1)*p/(k*q)
      eps_exp : rel. error of exp(n*log(q))  (initialisation probability q^n)
    """
    q = 1.0 - p
    z_lo = max(q ** n, sys.float_info.min)
    rnd = FP_TO_FPTAYLOR_RND[fp]

    return (
        "Variables\n"
        f"  real z in [{z_lo:.20e}, 1.0],\n"
        f"  real k in [1.0, {float(n):.1f}];\n\n"
        + "Definitions\n"
        f"  n = {float(n):.1f},\n"
        f"  p = {p:.20e},\n"
        f"  q = {q:.20e},\n"
        f"  ratio {rnd}= z * (n - k + 1) * p / (k * q),\n"
        f"  init  {rnd}= exp(n * log(q));\n\n"
        + "Expressions\n"
        f"  delta_1 = ratio;\n"
        f"  eps_exp = init;\n"
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
        return ROOT / "binom_runs"
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "binom_runs"
    return ROOT / f"binom_runs_{lf.stem}"


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
        mean = n * p
        tag = safe_pair_name(n, p)

        input_path = inputs_dir / f"binom_{args.fp}_{tag}.txt"
        input_path.write_text(make_template(n, p, args.fp))

        code, output = run_command(
            [fptaylor, "--rel-error", "true", str(input_path)],
            cwd=ROOT, env=env,
        )
        out_path = outputs_dir / f"binom_{args.fp}_{tag}.out"
        out_path.write_text(output)
        if args.verbose:
            print(f"--- FPTaylor binomial (n={n}, p={p}) ---\n{output}")
        if code != 0:
            raise RuntimeError(f"FPTaylor failed for n={n}, p={p}; see {out_path}")

        deltas = extract_deltas_by_problem(output, f"n={n} p={p}")
        delta_1 = deltas["delta_1"]
        eps_exp = deltas["eps_exp"]
        tv = 0.5 * mean * delta_1 + 0.5 * eps_exp

        rows.append({
            "n": n,
            "p": f"{p:.17g}",
            "delta_1": f"{delta_1:.17e}",
            "eps_exp": f"{eps_exp:.17e}",
            "tv": f"{tv:.17e}",
        })
        print(f"n={n} p={p} delta_1={delta_1:.6e} eps_exp={eps_exp:.6e} TV={tv:.6e}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    xs = [float(r["n"]) * float(r["p"]) for r in rows]
    series = []
    if plot_components:
        series += [
            ("delta_1", [float(r["delta_1"]) for r in rows], "o"),
            ("eps_exp", [float(r["eps_exp"]) for r in rows], "s"),
        ]
    series.append(("TV", [float(r["tv"]) for r in rows], "^"))
    save_loglog_plot(xs, series, xlabel="np  (mean)", ylabel="error",
                     plot_path=plot_path, plot_pgf=plot_pgf)

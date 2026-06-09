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
    run_cire_llvm, extract_cire_abs_error,
)

NAME = "geometric"
CSV_FIELDS = ["p", "eps0", "eps1", "eps2", "delta", "tv", "backend"]

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
# CIRE C code
# ---------------------------------------------------------------------------

_GEOM_INVERSION_C = """\
#include <math.h>
/* delta: abs error of log(1-u)/log(1-p), p passed as param to prevent folding */
double geometric_H(double u, double p) { return log(1.0 - u) / log(1.0 - p); }
"""

_GEOM_SEARCH_C = """\
/* eps0: abs error of 1-p */
double geometric_q(double p) { return 1.0 - p; }
/* eps1: abs error of z*q, q = 1-p (exact constant passed in) */
double geometric_prod(double z, double q) { return z * q; }
/* eps2: abs error of sum+prod */
double geometric_sum(double s, double pr) { return s + pr; }
"""


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
# Per-backend analysis helpers
# ---------------------------------------------------------------------------

def _run_fptaylor_geometric(fptaylor, p, tag, args, inputs_dir, outputs_dir, env):
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
        print(f"p={p} delta={delta:.6e} TV={tv:.6e}")
        return {"p": f"{p:.17g}", "eps0": "nan", "eps1": "nan", "eps2": "nan",
                "delta": f"{delta:.17e}", "tv": f"{tv:.17e}", "backend": "fptaylor"}
    else:
        deltas = extract_deltas_by_problem(output, f"p={p}")
        eps0, eps1, eps2 = deltas["eps0"], deltas["eps1"], deltas["eps2"]
        tv = 0.5 * (eps0 + eps1 * p + eps2 * 7.0 / math.log(1.0 / (1.0 - p)))
        print(f"p={p} eps0={eps0:.6e} eps1={eps1:.6e} eps2={eps2:.6e} TV={tv:.6e}")
        return {"p": f"{p:.17g}", "eps0": f"{eps0:.17e}", "eps1": f"{eps1:.17e}",
                "eps2": f"{eps2:.17e}", "delta": "nan", "tv": f"{tv:.17e}", "backend": "fptaylor"}


def _run_cire_geometric(cire, p, tag, args, inputs_dir, outputs_dir):
    q = 1.0 - p

    def _run(c_code, func, domains, label):
        rc, out = run_cire_llvm(cire, c_code, func, domains, tag,
                                inputs_dir, outputs_dir, verbose=args.verbose)
        if rc != 0:
            raise RuntimeError(f"CIRE failed for {label} (p={p})")
        return extract_cire_abs_error(out, label)

    if _use_inversion(p):
        delta = _run(_GEOM_INVERSION_C, "geometric_H",
                     [(0.0, 0.9999999), (p, p)], "delta")
        log_inv_q = math.log(1.0 / q)
        tv = 2.0 * q / p * math.sinh(delta * log_inv_q) + 1e-7
        print(f"p={p} delta={delta:.6e} TV={tv:.6e}")
        return {"p": f"{p:.17g}", "eps0": "nan", "eps1": "nan", "eps2": "nan",
                "delta": f"{delta:.17e}", "tv": f"{tv:.17e}", "backend": "cire"}
    else:
        z_lo = p * math.exp(-22)
        # relative error = abs_error / lower_bound_of_exact_expression
        # eps0: 1-p  → lower bound = q (single-valued)
        # eps1: z*q  → lower bound = z_lo * q
        # eps2: s+pr → lower bound = p (min sum = p, min prod = 0)
        abs0 = _run(_GEOM_SEARCH_C, "geometric_q",   [(p, p)],              "eps0")
        abs1 = _run(_GEOM_SEARCH_C, "geometric_prod", [(z_lo, 1.0), (q, q)], "eps1")
        abs2 = _run(_GEOM_SEARCH_C, "geometric_sum",  [(p, 1.0), (0.0, 1.0)], "eps2")
        eps0 = abs0 / q
        eps1 = abs1 / max(z_lo * q, 1e-300)
        eps2 = abs2 / p
        tv = 0.5 * (eps0 + eps1 * p + eps2 * 7.0 / math.log(1.0 / q))
        print(f"p={p} eps0={eps0:.6e} eps1={eps1:.6e} eps2={eps2:.6e} TV={tv:.6e}")
        return {"p": f"{p:.17g}", "eps0": f"{eps0:.17e}", "eps1": f"{eps1:.17e}",
                "eps2": f"{eps2:.17e}", "delta": "nan", "tv": f"{tv:.17e}", "backend": "cire"}


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
    backend = getattr(args, "backend", "fptaylor")
    stem = lf.stem if lf is not None else ""
    parts = ["geometric_runs"] + ([stem] if stem else []) + [backend]
    return ROOT / "_".join(parts)


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
        try:
            if args.backend == "cire":
                row = _run_cire_geometric(fptaylor, p, tag, args, inputs_dir, outputs_dir)
            else:
                row = _run_fptaylor_geometric(fptaylor, p, tag, args, inputs_dir, outputs_dir, env)
            rows.append(row)
        except Exception as exc:
            print(f"WARNING: skipping p={p}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    import os, contextlib, math

    THRESHOLD = 1.0 / 3.0

    # Both groups use k = -log2(p) as x-coordinate.
    above = [(r, -math.log2(float(r["p"]))) for r in rows if float(r["p"]) >= THRESHOLD]
    below = [(r, -math.log2(float(r["p"]))) for r in rows if float(r["p"]) <  THRESHOLD]

    def _tv(group):
        return [(k, float(r["tv"])) for r, k in group
                if math.isfinite(float(r["tv"])) and float(r["tv"]) > 0]

    def _field(group, key):
        return [(k, float(r[key])) for r, k in group
                if r.get(key, "nan") != "nan" and math.isfinite(float(r[key])) and float(r[key]) > 0]

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

        BACKEND_STYLE = {
            "fptaylor": dict(linestyle="-",  marker="^", color="tab:blue"),
            "cire":     dict(linestyle="--", marker="s", color="tab:orange"),
        }
        COMPONENT_MARKERS = [
            ("eps0", "o"), ("eps1", "s"), ("eps2", "d"), ("delta", "x"),
        ]

        for ax, group, title, xlabel in [
            (axes[0], above, "p ≥ 1/3  (search region)",   "k = −log₂(p)"),
            (axes[1], below, "p < 1/3  (inversion region)", "k = −log₂(p)"),
        ]:
            for backend, style in BACKEND_STYLE.items():
                bgroup = [(r, k) for r, k in group if r.get("backend") == backend]
                if not bgroup:
                    continue

                if plot_components:
                    for label, cmarker in COMPONENT_MARKERS:
                        pts = _field(bgroup, label)
                        if pts:
                            ks, ys = zip(*pts)
                            ax.loglog(ks, ys, marker=cmarker,
                                      linestyle=style["linestyle"],
                                      color=style["color"], alpha=0.6,
                                      label=f"{label} ({backend})")

                pts = _tv(bgroup)
                if pts:
                    ks, ys = zip(*pts)
                    ax.loglog(ks, ys, label=f"TV ({backend})", **style)

            ax.set_xlabel(xlabel)
            ax.set_ylabel("error")
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

        fig.suptitle("Geometric FP error")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        if plot_pgf:
            plt.savefig(plot_path.with_suffix(".pgf"), backend="pgf")
        plt.close()

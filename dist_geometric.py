"""
Geometric sampler FP-error analysis.
Two regimes, split at p = _SWITCH (1/3 by default):
  p >= _SWITCH : search    -- sequential search; relative errors eps0/eps1/eps2
                              of its three FP ops
  p <  _SWITCH : inversion -- X = ceil(log(1-u)/log(1-p)); absolute error
                              delta of H(u) = log(1-u)/log(1-p)
Both regimes run on FPTaylor (default) or CIRE (--backend cire).

Every runner takes p either as a point or as a (lo, hi) interval (interval
mode, --p-range): see dist_common's "Interval (box) mode" section.  The TV
formulas' p-dependent factors are each taken at their own worst end of the
interval.
"""
import math
import time
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    extract_deltas_by_problem, extract_abs_errors_by_problem,
    run_cire_llvm, extract_cire_abs_error, run_fptaylor_isolated,
    iv, param_ivar,
    vprint, elapsed_since, format_seconds,
    dist_switch,
    analyse_param_box, max_fields, parse_range, safe_box_name, box_label,
    csv_num, fmt_num,
)

NAME = "geometric"
CSV_FIELDS = ["p", "p_lo", "p_hi", "fp", "regime", "backend", "eps0", "eps1",
              "eps2", "delta", "tv", "n_boxes", "time_s"]

# p threshold: inversion below, search above -- overridable via
# fptaylor_settings.toml's [geometric].switch (dist_common.dist_switch).
_SWITCH = dist_switch(NAME, 1.0 / 3.0)


# ---------------------------------------------------------------------------
# Search FPTaylor template / CIRE code  (p >= _SWITCH)
# ---------------------------------------------------------------------------

def _make_search_template(p, fp):
    """
    Sequential-search sampler template (p >= _SWITCH).

        q     = 1.0 - p
        prod *= q            i.e. prod = z * q,    z   in [p*e^-22, 1]
        sum  += prod         i.e. sum  = sum + prod, sum in [p, 1], prod in [0, 1]

      eps0 : rel. error of q    = 1.0 - p
      eps1 : rel. error of prod = z * q
      eps2 : rel. error of sum  = sum + prod
    """
    p_lo = iv(p)[0]          # z's and sum's ranges only widen as p shrinks
    z_lo = p_lo * math.exp(-22)
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real z in [{z_lo:.20e}, 1.0],\n"
        f"  real sum in [{p_lo:.20e}, 1.0],\n"
        + param_ivar("p", p) + ";\n\n"
        + "Definitions\n"
        f"  q         {rnd}= 1.0 - p,\n"
        f"  prod      {rnd}= z * q,\n"
        f"  sum_step  {rnd}= sum + prod;\n\n"
        + "Expressions\n"
        f"  eps0 = q;\n"
        f"  eps1 = prod;\n"
        f"  eps2 = sum_step;\n"
    )


_SEARCH_C = """\
/* eps0: abs error of 1-p */
double geometric_q(double p) { return 1.0 - p; }
/* eps1: abs error of z*q, q = 1-p (exact constant passed in) */
double geometric_prod(double z, double q) { return z * q; }
/* eps2: abs error of sum+prod */
double geometric_sum(double s, double pr) { return s + pr; }
"""


def _compute_search_tv(p, eps0, eps1, eps2):
    """tv from the search loop's per-op relative errors: eps2 is charged on
    the 7/log(1/q) expected summation steps.  For an interval, eps1's factor
    p is largest at hi and 7/log(1/q) at lo."""
    p_lo, p_hi = iv(p)
    return 0.5 * (eps0 + eps1 * p_hi + eps2 * 7.0 / math.log(1.0 / (1.0 - p_lo)))


def _run_search_fptaylor(fptaylor, p, args, tag, inputs_dir, outputs_dir, env):
    """(eps0, eps1, eps2, tv) for the search regime at p (point or
    interval), via FPTaylor."""
    label = _p_label(p)
    fp, verbose = args.fp, args.verbose
    vprint(verbose, f"geometric search {label}", z_lo=iv(p)[0] * math.exp(-22))

    search_input  = inputs_dir  / f"geometric_search_{fp}_{tag}.txt"
    search_output = outputs_dir / f"geometric_search_{fp}_{tag}.out"
    search_input.write_text(_make_search_template(p, fp))

    # See dist_common.run_fptaylor_isolated's docstring: a bare run_command
    # here (no --tmp-base-dir/--log-base-dir) shares FPTaylor's default
    # tmp/log dirs across every concurrent search query, observed in
    # practice (dist_binomial's identical pattern) as
    # "Sys_error(...log: Permission denied)" under
    # run_interval_benchmarks.sh's parallel xargs -P.
    code, output = run_fptaylor_isolated(
        fptaylor, ["--rel-error", "true"], search_input, outputs_dir, env)
    search_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor search ({label}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor search failed for {label}; see {search_output}")

    deltas = extract_deltas_by_problem(output, label)
    eps0, eps1, eps2 = deltas["eps0"], deltas["eps1"], deltas["eps2"]
    return eps0, eps1, eps2, _compute_search_tv(p, eps0, eps1, eps2)


def _run_search_cire(cire, p, args, tag, inputs_dir, outputs_dir):
    """(eps0, eps1, eps2, tv) for the search regime at p (point or
    interval), via CIRE.  CIRE reports absolute errors; each is turned into
    a relative one by dividing by the lower bound of its exact expression."""
    label = _p_label(p)
    p_lo, p_hi = iv(p)
    q_lo, q_hi = 1.0 - p_hi, 1.0 - p_lo
    z_lo = p_lo * math.exp(-22)
    vprint(args.verbose, f"geometric search {label}", z_lo=z_lo)

    def one(func, domains, what):
        rc, out = run_cire_llvm(cire, _SEARCH_C, func, domains, tag,
                                inputs_dir, outputs_dir, verbose=args.verbose)
        if rc != 0:
            raise RuntimeError(f"CIRE search failed for {what} ({label})")
        return extract_cire_abs_error(out, what)

    # eps0: 1-p  -> lower bound = q
    # eps1: z*q  -> lower bound = z_lo * q
    # eps2: s+pr -> lower bound = p (min sum = p, min prod = 0)
    eps0 = one("geometric_q",    [(p_lo, p_hi)],               "eps0") / q_lo
    eps1 = one("geometric_prod", [(z_lo, 1.0), (q_lo, q_hi)],  "eps1") / max(z_lo * q_lo, 1e-300)
    eps2 = one("geometric_sum",  [(p_lo, 1.0), (0.0, 1.0)],    "eps2") / p_lo
    return eps0, eps1, eps2, _compute_search_tv(p, eps0, eps1, eps2)


# ---------------------------------------------------------------------------
# Inversion FPTaylor template / CIRE code  (p < _SWITCH)
# ---------------------------------------------------------------------------

def _make_inversion_template(p, fp):
    """
    Inversion / log-formula template (p < _SWITCH).

        H(u) = log(1-u) / log(1-p),  u in [0, 0.9999999]

      delta : abs error of rnd64(rnd64(log(1-u)) / rnd64(log(1-p)))
                        vs  exact   log(1-u) / log(1-p)
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real u in [0.0, 9.99999900000000000000e-01],\n"
        + param_ivar("p", p) + ";\n\n"
        + "Definitions\n"
        f"  log_q   {rnd}= log(1.0 - p),\n"
        f"  log_1mu {rnd}= log(1.0 - u),\n"
        f"  H       {rnd}= log_1mu / log_q;\n\n"
        + "Expressions\n"
        f"  delta = H;\n"
    )


_INVERSION_C = """\
#include <math.h>
/* delta: abs error of log(1-u)/log(1-p), p passed as param to prevent folding */
double geometric_H(double u, double p) { return log(1.0 - u) / log(1.0 - p); }
"""


def _compute_inversion_tv(p, delta):
    """tv from H's absolute error delta, plus the 1e-7 of u-mass above the
    analysed u <= 0.9999999.

    For an interval, write tv - 1e-7 = 2*delta * h(p) * S(delta*L(p)) with
    L = log(1/q), h(p) = (1-p)*L(p)/p and S(t) = sinh(t)/t: h decreases in
    p (from 1 at p -> 0; checked in tests/test_interval.py) and S increases
    in t, with L increasing in p, so h is taken at lo and S at hi.  Taking
    (1-p)/p and L at opposite ends instead was ~300x loose over p in
    [0.001, 0.3]."""
    if not isinstance(p, tuple):
        log_inv_q = math.log(1.0 / (1.0 - p))
        return 2.0 * (1.0 - p) / p * math.sinh(delta * log_inv_q) + 1e-7
    p_lo, p_hi = p
    h_lo = (1.0 - p_lo) * math.log(1.0 / (1.0 - p_lo)) / p_lo
    t_hi = delta * math.log(1.0 / (1.0 - p_hi))
    s_hi = math.sinh(t_hi) / t_hi if t_hi > 0.0 else 1.0
    return 2.0 * delta * h_lo * s_hi + 1e-7


def _run_inversion_fptaylor(fptaylor, p, args, tag, inputs_dir, outputs_dir, env):
    """(delta, tv) for the inversion regime at p (point or interval), via
    FPTaylor."""
    label = _p_label(p)
    fp, verbose = args.fp, args.verbose
    vprint(verbose, f"geometric inversion {label}", log_q=math.log(1.0 - iv(p)[1]))

    inv_input  = inputs_dir  / f"geometric_inversion_{fp}_{tag}.txt"
    inv_output = outputs_dir / f"geometric_inversion_{fp}_{tag}.out"
    inv_input.write_text(_make_inversion_template(p, fp))

    # See dist_common.run_fptaylor_isolated's docstring: a bare run_command
    # here (no --tmp-base-dir/--log-base-dir) shares FPTaylor's default
    # tmp/log dirs across every concurrent inversion query, observed in
    # practice (dist_binomial's identical pattern) as
    # "Sys_error(...log: Permission denied)" under
    # run_interval_benchmarks.sh's parallel xargs -P.
    code, output = run_fptaylor_isolated(
        fptaylor, ["--rel-error", "true"], inv_input, outputs_dir, env)
    inv_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor inversion ({label}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor inversion failed for {label}; see {inv_output}")

    abs_errors = extract_abs_errors_by_problem(output)
    if "delta" not in abs_errors:
        raise RuntimeError(f"{label}: could not parse absolute error for 'delta'")
    delta = abs_errors["delta"]
    return delta, _compute_inversion_tv(p, delta)


def _run_inversion_cire(cire, p, args, tag, inputs_dir, outputs_dir):
    """(delta, tv) for the inversion regime at p (point or interval), via
    CIRE."""
    label = _p_label(p)
    vprint(args.verbose, f"geometric inversion {label}", log_q=math.log(1.0 - iv(p)[1]))
    rc, out = run_cire_llvm(cire, _INVERSION_C, "geometric_H",
                            [(0.0, 0.9999999), iv(p)], tag,
                            inputs_dir, outputs_dir, verbose=args.verbose)
    if rc != 0:
        raise RuntimeError(f"CIRE inversion failed for delta ({label})")
    delta = extract_cire_abs_error(out, "delta")
    return delta, _compute_inversion_tv(p, delta)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _use_inversion(p):
    return p < _SWITCH


def _p_label(p):
    return box_label({"p": p}) if isinstance(p, tuple) else f"p={p}"


def safe_p_name(p):
    return "p" + f"{p:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def _validate(p, loc=""):
    prefix = f"{loc}: " if loc else ""
    if not (0 < p < 1):
        raise ValueError(f"{prefix}p must be in (0, 1)")


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
            _validate(p, f"{path}:{lineno}")
            ps.append(p)
    return ps


def _empty_row(p, fp, backend):
    return {"p": "" if p == "" else f"{p:.17g}", "p_lo": "", "p_hi": "",
            "fp": fp, "regime": "", "backend": backend,
            "eps0": "", "eps1": "", "eps2": "", "delta": "", "tv": "",
            "n_boxes": "", "time_s": ""}


# ---------------------------------------------------------------------------
# Interval mode  (--p-range)
# ---------------------------------------------------------------------------

def _box_regimes(box):
    lo, hi = box["p"]
    return (({"inversion"} if lo < _SWITCH else set())
            | ({"search"} if hi >= _SWITCH else set()))


def _split_at_switch(box):
    lo, hi = box["p"]
    return [{"p": (lo, math.nextafter(_SWITCH, 0.0))}, {"p": (_SWITCH, hi)}]


def run_box(args, fptaylor, inputs_dir, outputs_dir, env, p_iv):
    """One row bounding TV over every p in p_iv."""
    start = time.perf_counter()
    cire = args.backend == "cire"

    def analyse(sub, regime):
        p, tag = sub["p"], safe_box_name(sub)
        if regime == "inversion":
            if cire:
                delta, tv = _run_inversion_cire(
                    fptaylor, p, args, tag, inputs_dir, outputs_dir)
            else:
                delta, tv = _run_inversion_fptaylor(
                    fptaylor, p, args, tag, inputs_dir, outputs_dir, env)
            return {"delta": delta, "tv": tv}
        if cire:
            eps0, eps1, eps2, tv = _run_search_cire(
                fptaylor, p, args, tag, inputs_dir, outputs_dir)
        else:
            eps0, eps1, eps2, tv = _run_search_fptaylor(
                fptaylor, p, args, tag, inputs_dir, outputs_dir, env)
        return {"eps0": eps0, "eps1": eps1, "eps2": eps2, "tv": tv}

    fields = ("eps0", "eps1", "eps2", "delta", "tv")
    results = analyse_param_box({"p": p_iv}, _box_regimes, _split_at_switch, analyse,
                                args.split_depth, verbose=args.verbose)
    worst = max_fields(results, fields)
    regimes = sorted({r["regime"] for r in results})

    row = _empty_row("", args.fp, args.backend)
    row.update({"p_lo": f"{p_iv[0]:.17g}", "p_hi": f"{p_iv[1]:.17g}",
                "regime": "+".join(regimes), "n_boxes": len(results),
                "time_s": f"{elapsed_since(start):.6f}"})
    row.update({f: csv_num(worst[f]) for f in fields})
    print(f"{box_label({'p': p_iv})} [{'+'.join(regimes)}] boxes={len(results)}"
          f" eps0={fmt_num(worst['eps0'])} eps1={fmt_num(worst['eps1'])}"
          f" eps2={fmt_num(worst['eps2'])} delta={fmt_num(worst['delta'])}"
          f" TV={fmt_num(worst['tv'])} time={format_seconds(float(row['time_s']))}")
    return row


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with p values, one or more per line")
    source.add_argument("--p", type=float, default=None,
                        help="Single probability p in (0,1)")
    source.add_argument("--p-range", nargs=2, type=float, default=None,
                        metavar=("PMIN", "PMAX"),
                        help="Interval mode: one TV bound valid for every p "
                             "in [PMIN, PMAX] (see --split-depth)")


def default_out_dir(args):
    if getattr(args, "p_range", None) is not None:
        return ROOT / f"geometric_runs_interval_{args.backend}"
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / f"geometric_runs_{args.backend}"
    return ROOT / f"geometric_runs_{lf.stem}_{args.backend}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    # `fptaylor` is whichever backend binary main.py resolved (CIRE_LLVM
    # under --backend cire).
    if args.p_range is not None:
        p_iv = parse_range(args.p_range, "--p-range")
        for p in p_iv:
            _validate(p, "--p-range")
        return [run_box(args, fptaylor, inputs_dir, outputs_dir, env, p_iv)]
    if args.p is not None:
        _validate(args.p)
        ps = [args.p]
    else:
        ps = read_ps(args.input_file)
    if not ps:
        raise ValueError("no p values found in input")
    cire = args.backend == "cire"

    rows = []
    for p in ps:
        start = time.perf_counter()
        tag = safe_p_name(p)
        try:
            row = _empty_row(p, args.fp, args.backend)

            # ---- low range (inversion) ----
            if _use_inversion(p):
                if cire:
                    delta, tv = _run_inversion_cire(
                        fptaylor, p, args, tag, inputs_dir, outputs_dir)
                else:
                    delta, tv = _run_inversion_fptaylor(
                        fptaylor, p, args, tag, inputs_dir, outputs_dir, env)

                row.update({
                    "regime": "inversion",
                    "delta": f"{delta:.17e}",
                    "tv": f"{tv:.17e}",
                    "time_s": f"{elapsed_since(start):.6f}",
                })
                rows.append(row)
                print(f"p={p} [inversion] delta={delta:.6e} TV={tv:.6e}"
                      f" time={format_seconds(float(row['time_s']))}")
                continue

            # ---- high range (search) ----
            if cire:
                eps0, eps1, eps2, tv = _run_search_cire(
                    fptaylor, p, args, tag, inputs_dir, outputs_dir)
            else:
                eps0, eps1, eps2, tv = _run_search_fptaylor(
                    fptaylor, p, args, tag, inputs_dir, outputs_dir, env)

            row.update({
                "regime": "search",
                "eps0": f"{eps0:.17e}",
                "eps1": f"{eps1:.17e}",
                "eps2": f"{eps2:.17e}",
                "tv": f"{tv:.17e}",
                "time_s": f"{elapsed_since(start):.6f}",
            })
            rows.append(row)
            print(f"p={p} [search] eps0={eps0:.6e} eps1={eps1:.6e}"
                  f" eps2={eps2:.6e} TV={tv:.6e}"
                  f" time={format_seconds(float(row['time_s']))}")
        except Exception as exc:
            print(f"WARNING: skipping p={p}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    import contextlib
    import os

    rows = [r for r in rows if r["p"]]          # interval rows aren't points
    if not rows:
        print("Nothing to plot: interval-mode rows are not points on the p axis")
        return False

    # Both groups use k = -log2(p) as x-coordinate.
    search    = [(r, -math.log2(float(r["p"]))) for r in rows if r["regime"] == "search"]
    inversion = [(r, -math.log2(float(r["p"]))) for r in rows if r["regime"] == "inversion"]

    def _field(group, key):
        return [(k, float(r[key])) for r, k in group
                if r.get(key) and math.isfinite(float(r[key])) and float(r[key]) > 0]

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
            (axes[0], search,    f"p ≥ {_SWITCH:.3g}  (search region)",    "k = −log₂(p)"),
            (axes[1], inversion, f"p < {_SWITCH:.3g}  (inversion region)", "k = −log₂(p)"),
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

                pts = _field(bgroup, "tv")
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

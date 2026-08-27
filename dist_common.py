"""
Shared utilities for FP-error analysis scripts.
All distribution modules import from here.
"""
import argparse
import contextlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

import interval_error as ie
from analyticError import FP_BETA

ROOT = Path(__file__).resolve().parent

FP_TO_FPTAYLOR_RND = {
    "fp32": "rnd32",
    "fp64": "rnd64",
    "fp128": "rnd128",
}

ABS_ERROR_RE = re.compile(r"Absolute error \(exact\)[^:]*:\s*([-+\deE.]+)")
BOUNDS_LO_RE = re.compile(r"Bounds \(without rounding\):\s*\[([-+\deE.]+),")
CIRE_ABS_ERROR_RE = re.compile(r"Absolute Error Bound:\s*([-+\deE.]+)")


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def find_fptaylor(explicit=None):
    if explicit:
        return explicit
    env_val = os.environ.get("FPTAYLOR")
    if env_val:
        return env_val
    local = ROOT / "FPTaylor" / "fptaylor"
    if local.exists():
        return str(local)
    return shutil.which("fptaylor")


def find_cire(explicit=None):
    if explicit:
        return explicit
    env_val = os.environ.get("CIRE")
    if env_val:
        return env_val
    local = ROOT / "cire" / "build" / "CIRE_LLVM"
    if local.exists():
        return str(local)
    return shutil.which("CIRE_LLVM")


def find_clang():
    clang = shutil.which("clang")
    if clang:
        return clang
    try:
        result = subprocess.run(["xcrun", "-f", "clang"],
                                capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    raise RuntimeError("clang not found; required by the CIRE backend")


def find_gelpia(explicit=None):
    if explicit:
        return explicit
    env_val = os.environ.get("GELPIA")
    if env_val:
        return env_val
    gelpia_path = os.environ.get("GELPIA_PATH")
    if gelpia_path:
        return str(Path(gelpia_path) / "bin" / "gelpia")
    for candidate in (
        ROOT / "gelpia" / "bin" / "gelpia",
        ROOT / "FPTaylor" / "gelpia" / "bin" / "gelpia",
    ):
        if candidate.exists():
            return str(candidate)
    return shutil.which("gelpia")


def fptaylor_env():
    """
    env for running FPTaylor, with the active opam switch's bin dir put
    first on PATH. --opt bb compiles each query with ocamlfind/ocamlopt at
    runtime (FPTaylor/b_and_b/compile.sh); if PATH doesn't already resolve
    to the same OCaml that built FPTaylor/fptaylor, the compile silently
    uses whatever else is on PATH (e.g. a different opam switch, or a
    Homebrew OCaml), producing a .cmi incompatible with the already-built
    binary -- surfacing as "is not a compiled interface for this version
    of OCaml" on every subsequent --opt bb call, not just this one, since
    the mismatched .cmi is shared build state.
    """
    env = os.environ.copy()
    env.setdefault("FPTAYLOR_BASE", str(ROOT / "FPTaylor"))
    opam_bin = shutil.which("opam")
    if opam_bin:
        try:
            bin_dir = subprocess.run(
                [opam_bin, "var", "bin"], capture_output=True, text=True,
                timeout=5, check=True).stdout.strip()
            if bin_dir:
                path = env.get("PATH", "")
                if bin_dir not in path.split(os.pathsep):
                    env["PATH"] = bin_dir + os.pathsep + path
        except (subprocess.SubprocessError, OSError):
            pass
    return env


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------

def run_command(cmd, cwd=None, env=None):
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


# ---------------------------------------------------------------------------
# Verbosity
# ---------------------------------------------------------------------------
#   -v  (verbose >= 1): internal parameters derived per problem
#   -vv (verbose >= 2): full FPTaylor / CIRE / Gelpia output

def vprint(verbose, label, **values):
    """Level-1 trace of the internal parameters a template was built from."""
    if verbose < 1:
        return
    body = "  ".join(
        f"{k}={v:.12g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in values.items()
    )
    print(f"[{label}] {body}")


def elapsed_since(start):
    return time.perf_counter() - start


def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"


def hormann_proposal_deviation(delta_g, a, b):
    denom = 4.0 * a + b
    if denom <= 0.0:
        raise ValueError(f"Hormann proposal density bound needs 4*a+b > 0, got {denom}")
    return 2.0 * delta_g * (1.0 + 3.0 / denom)


def rou_proposal_deviation(delta_g, d8):
    if d8 <= 0.0:
        raise ValueError(f"RoU proposal bound needs d8 > 0, got {d8}")
    return 2.0 * delta_g * (1.0 + 1.0 / (2.0 * d8))


def acceptance_tv(delta_a):
    if delta_a < 0.0:
        raise ValueError(f"acceptance error must be non-negative, got {delta_a}")
    return 2.0 * math.expm1(delta_a)


# ---------------------------------------------------------------------------
# CIRE runner and output parser
# ---------------------------------------------------------------------------

def run_cire_llvm(cire_bin, c_code, func_name, domains, tag, inputs_dir, outputs_dir,
                  verbose=False, param_names=None):
    """
    Compile c_code to LLVM IR with clang -O1, then run CIRE_LLVM for func_name.

    domains     : list of (lo, hi) in C parameter order
    param_names : optional list of names matching domains order; defaults to arg_0, arg_1, …
    Returns     : (returncode, output_str)
    """
    import json

    names = param_names if param_names else [f"arg_{i}" for i in range(len(domains))]

    c_path    = inputs_dir  / f"cire_{tag}.c"
    ll_path   = inputs_dir  / f"cire_{tag}.ll"
    out_path  = outputs_dir / f"cire_{tag}_{func_name}.out"
    json_path = outputs_dir / f"cire_{tag}_{func_name}.json"

    c_path.write_text(c_code)

    clang = find_clang()
    cc_ret, cc_out = run_command(
        [clang, "-S", "-emit-llvm", "-O1", str(c_path), "-o", str(ll_path)]
    )
    if cc_ret != 0:
        raise RuntimeError(f"clang failed for {tag}:\n{cc_out}")

    domain_dict = {name: [lo, hi] for name, (lo, hi) in zip(names, domains)}
    json_path.write_text(json.dumps(domain_dict, indent=2))

    cmd = [cire_bin, str(ll_path), "--domain", str(json_path),
           "--function", func_name]
    ret, output = run_command(cmd)
    out_path.write_text(output)

    if verbose >= 2:
        print(f"--- CIRE {tag}/{func_name} ---\n{output}")
    return ret, output


def extract_cire_abs_error(output, label=""):
    m = CIRE_ABS_ERROR_RE.search(output)
    if not m:
        loc = f" ({label})" if label else ""
        raise RuntimeError(f"could not parse CIRE absolute error bound{loc}")
    return float(m.group(1))


# ---------------------------------------------------------------------------
# FPTaylor output parsers
# ---------------------------------------------------------------------------

def extract_abs_error(output, label):
    m = re.search(r"Absolute error [^:]*:\s*([-+\deE.]+)", output)
    if not m:
        raise RuntimeError(f"could not parse {label} FPTaylor absolute error")
    return float(m.group(1))


def extract_abs_errors_by_problem(output):
    errors = {}
    current = None
    for line in output.splitlines():
        pm = re.match(r"Problem:\s*(\S+)", line)
        if pm:
            current = pm.group(1)
            continue
        if current is None:
            continue
        em = re.match(r"Absolute error [^:]*:\s*([-+\deE.]+)", line)
        if em:
            errors[current] = float(em.group(1))
            current = None
    return errors



def extract_deltas_by_problem(output, label):
    """
    delta = abs_error / lower_bound — a sound upper bound on relative error:
      |fl(e)-e|/|e|  <=  max|fl(e)-e| / min|e|  =  abs_error / lower_bound

    FPTaylor's built-in --rel-error warns "close to zero" for small ranges, so
    we derive the relative error ourselves from the absolute error and bounds
    that FPTaylor always computes when --rel-error true is passed.
    """
    deltas = {}
    for section in re.split(r"(?=^-{10,})", output, flags=re.MULTILINE):
        m = re.search(r"Problem:\s*(\S+)", section)
        if not m:
            continue
        name = m.group(1)
        abs_m = ABS_ERROR_RE.search(section)
        lo_m = BOUNDS_LO_RE.search(section)
        if not abs_m:
            raise RuntimeError(f"{label}: could not parse absolute error for '{name}'")
        if not lo_m:
            raise RuntimeError(f"{label}: could not parse expression bounds for '{name}'")
        abs_error = float(abs_m.group(1))
        lower_bound = float(lo_m.group(1))
        if lower_bound <= 0:
            raise RuntimeError(
                f"{label} '{name}': expression lower bound non-positive ({lower_bound})"
            )
        deltas[name] = abs_error / lower_bound
    return deltas


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_mpl_dirs(plot_path):
    for d in (plot_path.parent / ".matplotlib", plot_path.parent / ".cache"):
        d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_path.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_path.parent / ".cache"))


def save_loglog_plot(
    xs, series, xlabel, ylabel, plot_path, plot_pgf=False, ylim_top=None
):
    """
    Generic log-log plot.  series = [(label, ys, marker), ...]
    Only finite, positive y-values are plotted.
    """
    _setup_mpl_dirs(plot_path)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4.5))
        for label, ys, marker in series:
            pts = [(x, y) for x, y in zip(xs, ys)
                   if math.isfinite(y) and y > 0]
            if not pts:
                continue
            sx, sy = zip(*pts)
            plt.loglog(sx, sy, marker=marker, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ylim_top is not None:
            plt.ylim(top=ylim_top)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        if plot_pgf:
            plt.savefig(plot_path.with_suffix(".pgf"), backend="pgf")
        plt.close()


# ---------------------------------------------------------------------------
# Common argparse additions (applied to every subparser)
# ---------------------------------------------------------------------------

def add_common_args(parser):
    parser.add_argument("--backend", choices=("fptaylor", "cire"), default="fptaylor",
                        help="FP analysis backend (default: fptaylor)")
    parser.add_argument("--fptaylor", default=None,
                        help="Path to FPTaylor executable")
    parser.add_argument("--cire", default=None,
                        help="Path to CIRE_LLVM executable")
    parser.add_argument("--fp", choices=("fp32", "fp64", "fp128"), default="fp64",
                        help="Floating-point format (default: fp64)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: <dist>_runs[_<stem>]/)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot TV vs distribution parameter")
    parser.add_argument("--plot-components", action="store_true",
                        help="Include individual error components in the plot")
    parser.add_argument("--plot-pgf", action="store_true",
                        help="Also save the plot in PGF format")
    parser.add_argument("--plot-file", type=Path, default=None,
                        help="Plot output path (default: <out-dir>/tv_vs_param.png)")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity: -v prints the internal "
                             "parameters derived per problem (interval "
                             "bounds, sampler constants), -vv also prints the "
                             "full FPTaylor/CIRE/Gelpia output to stdout")
    parser.add_argument("--cache", action="store_true",
                        help="If summary.csv already exists in out-dir, load it and skip re-running")
    parser.add_argument("--v-trunc", type=float, default=None,
                        help="Truncation floor for the raw uniform-seed "
                             "domain v (BTRS/PTRS: the log(v) domain lower "
                             "bound, replacing the hardcoded 2^-53). v can "
                             "realize any value in (0,1), including values "
                             "arbitrarily close to 0 where log(v) has no "
                             "finite rigorous bound, so log(v) is only "
                             "certified down to this floor; the flat amount "
                             "the reported TV is inflated by for the "
                             "region below it, e.g. --v-trunc 1e-8 adds "
                             "exactly 1e-8 to TV (default: fptaylor_settings"
                             ".toml's per-distribution value, else 2^-53).")
    parser.add_argument("--u-trunc", type=float, default=None,
                        help="BTRS/PTRS: minimum allowed us = "
                             "0.5 - |u| at the sampler's own reachable-k "
                             "boundary (us_lo, us_hi -- unlike v, u is "
                             "bounded away from the log singularity by the "
                             "discrete k=0/k=n edge, so this is a sanity "
                             "check, not a precision/TV tradeoff like "
                             "--v-trunc). If the boundary computed from "
                             "(n, p) ever dips below this, that is treated "
                             "as an error and raised rather than silently "
                             "absorbed into TV. Default: fptaylor_settings"
                             ".toml's per-distribution value, else 0.0 (no "
                             "restriction, never raises).")
    parser.add_argument("--bb-geometric-ratio-tol", type=float, default=2.0,
                        help="FPTaylor's --bb-geometric-ratio-tol: how far "
                             "(as a ratio) the branch-and-bound geometric "
                             "splitter narrows a sub-domain before stopping. "
                             "Must be > 1 (default: 2.0).")
    parser.add_argument("--bb-eval", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Use FPTaylor's interpreted --opt bb-eval "
                             "backend instead of the default compiling "
                             "--opt bb --bb-split midpoint. No compile "
                             "step, so queries can run concurrently (see "
                             "--jobs) and there is no risk of the bb compile "
                             "race, but bb-eval ignores --bb-split / "
                             "--bb-geometric-ratio-tol entirely -- pair with "
                             "--opt-x-abs-tol to control precision instead. "
                             "Default: fptaylor_settings.toml's "
                             "per-distribution value, else false (compiled "
                             "bb). See --approx: which of {bb, bb-eval} x "
                             "{approx, no-approx} is actually fastest is "
                             "distribution-specific and was found to differ "
                             "in both directions by sweep_fptaylor.py -- "
                             "that's why this is a per-distribution setting "
                             "now, not a single global default.")
    parser.add_argument("--approx", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Bound each per-operation error term "
                             "separately then sum (--opt-approx true "
                             "--opt-exact false), instead of FPTaylor's own "
                             "default of summing every term's expression "
                             "symbolically first and solving one much "
                             "harder joint optimization over the combined "
                             "expression (--opt-exact true). The joint form "
                             "can in principle find a tighter bound, but "
                             "has been observed to run 10+ minutes with no "
                             "progress on a query --approx solves in under "
                             "a second; both use the same rigorous "
                             "interval-arithmetic optimizer underneath, so "
                             "this is a tightness/speed tradeoff, not a "
                             "soundness one. Which way is faster is "
                             "distribution- (and backend-) specific: e.g. "
                             "confirmed via sweep_fptaylor.py that compiled "
                             "--opt bb triggers a separate OCaml compile per "
                             "term under --approx, making it ~12x slower "
                             "than --no-approx for binomial/poisson, while "
                             "for hypergeometric --no-approx's single joint "
                             "problem doesn't converge at all (900s+) and "
                             "--approx is the only viable choice. Default: "
                             "fptaylor_settings.toml's per-distribution "
                             "value, else true.")
    parser.add_argument("--opt-x-abs-tol", type=float, default=None,
                        help="FPTaylor's --opt-x-abs-tol: domain-size "
                             "termination tolerance for the optimizer, "
                             "applied to every variable not overridden by "
                             "--opt-x-abs-tol-vars (FPTaylor's own default "
                             "is 0.01). Passed through for both --opt bb "
                             "and --bb-eval -- with --bb-split midpoint "
                             "(the default), every declared variable is "
                             "checked against this width directly, not just "
                             "the ones FPTaylor's geometric splitter used to "
                             "exempt, so this now matters for the compiled "
                             "path too, not only --bb-eval.")
    parser.add_argument("--opt-x-abs-tol-vars", type=str, default=None,
                        help="FPTaylor's --opt-x-abs-tol-vars: per-variable "
                             "overrides of the domain-size termination "
                             "tolerance, format 'NAME1=VAL1,NAME2=VAL2,...'. "
                             "Variables not listed fall back to "
                             "--opt-x-abs-tol (or FPTaylor's own 0.01 "
                             "default if that isn't set either) -- so "
                             "covering the actual bottleneck variable(s) "
                             "matters more than covering the ones that "
                             "merely look wide (check each declared "
                             "variable's width in the .txt query directly "
                             "before guessing). Passed through for both "
                             "--opt bb and --bb-eval. Applies to both the "
                             "floor and accept queries unless overridden by "
                             "--floor-opt-x-abs-tol-vars / "
                             "--accept-opt-x-abs-tol-vars below.")
    parser.add_argument("--floor-opt-x-abs-tol-vars", type=str, default=None,
                        help="Overrides --opt-x-abs-tol-vars for floor "
                             "queries only (eps_floor). Floor's error is "
                             "often inherently unbounded near a sampler's "
                             "own truncation edge (e.g. us -> 0, where "
                             "division amplifies rounding error "
                             "quadratically in 1/us, unlike accept's log() "
                             "terms which only amplify it linearly) and "
                             "already gets charged to TV via --u-trunc "
                             "rather than needing a tight finite bound -- "
                             "so it's often fine to loosen this "
                             "independently of accept's setting.")
    parser.add_argument("--accept-opt-x-abs-tol-vars", type=str, default=None,
                        help="Overrides --opt-x-abs-tol-vars for accept "
                             "queries only (eps_accept). See "
                             "--floor-opt-x-abs-tol-vars.")


# ---------------------------------------------------------------------------
# fptaylor_settings.toml: per-distribution FPTaylor optimizer defaults,
# benchmarked by sweep_fptaylor.py (see that file for method and raw
# results: binomial/poisson are ~12x faster with --no-approx on the
# compiled --opt bb backend because --approx triggers a separate OCaml
# compile per per-operation error term there; hypergeometric is the
# opposite -- --no-approx's single joint optimization never converges
# (900s+) while --approx does, and --bb-eval edges out compiled --opt bb
# for that one). CLI flags always win over these; these only fill in
# values the user didn't pass explicitly (detected via the None-sentinel
# defaults on --approx/--bb-eval/--v-trunc/--u-trunc/etc above).
# ---------------------------------------------------------------------------

SETTINGS_TOML_PATH = ROOT / "fptaylor_settings.toml"

# Applied only if neither the CLI nor the TOML sets a value -- matches this
# module's original argparse defaults, so a distribution with no TOML
# section (or a missing TOML file entirely) behaves exactly as it did
# before this settings file existed.
_HARDCODED_DEFAULTS = {
    "approx": True,
    "bb_eval": False,
    "v_trunc": 2.0 ** -53,
    "u_trunc": 0.0,
}

_SETTINGS_CACHE = None


def load_settings_toml(path=None):
    """Load fptaylor_settings.toml once and cache it. Returns {} if the
    file doesn't exist -- it's an optional speed optimization, not a
    requirement (every field still has a hardcoded fallback below)."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is not None:
        return _SETTINGS_CACHE
    p = path or SETTINGS_TOML_PATH
    _SETTINGS_CACHE = {} if not p.exists() else tomllib.load(p.open("rb"))
    return _SETTINGS_CACHE


def apply_settings_defaults(args):
    """Fill in any of {approx, bb_eval, v_trunc, u_trunc, opt_x_abs_tol,
    opt_x_abs_tol_vars, floor_opt_x_abs_tol_vars, accept_opt_x_abs_tol_vars}
    the user left unset (None) on the command line, from
    fptaylor_settings.toml's [args.dist] section, then from
    _HARDCODED_DEFAULTS for the fields that must never reach a
    distribution module still None. Call once, right after
    argparse.parse_args(), before any distribution module reads these
    fields off args.

    Precedence: explicit CLI flag > fptaylor_settings.toml[dist] >
    hardcoded default.
    """
    settings = load_settings_toml()
    dist_settings = settings.get(args.dist, {})

    for field in ("approx", "bb_eval", "v_trunc", "u_trunc",
                  "opt_x_abs_tol", "opt_x_abs_tol_vars",
                  "floor_opt_x_abs_tol_vars", "accept_opt_x_abs_tol_vars"):
        if getattr(args, field, None) is None and field in dist_settings:
            setattr(args, field, dist_settings[field])

    for field, default in _HARDCODED_DEFAULTS.items():
        if getattr(args, field, None) is None:
            setattr(args, field, default)


def dist_switch(dist, default):
    """Regime-switch threshold for `dist` (e.g. binomial's n*p threshold,
    hypergeometric's sample-count threshold, geometric's p threshold) from
    fptaylor_settings.toml's [dist].switch, falling back to `default` if
    the file, the [dist] section, or the switch key is missing. Unlike
    apply_settings_defaults's fields, these aren't CLI args -- each
    distribution module reads this once at import time into its own
    module-level constant (_BTRS_SWITCH, _HRUA_SWITCH, _SWITCH, ...).
    """
    settings = load_settings_toml()
    return settings.get(dist, {}).get("switch", default)


# ---------------------------------------------------------------------------
# Shared FPTaylor helpers for Hormann-style transformed-rejection samplers
# (BTRS for binomial, PTRS for Poisson): native FPTaylor lgamma/log and
# standalone cached queries for -log(v) and -2*log(us).
# ---------------------------------------------------------------------------

def us_root(a, b, gamma):
    """
    Smallest us on one side of u = 0 for the Hormann floor form
    y(u) = (2*a/us + b)*u + c: the positive root of

        b*t^2 + beta*t - a = 0,      beta = 2*a - 0.5*b + gamma

    written cancellation-free as 2*a / (beta + sqrt(beta^2 + 4*a*b)), since
    beta ~ n swamps 4*a*b in the usual (-beta + sqrt(disc)) form.

    y is extremely steep near us = 0 (dy/dt = -+(a/t^2 + b)), so shave a
    relative 1e-12 off t: rounding in this solve can then only widen the
    enclosure, never clip a reachable u.  t is increasing in a and in b and
    decreasing in gamma, which is how box mode picks its corner.
    """
    beta = 2.0 * a - 0.5 * b + gamma
    t = 2.0 * a / (beta + math.sqrt(beta * beta + 4.0 * a * b))
    return min(t * (1.0 - 1e-12), 0.5)


def hormann_u_at(a, b, c, y):
    """
    The u with (2*a/us + b)*u + c = y, us = 0.5 - |u|.  y is strictly
    increasing in u (dy/du = a/us^2 + b > 0 on both sides of 0), so this
    inverse is unique and turns a constraint on k into one on u.
    """
    if y >= c:
        return 0.5 - us_root(a, b, y - c)
    return -(0.5 - us_root(a, b, c - y))


def hormann_k_defs(sign, setup_lines, c_expr):
    """
    k = floor(y) for one sign of u, as exact (unrounded) Definitions, shared
    by BTRS and PTRS.  setup_lines must define ax_, bx_ (and whatever they
    need) without rounding; c_expr is the exact shift.

    Two things are going on here.  k is written as y - f with f in [0, 1],
    which encodes floor exactly and keeps k tied to u instead of letting it
    roam over its window independently.  And y is rearranged: with
    u = s*(0.5 - us),

        (2*a/us + b)*u + c  =  s*(a/us - 2*a + 0.5*b - b*us) + c

    the same real number, but one that bounds tightly under interval
    arithmetic -- in the C form u and us = 0.5 - |u| appear as separate
    factors, so FPTaylor's conservative range for it spans zero and 1/(k+1)
    trips its division-by-zero check.

    Everything here is unrounded, and deliberately uses its own exact copies
    of the setup constants rather than the rounded ones the acceptance
    expression uses: k is one integer, computed once, and *both* samplers
    feed the same integer into the acceptance test.  Letting the rounding of
    us reach k instead propagates it with derivative a/us^2 ~ 1e6 into
    loggam(k+1), inflating eps_accept by three orders of magnitude and
    double-counting the floor disagreement eps_floor already bounds.
    """
    s = f"{float(sign):+.1f}"
    return list(setup_lines) + [
        f"  usx_   = 0.5 - abs(u),",
        f"  ys_    = {s} * (ax_ / usx_ - 2.0 * ax_ + 0.5 * bx_"
        f" - bx_ * usx_),",
        f"  yk_    = ys_ + {c_expr},",
        f"  k_     = yk_ - f,",
        f"  k1_    = k_ + 1.0,",
    ]

# Legacy coefficients used only by the interval_error helper below. The
# FPTaylor templates for the current rejection-sampler analysis use native
# FPTaylor lgamma/log through loggam_defs().
LOGGAM_A = [
     8.333333333333333e-02, -2.777777777777778e-03,
     7.936507936507937e-04, -5.952380952380952e-04,
     8.417508417508418e-04, -1.917526917526918e-03,
     6.410256410256410e-03, -2.955065359477124e-02,
     1.796443723688307e-01, -1.39243221690590e+00,
]
LOGGAM_LG2PI = 1.8378770664093453e+00


def loggam_defs(x_expr, prefix, rnd, backend="ieee"):
    """
    Return FPTaylor Definitions lines computing lgamma(x_expr). The analysis
    intentionally uses FPTaylor's native lgamma operator for the acceptance
    kernels, rounded under ULP_LEDGER[backend]["lgamma"]'s scale rather than
    FPTaylor's own default 0.5ulp assumption (see ulp_rnd_op) -- every
    lgamma call in every distribution's templates goes through this one
    function, so this is the single place that correction needs applying.

    prefix must be unique per call-site.  The last entry is the result name.
    `rnd` is still the plain rnd32/64/128 token (only its bit width is
    used); pass fp-consistent rnd here exactly as before.
    """
    name = f"{prefix}_gl"
    return [f"  {name} {ulp_rnd_op(rnd, 'lgamma', backend)}= lgamma({x_expr}),"], name


TRANSCENDENTAL_BACKENDS = ("ieee", "crlibm", "rlibm")

# Hardcoded fallback for each op, used only when fptaylor_settings.toml's
# [global.ulp_ledger] doesn't set that op -- log/lgamma default to 5 (the
# two ops this ledger is actually wired for so far), exp/sin/cos to 10
# (their old flat placeholder, now just routed through the same toml
# lookup instead of being a bare literal in the dict below).
_ULP_DEFAULTS = {"log": 5.0, "lgamma": 5.0, "exp": 10.0, "sin": 10.0, "cos": 10.0}

# PLACEHOLDER ledger: per-(backend, op) ULP bound k in the k * x * eps error
# model below.  Every op is overridable via fptaylor_settings.toml's
# [global.ulp_ledger], falling back to _ULP_DEFAULTS if that table or the
# op's own key is missing -- none of these numbers have been pulled from
# the actual accuracy guarantees each library documents yet:
#   - "ieee": no correct-rounding guarantee for transcendentals in IEEE-754
#     itself; treat as the conservative default until a real libm is named.
#   - "crlibm": correctly-rounded libm (<= 0.5ulp) for every function it
#     implements -- real entries should end up near 1, not 5-10.
#   - "rlibm": accuracy varies by function and by RLIBM variant (RLIBM-ALL,
#     RLIBM-FAST, ...); needs one entry per variant once real numbers land.
# Replace op-by-op as each library's documented bound is looked up; until
# then every backend behaves identically to this same flat placeholder.
_ulp_ledger_settings = load_settings_toml().get("global", {}).get("ulp_ledger", {})


def _ulp_default(op):
    return _ulp_ledger_settings.get(op, _ULP_DEFAULTS[op])


ULP_LEDGER = {
    backend: {op: _ulp_default(op) for op in _ULP_DEFAULTS}
    for backend in TRANSCENDENTAL_BACKENDS
}


def ulp_rnd_op(rnd, op, backend="ieee"):
    """
    FPTaylor rounding-operator token for one transcendental call ('name
    rnd[...]= log(x),' etc.), scaled by ULP_LEDGER[backend][op] instead of
    FPTaylor's own default 0.5ulp assumption.  `rnd` is the plain
    rnd32/64/128 token (only its bit width is used) already in scope at
    every call site that needs this (loggam_defs and the various
    *_setup_defs functions that isolate one log() call of their own).

    FPTaylor's general rounding operator rnd[bits, type, scale, eps, delta]
    models rnd(f) = f + f*e + d with |e| <= scale * 2^eps (type=ne)
    [FPTaylor/REFERENCE.md "Rounding"]; the simplified rnd[bits, type,
    scale] form (the one used here) auto-derives eps/delta from bits/type,
    so scale alone controls how many multiples of the default's 0.5ulp
    this call is allowed: plain rnd64 = rnd[64, ne] = rnd[64, ne, 1.0].
    Confirmed directly: rnd[64, ne, 5.0] on the same lgamma call reports
    exactly 5x the absolute error rnd64 does.

    This only widens the rounding model FPTaylor assumes for *this specific
    call* -- it must be the rounding operator on a Definition computing the
    op alone (`name rnd[...]= log(x),`), not one folded into a larger
    expression under another rounding operator, since a rounding operator
    is not applied inside another one [REFERENCE.md "Definitions": "rnd is
    not applied to another rounding operator and it does not propagate
    inside another rounding operator"].  A log/lgamma call embedded in a
    bigger rnd64-wrapped expression needs to be pulled out into its own
    Definition line first to get this treatment; see loggam_defs for a
    call site that already isolates it.
    """
    bits = int(rnd.removeprefix("rnd"))
    scale = ULP_LEDGER[backend][op]
    return f"rnd[{bits}, ne, {scale:.17g}]"


def transcendental_error_bound(op, x, fp, backend="ieee"):
    """
    Upper bound on the absolute error of one call to a non-elementary math
    function (log, exp, lgamma, ...) at argument x, standing in
    for whatever the actual backend library (CoreMath's correctly-rounded
    implementations, crlibm, RLIBM, the platform's standard libm, ...)
    guarantees for that op.  `backend` selects the row of ULP_LEDGER (see
    there); its entries are still placeholders, not real per-library figures.

    This is the same k*x*eps model ulp_rnd_op hands straight to FPTaylor as
    a scaled rounding operator (the actually-used mechanism, since FPTaylor
    then propagates it through its own Taylor-form analysis rather than
    Python having to separately track and add it after the fact). This
    standalone function is for a caller that needs the number itself
    outside an FPTaylor query (e.g. a non-FPTaylor analytic bound); it
    currently has no such caller.
    """
    eps = 2.0 ** -FP_BETA[fp]
    k = ULP_LEDGER[backend][op]
    return k * x * eps


def loggam_error_bound(x, fp, backend="ieee"):
    """Legacy lgamma case of transcendental_error_bound; see there."""
    return transcendental_error_bound("lgamma", x, fp, backend=backend)


def log_ie(x_lo, x_hi):
    """Enclosure of math.log(x) for x in [x_lo, x_hi], x_lo > 0: log is
    monotone increasing on (0, inf), so the endpoints bound it."""
    return math.log(x_lo), math.log(x_hi)


def exp_ie(x_lo, x_hi):
    """Enclosure of math.exp(x) for x in [x_lo, x_hi]: exp is monotone
    increasing everywhere, so the endpoints bound it."""
    return math.exp(x_lo), math.exp(x_hi)


_LGAMMA_ARGMIN = 1.4616321449683623  # where lgamma is smallest on (0, inf)


def lgamma_ie(x_lo, x_hi):
    """
    Enclosure (lo, hi) of the exact math.lgamma(x) for x in [x_lo, x_hi],
    x_lo >= 1. lgamma is convex on (0, inf) with a single minimum at
    _LGAMMA_ARGMIN, so it is monotone on each side of that point: the
    enclosure is the two endpoint values, plus the minimum itself if it
    falls inside [x_lo, x_hi].
    """
    lo = min(math.lgamma(x_lo), math.lgamma(x_hi))
    if x_lo < _LGAMMA_ARGMIN < x_hi:
        lo = min(lo, math.lgamma(_LGAMMA_ARGMIN))
    hi = max(math.lgamma(x_lo), math.lgamma(x_hi))
    return lo, hi


def loggam_ie(x, shift=0):
    """
    Legacy random_loggam(x) in (enclosure, error) interval arithmetic. The
    active FPTaylor rejection-sampler templates use native lgamma/log instead.

    `shift` reproduces random_loggam's argument reduction n = (int64_t)(7 - x)
    [random_poisson_ptrs.c:44]: shift == 0 is the Stirling branch, and
    shift > 0 evaluates Stirling at x0 = x + shift and then peels the shift
    factors back off with `gl -= log(x0 - 1)` [random_poisson_ptrs.c:59-61].
    That branch is what covers k + 1 < 7, which the Stirling-only FPTaylor
    template excludes from its domain.

    Operation order matches the C exactly, including the fact that the Horner
    step is a separate multiply and add (`gl0 *= x2; gl0 += a[k];`) rather than
    an fma, so both roundings are charged.
    """
    x0 = ie.add(x, ie.const(float(shift))) if shift else x
    inv = ie.div(ie.const(1.0), x0)
    x2 = ie.mul(inv, inv)

    gl0 = ie.const(LOGGAM_A[9])
    for k in range(8, -1, -1):
        gl0 = ie.add(ie.mul(gl0, x2), ie.const(LOGGAM_A[k]))

    # gl = gl0/x0 + 0.5*lg2pi + (x0 - 0.5)*log(x0) - x0, left to right.
    # 0.5*lg2pi is a scaling by a power of two, hence exact.
    gl = ie.add(ie.div(gl0, x0), ie.const(0.5 * LOGGAM_LG2PI))
    gl = ie.add(gl, ie.mul(ie.sub(x0, ie.const(0.5)), ie.ilog(x0)))
    gl = ie.sub(gl, x0)

    for _ in range(shift):
        x0m1 = ie.sub(x0, ie.const(1.0))
        gl = ie.sub(gl, ie.ilog(x0m1))
        x0 = x0m1
    return gl


# Floor and accept want opposite things from u's tolerance, which is why
# they get independently overridable values instead of sharing one:
#
#   floor: y_ = (2*a/us + b)*u + c is a handful of arithmetic ops -- cheap
#   to re-evaluate on every sub-box FPTaylor's branch-and-bound produces --
#   and eps_floor feeds directly into TV (2*eps_floor*accept_iter) with no
#   other query able to tighten it later, so it's worth spending a tight u
#   tolerance here: precision is what's needed, and it's affordable.
#
#   accept: the same u domain now drives a much deeper expression (lgamma,
#   log, division near us -> 0, and in box mode n/p/x/fm all threading
#   through it too), so splitting u as finely costs far more per split --
#   confirmed directly: individual accept sub-expressions can each still
#   show 50-97% suboptimality at a "tight" tolerance, meaning most of that
#   splitting effort buys little extra tightness anyway. Loosening u here
#   trades some of that (already partly wasted) tightness for runtime that
#   stays tractable as n, p (or lambda, N/K/n, ...) scale up -- scalability
#   is what's needed, at some cost in bound tightness.

def floor_x_abs_tol_vars(args):
    """--floor-opt-x-abs-tol-vars if set, else the shared --opt-x-abs-tol-vars."""
    return (args.floor_opt_x_abs_tol_vars if args.floor_opt_x_abs_tol_vars is not None
            else args.opt_x_abs_tol_vars)


def accept_x_abs_tol_vars(args):
    """--accept-opt-x-abs-tol-vars if set, else the shared --opt-x-abs-tol-vars."""
    return (args.accept_opt_x_abs_tol_vars if args.accept_opt_x_abs_tol_vars is not None
            else args.opt_x_abs_tol_vars)


def fptaylor_cmd(fptaylor, input_path, work_dir, ratio_tol=2.0, bb_eval=False,
                 x_abs_tol=None, x_abs_tol_vars=None, approx=True):
    """
    argv for one FPTaylor query, with its own temporary and log directories.

    Default: the compiling `bb` optimizer with midpoint branch-and-bound
    splitting, instead of Python-side binade shelling of u/v/k -- FPTaylor's
    own splitter handles the 1/us and log(us)-style conditioning that used
    to be pre-shelled.  Measured (not assumed): --bb-split geometric, despite
    existing for exactly this kind of near-singular conditioning, is both
    slower *and* looser than plain midpoint splitting on every query shape
    tested here (floor queries with us down to ~2e-4, and box-mode accept
    queries with a wide lgamma argument) -- typically ~2x tighter at equal
    iteration count, sometimes faster in wall time too.  --bb-geometric-
    ratio-tol is still passed and still validated by FPTaylor (> 1) even in
    midpoint mode, but no longer affects the actual splitting decision.

    `bb` writes its generated OCaml program to tmp-base-dir under fixed names
    -- tmp/bb_1.ml, tmp/bb [FPTaylor/default.cfg:141,
    FPTaylor/b_and_b/compile.sh] -- and runs it by a path relative to the
    working directory regardless of --tmp-base-dir, so concurrent bb queries
    race and clobber each other's compiled program (observed as hung
    processes, and in principle a bound reported for the wrong expression).
    Callers MUST run these serially (max_workers=1) when bb_eval=False; see
    dist_binomial._fptaylor_max.  `bb` has also been observed to emit OCaml
    that does not parse ("Syntax error" on a line of 81 closing parentheses)
    on some deeply-nested box-mode templates, after which FPTaylor dies with
    Not_found -- if that happens on a particular query, that query's template
    needs hand-splitting; there is no blanket workaround here.

    bb_eval=True switches to the interpreted `bb-eval` optimizer instead
    (--opt bb-eval): no compile step, so no compile race and safe to run
    concurrently, at the cost of the geometric splitter (bb-eval ignores
    --bb-split / --bb-geometric-ratio-tol entirely -- see
    FPTaylor/opt_bb_eval.ml, which calls Opt0.opt without them).

    x_abs_tol (--opt-x-abs-tol) and x_abs_tol_vars (--opt-x-abs-tol-vars,
    format "NAME1=VAL1,NAME2=VAL2") both tighten or loosen the domain-size
    termination tolerance, and both apply to *either* backend -- unlike the
    old bb-eval-only wiring, this matters for --opt bb too now that its
    default split mode is midpoint (see above): under midpoint,
    domain_small checks every declared variable's width against this
    tolerance directly (opt0.ml), not just the ones geometric mode used to
    exempt.  Named variables in x_abs_tol_vars use their own tolerance;
    anything not named falls back to x_abs_tol (or FPTaylor's own 0.01
    default if that is not set either).  Covering the wrong variable (one
    that already satisfies the default) does nothing -- check each
    variable's actual declared width in the .txt query before choosing what
    to loosen.

    Each query still gets its own temporary and log directory, so nothing in
    FPTaylor's scratch space is shared between runs.

    approx selects which of FPTaylor's two ways to turn the per-operation
    error terms into one final bound is used: approx=True (the default)
    passes --opt-approx true --opt-exact false, bounding each term
    separately then summing -- cheap per term, since each is a much
    simpler optimization than the alternative.  approx=False instead
    passes --opt-approx false --opt-exact true (FPTaylor's own upstream
    default): it sums every term's expression *symbolically first* into
    one combined expression and solves a single, much harder joint
    optimization over that -- "harder... but may yield a better result"
    (FPTaylor/default.cfg's own words), and in practice can run far longer
    with no visible progress (observed directly: 10+ minutes with zero
    output on a query approx solved in under a second). Both call the same
    rigorous interval-arithmetic optimizer underneath (compute_bound's
    Opt.find_max_abs vs Opt.find_max), so this is a tightness/speed choice,
    not a soundness one.
    """
    argv = [fptaylor, "--opt", "bb-eval"] if bb_eval else [
        fptaylor, "--opt", "bb",
        "--bb-split", "midpoint",
        "--bb-geometric-ratio-tol", f"{ratio_tol:.17g}"]
    argv += ["--opt-approx", "true" if approx else "false",
             "--opt-exact", "false" if approx else "true"]
    if x_abs_tol is not None:
        argv += ["--opt-x-abs-tol", f"{x_abs_tol:.17g}"]
    if x_abs_tol_vars:
        argv += ["--opt-x-abs-tol-vars", x_abs_tol_vars]
    return argv + [
            "--tmp-base-dir", str(work_dir / "tmp"),
            "--log-base-dir", str(work_dir / "log"),
            str(input_path)]


def run_fptaylor_query(fptaylor, input_path, outputs_dir, env, ratio_tol,
                       bb_eval=False, x_abs_tol=None, x_abs_tol_vars=None,
                       approx=True):
    """Run one FPTaylor query (a single input file, not a box list -- see
    dist_binomial._fptaylor_max for the box-list/max-reduction case) and
    return (code, output). Shared by PTRS and HRUA, which (unlike BTRS)
    never run more than one query per call site, so there is no compile
    race (dist_common.fptaylor_cmd) to guard against here even with the
    default --opt bb backend."""
    work = Path(tempfile.mkdtemp(prefix="fpt_", dir=outputs_dir))
    try:
        return run_command(
            fptaylor_cmd(fptaylor, input_path, work, ratio_tol, bb_eval,
                        x_abs_tol, x_abs_tol_vars, approx),
            cwd=ROOT, env=env)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def fp_var_type(fp):
    """FPTaylor variable type matching `fp`, for inputs that are already floats."""
    return {"fp32": "float32", "fp64": "float64", "fp128": "float128"}[fp]


_EXACT_DIGITS = 25   # far below FPTaylor's own error-bound precision (~1e-16)


def exact_bracket(x, digits=_EXACT_DIGITS):
    """
    (lo, hi) exact finite-decimal strings with lo <= x <= hi, rounded outward
    to `digits` significant digits.  FPTaylor reads a Variables literal as an
    exact rational (input_parser_env.ml:add_variable_with_uncertainty), so a
    ~20-significant-digit literal (round-trip-safe for re-parsing to the same
    double, but not x's true exact decimal expansion) would analyse a
    slightly different number than the actual fp64 value.  Rounding outward
    keeps it sound regardless (an integer collapses to lo == hi; a general
    float like p or lambda generally does not).
    """
    d = Decimal(x) if isinstance(x, int) else Decimal(float(x))
    if d == 0:
        return "0", "0"
    quant = Decimal(1).scaleb(d.adjusted() - digits + 1)
    lo = d.quantize(quant, rounding=ROUND_FLOOR)
    hi = d.quantize(quant, rounding=ROUND_CEILING)
    return format(lo, "f"), format(hi, "f")


def point_ivar(name, value, kind="real"):
    """One Variables-section line bracketing a single point value exactly;
    see exact_bracket.  FPTaylor collapses a degenerate real interval to an
    exact constant (add_variable_with_uncertainty), so callers can reference
    `name` afterward instead of re-embedding the literal at every occurrence."""
    lo, hi = exact_bracket(value)
    return f"  {kind} {name} in [{lo}, {hi}]"


def make_logv_template(fp, v_lo, v_hi=1.0):
    """
    Absolute error of -log(v), v in [v_lo, v_hi] — split out of the
    acceptance-test expression because folding it in adds v as an extra
    dimension to FPTaylor's joint branch-and-bound search and blows up
    the runtime of the whole eps_accept query.

    v is declared with a float type, not `real`.  That is not a tweak, it is
    what the sampler does: v is rk_double()'s return value, already a float,
    so no real -> float input conversion happens and none should be charged.
    Declaring it `real` makes FPTaylor insert that conversion and bound its
    size over the whole range at once -- 0.5*ulp(1) = 2^-53 absolute -- which
    -log then amplifies by 1/v.  At v = 2^-53 that is a factor of 2^53, so the
    bound collapses to ~1/2 (FPTaylor reports total2 = 0.5, absolute error
    0.508): vacuous.  With the float declaration the same query returns
    3.6e-15 over the full [2^-53, 1].  dist_poisson_stable.make_logv_template
    reached the same conclusion independently.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  {fp_var_type(fp)} v in [{v_lo:.20e}, {v_hi:.20e}];\n\n"
        "Definitions\n"
        f"  logv_step {ulp_rnd_op(rnd, 'log')}= - log(v);\n\n"
        "Expressions\n"
        "  eps_logv = logv_step;\n"
    )


_LOGV_EPS_CACHE = {}


def eps_logv(fptaylor, fp, vtail, inputs_dir, outputs_dir, env, verbose,
             ratio_tol=2.0, bb_eval=False, x_abs_tol=None, x_abs_tol_vars=None,
             approx=True):
    """
    Absolute error of -log(v) over v in [vtail, 1]; same for every distribution
    parameter, so cache it.  Returns (error, n_boxes).

    One query over the whole range: the 1/v conditioning near vtail is left to
    FPTaylor's own branch-and-bound splitting (--opt bb --bb-split midpoint,
    see fptaylor_cmd) instead of being pre-shelled by binade in Python.
    bb_eval/x_abs_tol/x_abs_tol_vars/approx are forwarded to fptaylor_cmd unchanged.
    """
    key = (fp, vtail, ratio_tol, bb_eval, x_abs_tol, x_abs_tol_vars, approx)
    if key in _LOGV_EPS_CACHE:
        return _LOGV_EPS_CACHE[key]

    stem = f"logv_{fp}_{vtail:.3e}"
    input_path = inputs_dir  / f"{stem}.txt"
    out_path   = outputs_dir / f"{stem}.out"
    input_path.write_text(make_logv_template(fp, vtail, 1.0))

    work = Path(tempfile.mkdtemp(prefix="fpt_", dir=outputs_dir))
    try:
        code, output = run_command(
            fptaylor_cmd(fptaylor, input_path, work, ratio_tol, bb_eval,
                        x_abs_tol, x_abs_tol_vars, approx),
            cwd=ROOT, env=env)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out_path.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor log(v) on [{vtail:.3e}, 1.0] ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor log(v) failed; see {out_path}")
    worst = extract_abs_errors_by_problem(output)["eps_logv"]

    result = (worst, 1)
    _LOGV_EPS_CACHE[key] = result
    return result


def make_logus_template(fp, utail, us_hi=0.5):
    """
    Absolute error of -2*log(us_), us_ in [utail, 0.5] — split out for the
    --fast path of accept-expression templates (see e.g.
    dist_binomial.make_btrs_accept_template).

    us_ is the free variable here, not u.  The sampler computes it as
    us = 0.5 - fabs(u) [btrs.c:60, random_poisson_ptrs.c:88], and on the
    reachable inputs that subtraction is *exact*: u = rk_double() - 0.5 is a
    multiple of 2^-53 with |u| <= 0.5, so 0.5 - |u| is another multiple of
    2^-53 no larger than 0.5, hence representable in 53 bits (and for
    |u| >= 0.25 it is Sterbenz-exact anyway).  The program's us_ therefore
    ranges over exactly the floats in [2^-53, 0.5], and taking it as a float
    input drops no error term.

    Keeping u as the variable instead costs the whole bound: u lives near
    0.5, so the real -> float input conversion FPTaylor charges on it is
    ~2^-54 absolute no matter how small us_ is, and -2*log amplifies that by
    2/us_ -- at us_ = 2^-53 the bound is 1.0, i.e. vacuous.  Reparametrised,
    the full range gives 1.4e-14.  See make_logv_template for the same effect
    on v, and why the declared type must be a float type rather than `real`.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  {fp_var_type(fp)} us_ in [{utail:.20e}, {us_hi:.20e}];\n\n"
        "Definitions\n"
        f"  logus_step {ulp_rnd_op(rnd, 'log')}= - 2.0 * log(us_);\n\n"
        "Expressions\n"
        "  eps_logus = logus_step;\n"
    )


_LOGUS_EPS_CACHE = {}


def eps_logus(fptaylor, fp, utail, inputs_dir, outputs_dir, env, verbose,
              ratio_tol=2.0, bb_eval=False, x_abs_tol=None, x_abs_tol_vars=None,
              approx=True):
    """
    Absolute error of -2*log(us_) over us_ in [utail, 0.5]; same for every
    distribution parameter, so cache it.  Returns (error, n_boxes).

    One query over the whole range: the 1/us conditioning near utail is left
    to FPTaylor's own branch-and-bound splitting (see eps_logv, fptaylor_cmd).
    bb_eval/x_abs_tol/x_abs_tol_vars/approx are forwarded to fptaylor_cmd unchanged.
    """
    key = (fp, utail, ratio_tol, bb_eval, x_abs_tol, x_abs_tol_vars, approx)
    if key in _LOGUS_EPS_CACHE:
        return _LOGUS_EPS_CACHE[key]

    stem = f"logus_{fp}_{utail:.3e}"
    input_path = inputs_dir  / f"{stem}.txt"
    out_path   = outputs_dir / f"{stem}.out"
    input_path.write_text(make_logus_template(fp, utail, 0.5))

    work = Path(tempfile.mkdtemp(prefix="fpt_", dir=outputs_dir))
    try:
        code, output = run_command(
            fptaylor_cmd(fptaylor, input_path, work, ratio_tol, bb_eval,
                        x_abs_tol, x_abs_tol_vars, approx),
            cwd=ROOT, env=env)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out_path.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor log(us) on [{utail:.3e}, 0.5] ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor log(us) failed; see {out_path}")
    worst = extract_abs_errors_by_problem(output)["eps_logus"]

    result = (worst, 1)
    _LOGUS_EPS_CACHE[key] = result
    return result

"""
Shared utilities for FP-error analysis scripts.
All distribution modules import from here.
"""
import contextlib
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

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
    env = os.environ.copy()
    env.setdefault("FPTAYLOR_BASE", str(ROOT / "FPTaylor"))
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


# ---------------------------------------------------------------------------
# Shared FPTaylor helpers for Hormann-style transformed-rejection samplers
# (BTRS for binomial, PTRS for Poisson): inline lgamma (Stirling, x>=7),
# and standalone cached queries for -log(v) and -2*log(us).
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

# Coefficients from random_loggam (numpy distributions.c)
LOGGAM_A = [
     8.333333333333333e-02, -2.777777777777778e-03,
     7.936507936507937e-04, -5.952380952380952e-04,
     8.417508417508418e-04, -1.917526917526918e-03,
     6.410256410256410e-03, -2.955065359477124e-02,
     1.796443723688307e-01, -1.39243221690590e+00,
]
LOGGAM_LG2PI = 1.8378770664093453e+00


def loggam_defs(x_expr, prefix, rnd):
    """
    Return FPTaylor Definitions lines implementing random_loggam(x_expr)
    for x_expr >= 7 (Stirling / asymptotic branch, straight-line).
    prefix must be unique per call-site.  The last entry is the result name.
    """
    a = LOGGAM_A
    lines = []
    # x2 = (1/x)*(1/x)
    lines.append(f"  {prefix}_x2 {rnd}= (1.0 / ({x_expr})) * (1.0 / ({x_expr})),")
    # Horner from a[9] down to a[0]
    lines.append(f"  {prefix}_h9 = {a[9]:.20e},")
    for i in range(8, -1, -1):
        lines.append(
            f"  {prefix}_h{i} {rnd}= {prefix}_h{i+1} * {prefix}_x2 + {a[i]:.20e},"
        )
    # gl = h0/x + 0.5*log(2pi) + (x-0.5)*log(x) - x
    lines.append(
        f"  {prefix}_gl {rnd}= {prefix}_h0 / ({x_expr})"
        f" + {0.5 * LOGGAM_LG2PI:.20e}"
        f" + (({x_expr}) - 0.5) * log({x_expr}) - ({x_expr}),"
    )
    return lines, f"{prefix}_gl"


def _fp_var_type(fp):
    """FPTaylor variable type matching `fp`, for inputs that are already floats."""
    return {"fp32": "float32", "fp64": "float64", "fp128": "float128"}[fp]


def make_logv_template(fp, vtail):
    """
    Absolute error of -log(v), v in [vtail, 1.0] — split out of the
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
        f"  {_fp_var_type(fp)} v in [{vtail:.20e}, 1.0];\n\n"
        "Definitions\n"
        f"  logv_step {rnd}= - log(v);\n\n"
        "Expressions\n"
        "  eps_logv = logv_step;\n"
    )


_LOGV_EPS_CACHE = {}


def eps_logv(fptaylor, fp, vtail, inputs_dir, outputs_dir, env, verbose):
    """Absolute error of -log(v); same for every distribution param, so cache it."""
    key = (fp, vtail)
    if key in _LOGV_EPS_CACHE:
        return _LOGV_EPS_CACHE[key]

    input_path = inputs_dir  / f"logv_{fp}_{vtail:.3e}.txt"
    out_path   = outputs_dir / f"logv_{fp}_{vtail:.3e}.out"
    input_path.write_text(make_logv_template(fp, vtail))

    code, output = run_command([fptaylor, str(input_path)], cwd=ROOT, env=env)
    out_path.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor log(v) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor log(v) failed; see {out_path}")

    result = extract_abs_errors_by_problem(output)["eps_logv"]
    _LOGV_EPS_CACHE[key] = result
    return result


def make_logus_template(fp, utail):
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
        f"  {_fp_var_type(fp)} us_ in [{utail:.20e}, 5.0e-01];\n\n"
        "Definitions\n"
        f"  logus_step {rnd}= - 2.0 * log(us_);\n\n"
        "Expressions\n"
        "  eps_logus = logus_step;\n"
    )


_LOGUS_EPS_CACHE = {}


def eps_logus(fptaylor, fp, utail, inputs_dir, outputs_dir, env, verbose):
    """Absolute error of -2*log(us_); same for every distribution param, so cache it."""
    key = (fp, utail)
    if key in _LOGUS_EPS_CACHE:
        return _LOGUS_EPS_CACHE[key]

    input_path = inputs_dir  / f"logus_{fp}_{utail:.3e}.txt"
    out_path   = outputs_dir / f"logus_{fp}_{utail:.3e}.out"
    input_path.write_text(make_logus_template(fp, utail))

    code, output = run_command([fptaylor, str(input_path)], cwd=ROOT, env=env)
    out_path.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor log(us) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor log(us) failed; see {out_path}")

    result = extract_abs_errors_by_problem(output)["eps_logus"]
    _LOGUS_EPS_CACHE[key] = result
    return result

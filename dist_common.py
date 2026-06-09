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

    if verbose:
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
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print FPTaylor/Gelpia output to stdout")
    parser.add_argument("--cache", action="store_true",
                        help="If summary.csv already exists in out-dir, load it and skip re-running")

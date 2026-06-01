"""
Geometric distribution FP-error analysis — stub.
Implement add_args / run / write_plot following the pattern in dist_binom.py.
"""
from pathlib import Path
from dist_common import ROOT

NAME = "geometric"
CSV_FIELDS = ["p", "delta", "tv"]


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
    raise NotImplementedError("geometric distribution analysis not yet implemented")


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    raise NotImplementedError

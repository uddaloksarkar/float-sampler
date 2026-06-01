"""
Zipf distribution FP-error analysis — stub.
Implement add_args / run / write_plot following the pattern in dist_binom.py.
"""
from pathlib import Path
from dist_common import ROOT

NAME = "zipf"
CSV_FIELDS = ["s", "n", "delta", "tv"]


def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with (s n) pairs, one per line")
    source.add_argument("--s", type=float, default=None,
                        help="Exponent parameter s > 1")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of elements")


def default_out_dir(args):
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "zipf_runs"
    return ROOT / f"zipf_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    raise NotImplementedError("Zipf distribution analysis not yet implemented")


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    raise NotImplementedError

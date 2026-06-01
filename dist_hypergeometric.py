"""
Hypergeometric distribution FP-error analysis — stub.
Implement add_args / run / write_plot following the pattern in dist_binom.py.
"""
from pathlib import Path
from dist_common import ROOT

NAME = "hypergeometric"
CSV_FIELDS = ["N", "K", "n", "delta", "tv"]


def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with (N K n) triples, one per line")
    source.add_argument("--N", type=int, default=None, dest="N_pop",
                        help="Population size")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of success states in population")
    parser.add_argument("--n", type=int, default=None, dest="n_draw",
                        help="Number of draws")


def default_out_dir(args):
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "hypergeometric_runs"
    return ROOT / f"hypergeometric_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    raise NotImplementedError("hypergeometric distribution analysis not yet implemented")


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    raise NotImplementedError

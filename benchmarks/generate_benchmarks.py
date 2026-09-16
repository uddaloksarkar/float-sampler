#!/usr/bin/env python3
"""
Generates the interval-mode benchmark suites in this directory
(binomial.tsv, poisson.tsv, hypergeometric.tsv): argument strings for
main.py aiming to cover the full range of parameter settings a caller could
pass, from small values up through and past FP64's exact-integer boundary
2^53 = 9007199254740992 (beyond it, not every integer is representable, so
box edges there are inherently approximate -- see this project's own
findings on that boundary).

Every row boxes ALL of that distribution's parameters at once (--lam-range;
--n-range AND --p-range; --N-range AND --K-range AND --n-range) -- no row
fixes one parameter to a point and sweeps only the others, since the goal is
TV bounds for arbitrary joint parameter settings, not single-axis slices.

Two different coverage strategies, by dimensionality:

  Poisson (1 free parameter, lambda): a genuine GAP-FREE geometric
  partition is tractable, so that's what this builds -- low regime
  (lambda < 30) chained at ratio 1.15 (a 14% relative width, safely under
  the ~20% ceiling this project measured by bisection at lambda~15), PTRS
  regime (lambda >= 30) chained at ratio 2.0 (doubling -- measured this
  session to never hit a width ceiling at any scale tested, up to 100%
  width). Every lambda in [LAMBDA_MIN, TOP] is covered by exactly one box.

  Binomial (n, p): GAP-FREE in both axes. n is chained edge-to-edge from 1
  to 2^53 at a uniform 10% step; p is chained edge-to-edge from 2^-53 to
  0.5 (p only needs (0, 0.5] -- see sampler_p's reflection) at the same 10%
  step. Every (n, p) row is one cell of the resulting 2D grid, kept only
  where the box's center satisfies n*p > 1 (a stricter, uniformly-safe
  width -- e.g. the ~0.5% ceiling measured at n~1e8 -- would make a
  gap-free 2D grid combinatorially unrunnable: two ~1D partitions of a few
  thousand steps each cross-multiply into the tens of millions). 10% is a
  documented, deliberately-chosen compromise between "actually gap-free"
  (the point of this rewrite -- an earlier draft used sparse power-of-2
  anchors with narrow local probes, which left large visible gaps, e.g.
  n=5,6,7 entirely uncovered between the n=4 and n=8 anchors) and a
  generatable/storable row count (~10^4-10^5, not 10^7+). Many cells at
  large n and/or small p are still expected to time out or produce vacuous
  bounds when actually run -- that is expected, not a flaw in the suite;
  gap-free coverage is a claim about which INPUTS have a box, not that
  every box's analysis will succeed quickly.

  Hypergeometric (N, K, n): still the dense-grid (not gap-free) approach
  below -- the same 2D-explosion argument applies even harder in 3D, and
  no equivalent width-ceiling data exists yet for its three shapes to even
  choose a principled uniform step from.

Regenerate with: python3 generate_benchmarks.py
"""
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

TOP_EXPONENT = 56          # a bit past 2^53's exact-integer boundary
EXACT_BOUNDARY = 2 ** 53


def pow2_anchors(lo_exp, hi_exp=TOP_EXPONENT):
    return [2 ** e for e in range(lo_exp, hi_exp + 1)]


def rng(center, w, integer=True, lo_floor=None):
    lo = center * (1 - w / 2)
    hi = center * (1 + w / 2)
    if lo_floor is not None:
        lo = max(lo, lo_floor)
    if integer:
        lo, hi = round(lo), round(hi)
        if lo == hi:
            hi = lo + 1
    return lo, hi


def sanitize(x):
    s = f"{x:.3e}" if isinstance(x, float) else str(x)
    return s.replace("+", "").replace(".", "p").replace("-", "m")


# ---------------------------------------------------------------------------
# Poisson: gap-free geometric partition of (LAMBDA_MIN, 2^TOP_EXPONENT].
# ---------------------------------------------------------------------------

LAMBDA_MIN = 1e-3     # practical floor -- covering to FP64's true denormal
                       # minimum (~5e-324) would need ~1000 more low-regime
                       # boxes for a range no real caller would pass
SWITCH = 30.0          # dist_poisson._SWITCH / analyticError.SWITCH


def poisson_rows():
    rows = []
    ratio_low, ratio_ptrs = 1.15, 2.0

    lam, i = LAMBDA_MIN, 0
    while lam < SWITCH:
        hi = min(lam * ratio_low, SWITCH)
        rows.append((f"p_low_{i:03d}", f"--lam-range {lam:.6g} {hi:.6g}"))
        lam, i = hi, i + 1

    lam, i = SWITCH, 0
    top = float(2 ** TOP_EXPONENT)
    while lam < top:
        hi = min(lam * ratio_ptrs, top)
        rows.append((f"p_ptrs_{i:03d}", f"--lam-range {lam:.6g} {hi:.6g}"))
        lam, i = hi, i + 1

    rows.append(("p_switch_straddle", "--lam-range 10 60"))
    return rows


N_MIN, N_MAX = 1, 2 ** 53
P_MIN, P_MAX = 2.0 ** -53, 0.5
GRID_RATIO = 1.10   # 10% step, chained edge-to-edge -- see module docstring


def chain_edges(lo, hi, ratio, integer=False):
    """[(e0,e1), (e1,e2), ...] covering [lo, hi] edge-to-edge at `ratio` --
    box i's hi and box i+1's lo are literally the same value, so rounding
    (integer=True) can't desync adjacent boxes: every edge is rounded once,
    into one shared sorted list, before pairing consecutive edges into
    boxes (rather than rounding each box's lo/hi independently, which would
    let a "bump up a degenerate zero-width box" fixup desync it from its
    neighbor's independently-rounded lo). A step that rounds to the same
    integer as its neighbor collapses away via the set() dedup -- that
    step was too fine to represent as its own integer box, not a gap.
    """
    edges = [float(lo)]
    x = float(lo)
    while x < hi:
        x = min(x * ratio, hi)
        edges.append(x)
    if integer:
        edges = sorted(set(max(1, round(e)) for e in edges))
    return list(zip(edges, edges[1:]))


def binomial_rows():
    rows = []
    rows.append(("b_switch_straddle", "--n-range 80 120 --p-range 0.28 0.32"))
    n_boxes = chain_edges(N_MIN, N_MAX, GRID_RATIO, integer=True)
    p_boxes = chain_edges(P_MIN, P_MAX, GRID_RATIO, integer=False)
    for i, (n_lo, n_hi) in enumerate(n_boxes):
        for j, (p_lo, p_hi) in enumerate(p_boxes):
            if math.sqrt(n_lo * n_hi) * math.sqrt(p_lo * p_hi) <= 1.0:
                continue   # degenerate mean -- skip, per n*p > 1 requirement
            rows.append((f"b_n{i:04d}_p{j:04d}",
                        f"--n-range {n_lo} {n_hi} --p-range {p_lo:.10g} {p_hi:.10g}"))
    return rows


# ---------------------------------------------------------------------------
# Hypergeometric: GAP-FREE in all three axes, mirroring binomial's approach
# but reparametrized to keep a 3-parameter grid tractable:
#   N          : the scale axis, chained edge-to-edge 1 .. 2^53 (like
#                binomial's n).
#   kappa=K/N  : a SHAPE axis, chained edge-to-edge 2^-53 .. 0.5. Only needs
#                (0, 0.5] because hrua_consts's mingoodbad = min(K, N-K) is
#                symmetric under K <-> N-K, the same reflection role p's
#                min(p, 1-p) plays for binomial (see dist_binomial.sampler_p
#                and hypergeometric_hrua.c's own d4 = mingoodbad/popsize).
#   nu=n/N     : a second SHAPE axis, same range and reflection reasoning
#                (hrua_consts's m = min(n, popsize-n)).
# K and n are then DERIVED per cell as interval products: K in
# [kappa_lo*N_lo, kappa_hi*N_hi], n in [nu_lo*N_lo, nu_hi*N_hi] -- both
# automatically stay within [0, N] since kappa, nu <= 0.5, so every
# generated triple is valid by construction (K <= N, n <= N).
#
# Going from 2 independent axes (binomial) to 3 turns a QUADRATIC cross
# product into a CUBIC one: binomial's own 10% step gives ~386 x ~379 =
# ~146K raw cells; the same 10% step here would give ~386 x ~379 x ~379 =
# ~55 MILLION raw cells before even filtering -- generatable in principle,
# but the resulting multi-GB file would take real time to write and be far
# beyond what any realistic run budget (even fully parallel) could get
# through, so "don't worry about row count" is read here as "don't
# artificially shrink it for tidiness," not "target a file too large to be
# a usable deliverable." GRID_RATIO_3D=1.7 (a 70% step) keeps the raw cross
# product around ~320K, the same order as binomial's ~72K post-filter --
# large, but one that generates instantly and stays a plausible thing to
# actually launch through run_interval_benchmarks.sh.
# ---------------------------------------------------------------------------

H_N_MIN, H_N_MAX = 1, 2 ** 53
H_FRAC_MIN, H_FRAC_MAX = 2.0 ** -53, 0.5   # kappa=K/N and nu=n/N range
GRID_RATIO_3D = 1.7


def hyper_rows():
    rows = []
    rows.append(("h_switch_straddle",
                "--N-range 950 1050 --K-range 280 320 --n-range 5 20"))

    N_boxes = chain_edges(H_N_MIN, H_N_MAX, GRID_RATIO_3D, integer=True)
    kappa_boxes = chain_edges(H_FRAC_MIN, H_FRAC_MAX, GRID_RATIO_3D)
    nu_boxes = chain_edges(H_FRAC_MIN, H_FRAC_MAX, GRID_RATIO_3D)

    for i, (N_lo, N_hi) in enumerate(N_boxes):
        for j, (k_lo, k_hi) in enumerate(kappa_boxes):
            K_lo, K_hi = max(1, round(k_lo * N_lo)), max(1, round(k_hi * N_hi))
            if K_lo > N_hi:
                continue
            for k, (n_lo, n_hi) in enumerate(nu_boxes):
                n_dlo, n_dhi = max(1, round(n_lo * N_lo)), max(1, round(n_hi * N_hi))
                if n_dlo > N_hi:
                    continue
                mean = (math.sqrt(N_lo * N_hi) * math.sqrt(k_lo * k_hi)
                        * math.sqrt(n_lo * n_hi))
                if mean <= 1.0:
                    continue   # degenerate mean -- skip, per n*K/N > 1
                rows.append((f"h_N{i:04d}_K{j:04d}_n{k:04d}",
                            f"--N-range {N_lo} {N_hi} --K-range {K_lo} {K_hi} "
                            f"--n-range {n_dlo} {n_dhi}"))
    return rows


def write_tsv(path, rows):
    seen = set()
    with open(path, "w") as f:
        f.write("# tag\targs  (generated by generate_benchmarks.py -- do not hand-edit)\n")
        for tag, args in rows:
            assert tag not in seen, f"duplicate tag {tag!r}"
            seen.add(tag)
            f.write(f"{tag}\t{args}\n")
    print(f"wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    write_tsv(HERE / "poisson.tsv", poisson_rows())
    write_tsv(HERE / "binomial.tsv", binomial_rows())
    write_tsv(HERE / "hypergeometric.tsv", hyper_rows())

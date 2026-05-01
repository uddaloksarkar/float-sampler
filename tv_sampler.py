"""
Empirical TV distance between a sampler and the true Poisson distribution.

Usage:
    python tv_sampler.py --lam 50 --n 100000
    python tv_sampler.py --lam 50 --n 100000 --plot
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def poisson_sampler(lam: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from Poisson(lam) using numpy (replace with custom sampler here)."""
    return rng.poisson(lam, size=n)


# ---------------------------------------------------------------------------
# TV distance
# ---------------------------------------------------------------------------

def tv_distance(samples: np.ndarray, lam: float) -> float:
    """
    Empirical TV distance between the sample histogram and Poisson(lam).

    TV(P, Q) = 0.5 * sum_k |P(k) - Q(k)|
    """
    lo = 0
    hi = max(int(samples.max()) + 1, int(lam + 10 * math.sqrt(lam)) + 1)
    support = np.arange(lo, hi)

    counts = np.bincount(samples, minlength=hi)[:hi]
    emp = counts / counts.sum()
    true = poisson.pmf(support, lam)

    return 0.5 * np.sum(np.abs(emp - true))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_histogram(samples: np.ndarray, lam: float):
    hi = max(int(samples.max()) + 1, int(lam + 8 * math.sqrt(lam)) + 1)
    support = np.arange(0, hi)

    counts = np.bincount(samples, minlength=hi)[:hi]
    emp  = counts / counts.sum()
    true = poisson.pmf(support, lam)

    tv = 0.5 * np.sum(np.abs(emp - true))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- left: histogram vs true pmf ---
    ax = axes[0]
    ax.bar(support, emp,  alpha=0.6, label='Sampler (empirical)', color='steelblue', width=0.8)
    ax.plot(support, true, 'r-o', markersize=3, linewidth=1.5, label=f'Poisson($\\lambda={lam}$)')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel('Probability')
    ax.set_title(f'Histogram vs Poisson PMF  ($n={len(samples):,}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- right: pointwise difference ---
    ax = axes[1]
    diff = emp - true
    colors = ['tomato' if d > 0 else 'steelblue' for d in diff]
    ax.bar(support, diff, color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\hat{p}(k) - p(k)$')
    ax.set_title(f'Pointwise difference  (TV = {tv:.4e})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam",  type=float, required=True,
                        help="Poisson parameter lambda")
    parser.add_argument("--n",    type=int,   default=100_000,
                        help="Number of samples (default: 100000)")
    parser.add_argument("--seed", type=int,   default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true",
                        help="Show histogram and difference plot")
    args = parser.parse_args()

    rng     = np.random.default_rng(args.seed)
    samples = poisson_sampler(args.lam, args.n, rng)
    tv      = tv_distance(samples, args.lam)

    print(f"lambda      = {args.lam}")
    print(f"n           = {args.n:,}")
    print(f"\\TV(sampler, Poisson) = {tv:.6e}")

    if args.plot:
        plot_histogram(samples, args.lam)

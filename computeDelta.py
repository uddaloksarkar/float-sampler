import argparse
import math


def _pyplot():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 10,
    })
    return plt

# Mantissa bits (total significand including implicit leading 1) per FP format
FP_BETA = {
    'fp32':  24,   # IEEE 754 binary32
    'fp64':  53,   # IEEE 754 binary64
    'fp128': 112,  # IEEE 754 binary128
    'fp256': 237,  # IEEE 754 binary256 (octuple)
}

COLORS = ['blue', 'green', 'red', 'orange']
SWITCH = 40 # lambda threshold for switching between low and high range computations

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_params(lam):
    gamma2 = 0.931 + 2.53 * math.sqrt(lam)
    gamma1 = -0.059 + 0.02483 * gamma2
    alpha  = 1.1239 + 1.1328 / (gamma2 - 3.4)
    return gamma1, gamma2, alpha

def delta_H(Kmax, epsH, gamma1):
        A1 = 5 / (24 * gamma1**2)
        A2 = 13 / (16 * gamma1**2)
        A3 = 7 / (16 * gamma1**2)
        A4 = 3 + 1 / (4 * gamma1**2)
        return A1 * Kmax**3 * epsH**2 + A2 * Kmax**2 * epsH**2 + A3 * Kmax * epsH + A4 * epsH

def _lr_error(lam, eps, dellog, Emax, alpha, gamma1, gamma2, delfactlog, epsiloninv):
    B0 = delfactlog(lam)
    B1 = math.log(lam) * (17*eps + dellog + eps*epsiloninv/2) + 2*Emax*eps + 3 *eps
    B2 = 16*eps + dellog + eps*epsiloninv/2
    B3 = (3*math.log(alpha) + Emax)*eps + Emax*dellog  #+ (3 * math.sqrt(3) / (16 * math.sqrt(gamma1 * gamma2)) + 27 / (1536 * gamma1 * gamma2)) * (16 *eps + dellog + eps*epsiloninv/2)
    return 2 / alpha * (B0 + B1 * lam + B2 * math.log(alpha) + 0.5 * B2 * math.log(2*math.pi*math.e*lam) + B3)

def delta_E(lam, eps, dellog, Emax, alpha, gamma1, gamma2, delfactlog, epsiloninv):
    B0 = delfactlog(lam)
    B1 = math.log(lam) * (17*eps + dellog + eps*epsiloninv/2) + 2*Emax*eps + 3 *eps
    B2 = 16*eps + dellog + eps*epsiloninv/2
    B3 = (3*math.log(alpha) + Emax)*eps + Emax*dellog + (3 * math.sqrt(3) / (16 * math.sqrt(gamma1 * gamma2)) + 27 / (1536 * gamma1 * gamma2)) * (16 *eps + dellog + eps*epsiloninv/2)
    B4 = 3 * math.sqrt(3) / (16 * math.sqrt(gamma1 * gamma2)) + 27 / (1536 * gamma1 * gamma2)
    DeltaE = _lr_error(lam, eps, dellog, Emax, alpha, gamma1, gamma2, delfactlog, epsiloninv) # + 2 * B4 / alpha
    return DeltaE, B4


def atomic_errors(lam, beta):
    """Return atomic floating-point error quantities for the high-range regime."""
    slack = 7 * math.sqrt(lam)
    Kmax  = lam + 9 * math.sqrt(lam) + 2
    Emax  = math.log2(Kmax)
    gamma1, gamma2, alpha = compute_params(lam)

    eps        = 2 ** (-beta)
    epslog     = 2 ** (-beta)
    dellog     = epslog * Emax * math.log(2)
    epsfact    = 10 ** (-10)
    delfactlog = lambda k: (2 * epsfact + 11 * eps + epslog
                            + 8 * k * eps + k * Emax * (9 * eps + epslog)
                            + Emax * (7 * eps + epslog) + dellog + eps)
    epsiloninv = (5 / 2) + (slack + 1) / gamma1
    epsH       = (14.5 + epsiloninv / 2) * eps

    return dict(
        slack=slack, Kmax=Kmax, Emax=Emax,
        gamma1=gamma1, gamma2=gamma2, alpha=alpha,
        eps=eps, dellog=dellog, delfactlog=delfactlog,
        epsiloninv=epsiloninv, epsH=epsH,
    )


def computeDeltaHighRange(lam, beta, verbose=False):
    ae = atomic_errors(lam, beta)
    slack, Kmax, Emax   = ae['slack'], ae['Kmax'], ae['Emax']
    gamma1, gamma2, alpha = ae['gamma1'], ae['gamma2'], ae['alpha']
    eps, dellog, delfactlog = ae['eps'], ae['dellog'], ae['delfactlog']
    epsiloninv, epsH    = ae['epsiloninv'], ae['epsH']

    DeltaH = delta_H(Kmax, epsH, gamma1)
    DeltaE, B4 = delta_E(lam, eps, dellog, Emax, alpha, gamma1, gamma2, delfactlog, epsiloninv)

    Deltahapperror = B4
    chernoff_term  = 2 * math.exp(-(slack**2) / (Kmax + lam))

    Delta = 2 * (DeltaH + DeltaE) * alpha #+ chernoff_term

    slam = math.sqrt(lam)
    b = 0.931 + 2.53 * slam
    vr = 0.9277 - 3.6224 / (b - 2)
    # assert 0 <= vr <= 1; print(f"Variance reduction factor vr={vr:.6f}")
    Delta = (1- 0.86*vr) * Delta

    if verbose:
        print(f"lambda={lam:.4g}  log2(lambda)={math.log2(lam):.3f}")
        print(f"  gamma1={gamma1:.6f}  gamma2={gamma2:.6f}  alpha={alpha:.6f}")
        print(f"  Kmax*epsH      = {Kmax*epsH:.3e}")
        print(f"  \\Delta_H        = {DeltaH:.6e}")
        print(f"  \\Delta_E - \\Delta_app   = {DeltaE - 2 * B4 / alpha:.6e}")
        print(f"  \\Delta_app      = {Deltahapperror:.6e}")
        print(f"  Chernoff        = {chernoff_term:.6e}")
        print(f"  \\Delta          = {Delta:.6e}")

    return Delta, DeltaH, DeltaE, Deltahapperror, chernoff_term


def computeDeltaLowRange(lam, beta):
    eps = 2 ** (-beta)
    return eps * lam / 2 + eps + eps


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _delta_sweep(beta):
    """Return (log2_lam_array, delta_array) sweeping lam from 2^0 to 2^beta."""
    import numpy as np

    beta_range = np.linspace(0, beta, 100)
    lam_values = 2 ** beta_range
    delta_values = [
        computeDeltaLowRange(lam, beta) if lam < SWITCH
        else computeDeltaHighRange(lam, beta)[0]
        for lam in lam_values
    ]
    return beta_range, delta_values


def _threshold_span(beta_range, delta_values):
    """Return (lo, hi) log2-lambda thresholds where Delta > 0.5, or (None, None)."""
    lo, hi, valley = None, None, False
    for x, d in zip(beta_range, delta_values):
        if lo is not None and d < 0.5:
            valley = True
        if d > 0.5 and hi is None and not valley:
            lo = x
        if d > 0.5 and hi is None and valley:
            hi = x
    return lo, hi


def plot_delta_vs_lambda(beta_values=None, save=None):
    plt = _pyplot()

    beta_values = beta_values or list(FP_BETA.values())
    plt.figure(figsize=(6, 3))
    for beta_val, color in zip(beta_values, COLORS):
        xs, ys = _delta_sweep(beta_val)
        plt.plot(xs, ys, label=f'Precision = {beta_val}', color=color)
    plt.axhline(y=0.5, color='black', linestyle='--',
                label=r'$\Delta = 0.5$')
    plt.xlabel(r'$\log_2(\lambda)$')
    plt.ylabel(r'$\Delta$')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.ylim(1e-10, 1e1)
    plt.tight_layout()
    if save:
        plt.savefig(save, format='pgf', bbox_inches='tight')
    plt.show()


def plot_components_vs_lambda(beta_values=None, save=None):
    import numpy as np

    plt = _pyplot()

    beta_values = beta_values or list(FP_BETA.values())
    n = len(beta_values)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.supylabel(r'Statistical Distance Bound $\Delta$', fontsize=16)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, beta_val in zip(axes, beta_values):
        beta_range = np.linspace(5, beta_val, 100)
        lam_values = 2 ** beta_range

        rows = [computeDeltaHighRange(lam, beta_val) for lam in lam_values if lam >= SWITCH] + \
                [(computeDeltaLowRange(lam, beta_val), 0, 0, 0, 0) for lam in lam_values if lam < SWITCH]
        delta_vals, dH, dE, dapp, chern = zip(*rows)

        lo, hi = _threshold_span(beta_range, delta_vals)

        ax.semilogy(beta_range, dH,    label=r'$\Delta_H$', linewidth=2)
        ax.semilogy(beta_range, dE,    label=r'$\Delta_E$', linewidth=2)
        ax.semilogy(beta_range, chern, label='Chernoff',    linewidth=2)
        ax.semilogy(beta_range, dapp,  label=r'$C_\mathrm{base}$', linewidth=2)
        ax.semilogy(beta_range, delta_vals, label=r'$\Delta$',
                    linewidth=2, linestyle='--')
        ax.set_xlabel(r'$\log_2(\lambda)$', fontsize=14)
        ax.set_title(f'Precision = {beta_val}', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3, linestyle='dotted')
        ax.set_ylim(top=1e0)
        if beta_val >= 112:
            ax.set_ylim(bottom=1e-30)
        ax.set_xlim(5, beta_val)
        if lo is not None and hi is not None:
            ax.axvspan(0,  lo, alpha=0.3, color='black',
                       label=r'$\Delta > 0.5$')
            ax.axvspan(hi, beta_val, alpha=0.3, color='black')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(save, format='pgf', bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loglam",  type=float, default=None,
                        help="log2 of lambda (lambda = 2^lam); required unless --plot/--save")
    parser.add_argument("--fp",   type=str, choices=FP_BETA.keys(), default=None,
                        help="Floating-point format (sets beta automatically)")
    parser.add_argument("--beta", type=int, default=None,
                        help="Mantissa precision bits (overrides --fp)")
    parser.add_argument("--plot", action="store_true",
                        help="Show summary and component plots across all FP formats")
    parser.add_argument("--save", action="store_true",
                        help="Save plots to PGF files (implies --plot)")
    args = parser.parse_args()

    if args.beta is not None:
        beta = args.beta
    elif args.fp is not None:
        beta = FP_BETA[args.fp]
    else:
        beta = list(FP_BETA.values())[-1]  # highest precision in FP_BETA

    if args.loglam is not None:
        lam = 2 ** args.loglam
        if lam < SWITCH:
            print(f"lambda={lam:.4g} is in the low range, using simplified Delta computation.")
            delta = computeDeltaLowRange(lam, beta)
            print(f"\\Delta = {delta:.6e}")
        else:
            computeDeltaHighRange(lam, beta, verbose=True)
    elif not (args.plot or args.save):
        parser.error("--loglam is required unless --plot or --save is specified")

    if args.plot or args.save:
        save_summary    = 'Delta_vs_Lambda.pgf'            if args.save else None
        save_components = 'DeltaH_DeltaE_Chernoff_vs_Lambda.pgf' if args.save else None
        plot_delta_vs_lambda(save=save_summary)
        plot_components_vs_lambda(save=save_components)

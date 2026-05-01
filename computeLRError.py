import argparse

from computeDelta import FP_BETA, atomic_errors, _lr_error

SWITCH = 30


def lr_error(lam, beta):
    """Return the _lr_error bound for the given lambda and precision beta."""
    if lam < SWITCH:
        raise ValueError(f"lambda={lam} is below the high-range threshold ({SWITCH}); _lr_error undefined here.")
    ae = atomic_errors(lam, beta)
    return _lr_error(
        lam,
        ae['eps'], ae['dellog'], ae['Emax'], ae['alpha'],
        ae['gamma1'], ae['gamma2'], ae['delfactlog'], ae['epsiloninv'],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the LR-error bound for Poisson sampling.")
    parser.add_argument("--lam", type=float, default=None,
                        required=True,
                        help="lambda")
    parser.add_argument("--fp",   type=str, choices=FP_BETA.keys(), default=None,
                        help="Floating-point format (sets beta automatically)")
    parser.add_argument("--beta", type=int, default=None,
                        help="Mantissa precision bits (overrides --fp)")
    args = parser.parse_args()

    if args.beta is not None:
        beta = args.beta
    elif args.fp is not None:
        beta = FP_BETA[args.fp]
    else:
        beta = list(FP_BETA.values())[-1]

    err = lr_error(args.lam, beta)
    print(f"lambda={args.lam:.6g}  beta={beta}  LR_error={err:.6e}")

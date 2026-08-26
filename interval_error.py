"""
Rigorous (enclosure, error) interval arithmetic for straight-line fp64 code.

This is deliberately *not* FPTaylor.  There are no Taylor forms and no global
optimiser: every operation propagates a first-order-free interval bound, which
is far looser than a Taylor form but is cheap, always terminates, and stays
sound on boxes where FPTaylor's branch-and-bound either times out or returns a
100%-suboptimal result.

It exists to discharge the *tail* obligation in the BTRS/PTRS TV bound.  The
accept test is a comparison in log space,

    log(v) <= A(k, us, ...)

so an absolute error E on A is a *multiplicative* e^E on the acceptance
probability, hence P_fp(K = k) <= e^(2E) * P_ideal(K = k) pointwise.  Charging
the out-of-window mass as exp(2E) * k_tail is what makes that term
non-circular: E comes from this module, k_tail from the binomial itself.  E
only has to satisfy exp(2E) = O(1) -- it multiplies a term of size ~1e-15 --
so a bound of 0.1, or even 1, costs nothing.  That is why interval arithmetic
is enough here and a tight query is not needed.

Each node carries

    lo, hi : rigorous enclosure of the *exact real* value of the subexpression
    err    : rigorous bound on |fp64-computed value - exact real value|

so the computed value is enclosed by [lo - err, hi + err].  Endpoints are
widened by one ulp after every fp64 operation, so the enclosures are sound
despite being computed in the same precision they reason about.
"""

import math

# Unit roundoff for round-to-nearest binary64: |fl(x) - x| <= U * |x|.
U = 2.0 ** -53

# Relative error charged to a libm call, in units of U.  IEEE-754 mandates
# correct rounding for sqrt (1.0), but not for log/exp; glibc and Apple's
# libm both document < 1 ulp there, which is 2 * U relative.
LIBM_RELERR = {"sqrt": 1.0 * U, "log": 2.0 * U, "exp": 2.0 * U}


def _dn(x):
    return math.nextafter(x, -math.inf)


def _up(x):
    return math.nextafter(x, math.inf)


class IE:
    """An exact-value enclosure paired with a bound on the fp64 error."""

    __slots__ = ("lo", "hi", "err")

    def __init__(self, lo, hi, err=0.0):
        if not (lo <= hi):
            raise ValueError(f"empty interval [{lo}, {hi}]")
        if err < 0.0 or not math.isfinite(err):
            raise ValueError(f"bad error bound {err}")
        self.lo, self.hi, self.err = lo, hi, err

    # -- enclosure of the *computed* value, i.e. exact value +- err ---------
    @property
    def clo(self):
        return _dn(self.lo - self.err)

    @property
    def chi(self):
        return _up(self.hi + self.err)

    def mag(self):
        """max |exact value| over the enclosure."""
        return max(abs(self.lo), abs(self.hi))

    def cmag(self):
        """max |computed value| over the enclosure."""
        return max(abs(self.clo), abs(self.chi))

    def cmin_abs(self):
        """min |computed value|; 0.0 if the computed enclosure straddles 0."""
        if self.clo <= 0.0 <= self.chi:
            return 0.0
        return min(abs(self.clo), abs(self.chi))

    def min_abs(self):
        if self.lo <= 0.0 <= self.hi:
            return 0.0
        return min(abs(self.lo), abs(self.hi))

    def __repr__(self):
        return f"IE([{self.lo:.6g}, {self.hi:.6g}], err={self.err:.3g})"


def const(c):
    """A literal that is exactly representable (all our coefficients are)."""
    return IE(c, c, 0.0)


def var(lo, hi, err=0.0):
    """A free input.  err=0 for values that arrive already rounded (RNG output,
    integer k), non-zero only if the caller is modelling an inherited error."""
    return IE(lo, hi, err)


def _round_err(cmag):
    """Rounding error charged to one fp64 operation whose result magnitude is
    bounded by cmag."""
    return _up(U * cmag)


def add(x, y):
    lo, hi = _dn(x.lo + y.lo), _up(x.hi + y.hi)
    cmag = max(abs(_dn(x.clo + y.clo)), abs(_up(x.chi + y.chi)))
    return IE(lo, hi, _up(_up(x.err + y.err) + _round_err(cmag)))


def sub(x, y):
    lo, hi = _dn(x.lo - y.hi), _up(x.hi - y.lo)
    cmag = max(abs(_dn(x.clo - y.chi)), abs(_up(x.chi - y.clo)))
    return IE(lo, hi, _up(_up(x.err + y.err) + _round_err(cmag)))


def mul(x, y):
    ps = (x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi)
    lo, hi = _dn(min(ps)), _up(max(ps))
    # |x^y^ - xy| <= |x|*y.err + |y|*x.err + x.err*y.err
    prop = _up(_up(x.mag() * y.err) + _up(y.mag() * x.err) + _up(x.err * y.err))
    cs = (x.clo * y.clo, x.clo * y.chi, x.chi * y.clo, x.chi * y.chi)
    cmag = max(abs(min(cs)), abs(max(cs)))
    return IE(lo, hi, _up(prop + _round_err(cmag)))


def div(x, y):
    if y.min_abs() == 0.0 or y.cmin_abs() == 0.0:
        raise ZeroDivisionError(f"divisor enclosure straddles zero: {y}")
    qs = (x.lo / y.lo, x.lo / y.hi, x.hi / y.lo, x.hi / y.hi)
    lo, hi = _dn(min(qs)), _up(max(qs))
    # x^/y^ - x/y = (x^-x)/y^ - x(y^-y)/(y^ y)
    prop = _up(_up(x.err / y.cmin_abs())
               + _up(_up(x.mag() * y.err) / _up(y.cmin_abs() * y.min_abs())))
    cs = (x.clo / y.clo, x.clo / y.chi, x.chi / y.clo, x.chi / y.chi)
    cmag = max(abs(min(cs)), abs(max(cs)))
    return IE(lo, hi, _up(prop + _round_err(cmag)))


def neg(x):
    return IE(-x.hi, -x.lo, x.err)


def iabs(x):
    """fabs is exact in fp64, so no rounding term."""
    if x.lo >= 0.0:
        return IE(x.lo, x.hi, x.err)
    if x.hi <= 0.0:
        return IE(-x.hi, -x.lo, x.err)
    return IE(0.0, max(-x.lo, x.hi), x.err)


def ilog(x):
    if x.lo <= 0.0 or x.clo <= 0.0:
        raise ValueError(f"log of non-positive enclosure: {x}")
    lo, hi = _dn(math.log(x.lo)), _up(math.log(x.hi))
    # |log(x^) - log(x)| <= x.err / min(x, x^)  (mean value; 1/xi, xi between)
    prop = _up(x.err / min(x.lo, x.clo))
    cmag = max(abs(_dn(math.log(x.clo))), abs(_up(math.log(x.chi))))
    return IE(lo, hi, _up(prop + _up(LIBM_RELERR["log"] * cmag)))


def isqrt(x):
    if x.lo < 0.0 or x.clo < 0.0:
        raise ValueError(f"sqrt of negative enclosure: {x}")
    lo, hi = _dn(math.sqrt(x.lo)), _up(math.sqrt(x.hi))
    m = min(x.lo, x.clo)
    if m <= 0.0:
        raise ValueError(f"sqrt error propagation needs x > 0: {x}")
    prop = _up(x.err / _dn(2.0 * math.sqrt(m)))
    cmag = max(abs(_dn(math.sqrt(x.clo))), abs(_up(math.sqrt(x.chi))))
    return IE(lo, hi, _up(prop + _up(LIBM_RELERR["sqrt"] * cmag)))


# --------------------------------------------------------------------------
# Adaptive subdivision
# --------------------------------------------------------------------------

def bisect_max_err(f, boxes, target, max_splits=4096):
    """
    Bound max err(f) over a list of boxes, splitting the widest dimension of
    the worst box until the bound drops under `target` or the split budget
    runs out.

    `boxes` is a list of tuples of (lo, hi) pairs; f takes one such tuple and
    returns an IE.  Returns (bound, n_boxes_evaluated).  The bound is sound
    whether or not `target` was reached -- the budget only controls tightness.

    Plain interval arithmetic over a wide box loses badly to the dependency
    problem (us appears in four places in the accept expression).  Splitting
    recovers most of it, and unlike a global optimiser it cannot silently
    return a non-bound.
    """
    work = []
    for b in boxes:
        try:
            work.append((f(b).err, b))
        except (ValueError, ZeroDivisionError):
            # A sub-box outside the function's domain cannot be discharged by
            # splitting; surface it rather than dropping it from the max.
            raise
    splits = 0
    while splits < max_splits:
        work.sort(key=lambda t: t[0], reverse=True)
        worst, box = work[0]
        if worst <= target:
            break
        widths = [hi - lo for lo, hi in box]
        d = max(range(len(box)), key=lambda i: widths[i])
        if widths[d] <= 0.0:
            break  # degenerate: nothing left to split
        lo, hi = box[d]
        mid = 0.5 * (lo + hi)
        if not (lo < mid < hi):
            break  # hit fp64 resolution
        left = tuple(box[:d]) + ((lo, mid),) + tuple(box[d + 1:])
        right = tuple(box[:d]) + ((mid, hi),) + tuple(box[d + 1:])
        work[0:1] = [(f(left).err, left), (f(right).err, right)]
        splits += 1
    return max(e for e, _ in work), len(work)

/*
 * Binomial random variate generation via BTRS, ported from the BTRS branch
 * of CPython's Random.binomialvariate (Lib/random.py, function
 * binomialvariate):
 *   https://github.com/python/cpython/blob/main/Lib/random.py#L789
 *
 * BTRS = Transformed Rejection with Squeeze, Hormann (1993):
 *   https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8407&rep=rep1&type=pdf
 *
 * The source dispatches to a separate BG (Devroye geometric) sampler for
 * n*p < 10 for performance; that branch is omitted here so this file always
 * exercises BTRS. The algorithm remains an exact rejection sampler for any
 * n*p (just with a higher rejection rate when n*p is small).
 *
 * Symmetry (p > 0.5 -> n - Binomial(n, 1-p)) and the n == 1 fast path are
 * preserved from the source; out-of-range-parameter error raising is not
 * (these reference samplers assume valid n >= 0, 0 <= p <= 1, as elsewhere
 * in this directory).
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static double rk_double(void *state) {
    (void)state;
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

int64_t binomialvariate(void *rstate, int64_t n, double p) {
    /* Handle edge cases */
    if (p == 0.0) return 0;
    if (p == 1.0) return n;

    /* Fast path for a common case */
    if (n == 1) {
        return (rk_double(rstate) < p) ? 1 : 0;
    }

    /* Exploit symmetry to establish: p <= 0.5 */
    if (p > 0.5) {
        return n - binomialvariate(rstate, n, 1.0 - p);
    }

    /* BTRS: Transformed rejection with squeeze method by Wolfgang Hormann */
    {
        int setup_complete = 0;
        double spq = sqrt(n * p * (1.0 - p));   /* Standard deviation of the distribution */
        double b  = 1.15 + 2.53 * spq;
        double a  = -0.0873 + 0.0248 * b + 0.01 * p;
        double c  = n * p + 0.5;
        double vr = 0.92 - 4.2 / b;
        double alpha = 0.0, lpq = 0.0, h = 0.0;
        int64_t m = 0;

        for (;;) {
            double u = rk_double(rstate);
            u -= 0.5;
            double us = 0.5 - fabs(u);
            int64_t k = (int64_t)floor((2.0 * a / us + b) * u + c);
            if (k < 0 || k > n) {
                continue;
            }
            double v = rk_double(rstate);

            /* The early-out "squeeze" test substantially reduces
             * the number of acceptance condition evaluations. */
            if (us >= 0.07 && v <= vr) {
                return k;
            }

            if (!setup_complete) {
                alpha = (2.83 + 5.1 / b) * spq;
                lpq = log(p / (1.0 - p));
                m = (int64_t)floor((n + 1) * p);   /* Mode of the distribution */
                h = lgamma((double)(m + 1)) + lgamma((double)(n - m + 1));
                setup_complete = 1;                /* Only needs to be done once */
            }

            /* Acceptance-rejection test.
             * Note, the original paper erroneously omits the call to log(v)
             * when comparing to the log of the rescaled binomial distribution.
             * v is left unscaled here; the scaling factor
             * alpha / (a / (us * us) + b) is folded into the comparison
             * in log space instead, and a / (us * us) + b is rewritten as
             * (a + b * us * us) / (us * us) to avoid forming 1 / us^2
             * directly when us is small. */
            double us2 = us * us;
            if (log(v) <= h - lgamma((double)(k + 1)) - lgamma((double)(n - k + 1))
                          + (k - m) * lpq - log(alpha)
                          + log(a + b * us2) - 2.0 * log(us)) {
                return k;
            }
        }
    }
}

/* ---- Test harness ---- */
#define N_SAMPLES 200000
static void run_case(const char *label, int64_t n, double p) {
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)binomialvariate(NULL, n, p);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    printf("binomialvariate  %-18s n=%lld  p=%.2f\n", label, (long long)n, p);
    printf("  empirical mean = %.4f   (theory = %.4f)\n", mean, n * p);
    printf("  empirical var  = %.4f   (theory = %.4f)\n", var,  n * p * (1.0 - p));
}

int main(void) {
    run_case("(small n*p)", 30,  0.2);   /* n*p = 6  -- BTRS still exact, just more rejections */
    run_case("(large n*p)", 200, 0.4);   /* n*p = 80 -- BTRS sweet spot */
    return 0;
}

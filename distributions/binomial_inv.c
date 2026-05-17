/*
 * Binomial random variate generation via inversion (sequential CDF search).
 *
 * Extracted from:
 *   numpy/random/src/legacy/legacy-distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *
 * Used when n*p < 30 (small expected count).
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static double rk_double(void *state) {
    (void)state;
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

int64_t binomial_inversion(void *rstate, int64_t n, double p) {
    double q, qn, np, p0, pu, U;
    int64_t y, bound;

    q     = 1.0 - p;
    qn    = exp(n * log(q));          /* P(X = 0) = (1-p)^n */
    np    = n * p;
    bound = (int64_t)fmin((double)n, np + 10.0 * sqrt(np * q + 1.0));

    p0 = qn;
    pu = qn;
    U  = rk_double(rstate);
    y  = 0;

    /* Walk PMF: P(X=k+1) = P(X=k) * (n-k)/(k+1) * p/q */
    while (U > pu && y < bound) {
        y++;
        p0 *= (double)(n - y + 1) * p / ((double)y * q);
        pu += p0;
    }
    return y;
}

/* ---- Test harness ---- */
#define N_SAMPLES 200000
int main(void) {
    const int64_t n = 50;
    const double  p = 0.3;   /* n*p = 15 < 30: inversion regime */
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)binomial_inversion(NULL, n, p);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    printf("binomial_inversion  n=%lld  p=%.2f\n", (long long)n, p);
    printf("  empirical mean = %.4f   (theory = %.4f)\n", mean, n * p);
    printf("  empirical var  = %.4f   (theory = %.4f)\n", var,  n * p * (1.0 - p));
    return 0;
}
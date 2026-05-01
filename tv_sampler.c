/*
 * tv_sampler.c
 *
 * C port of tv_sampler.py:
 *   - Draws n samples from Poisson(lambda) using the PTRS algorithm
 *   - Computes empirical TV distance against the true Poisson PMF
 *
 * PTRS sampler ported from:
 *   numpy/random/src/distributions/distributions.c  (commit 6059403)
 *
 * Build:
 *   gcc -O2 -o tv_sampler tv_sampler.c -lm
 *
 * Usage:
 *   ./tv_sampler --lam 50 --n 100000
 *   ./tv_sampler --lam 50 --n 100000 --seed 42
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* =========================================================================
 * Xoshiro256** PRNG
 * ========================================================================= */

typedef struct { uint64_t s[4]; } bitgen_t;

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro256ss_next(bitgen_t *bg) {
    uint64_t result = rotl64(bg->s[1] * 5, 7) * 9;
    uint64_t t      = bg->s[1] << 17;
    bg->s[2] ^= bg->s[0];
    bg->s[3] ^= bg->s[1];
    bg->s[1] ^= bg->s[2];
    bg->s[0] ^= bg->s[3];
    bg->s[2] ^= t;
    bg->s[3]  = rotl64(bg->s[3], 45);
    return result;
}

/* Uniform double in [0, 1) using 53 random bits. */
static double next_double(bitgen_t *bg) {
    return (double)(xoshiro256ss_next(bg) >> 11)
           * (1.0 / (double)(UINT64_C(1) << 53));
}

/* SplitMix64 seeding — guarantees all four state words are non-zero. */
static void bitgen_seed(bitgen_t *bg, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        bg->s[i] = z ^ (z >> 31);
    }
}

/* =========================================================================
 * Log-gamma
 * Ported verbatim from numpy/random/src/distributions/distributions.c
 * ========================================================================= */

double random_loggam(double x) {
    double x0, x2, lg2pi, gl, gl0;
    long long k, n;

    static double a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                           7.936507936507937e-04, -5.952380952380952e-04,
                           8.417508417508418e-04, -1.917526917526918e-03,
                           6.410256410256410e-03, -2.955065359477124e-02,
                           1.796443723688307e-01, -1.39243221690590e+00};

    if ((x == 1.0) || (x == 2.0)) {
        return 0.0;
    } else if (x < 7.0) {
        n = (long long)(7 - x);
    } else {
        n = 0;
    }
    x0 = x + n;
    x2 = (1.0 / x0) * (1.0 / x0);
    lg2pi = 1.8378770664093453e+00;
    gl0 = a[9];
    for (k = 8; k >= 0; k--) {
        gl0 *= x2;
        gl0 += a[k];
    }
    gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * log(x0) - x0;
    if (x < 7.0) {
        for (k = 1; k <= n; k++) {
            gl -= log(x0 - 1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

/* =========================================================================
 * PTRS Poisson sampler
 * Ported verbatim from numpy/random/src/distributions/distributions.c
 * ========================================================================= */

static long long random_poisson_ptrs(bitgen_t *bg, double lam) {
    long long k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;

    slam     = sqrt(lam);
    loglam   = log(lam);
    b        = 0.931 + 2.53 * slam;
    a        = -0.059 + 0.02483 * b;
    invalpha = 1.1239 + 1.1328 / (b - 3.4);
    vr       = 0.9277 - 3.6224 / (b - 2.0);

    for (;;) {
        U  = next_double(bg) - 0.5;
        V  = next_double(bg);
        us = 0.5 - fabs(U);
        k  = (long long)floor((2.0 * a / us + b) * U + lam + 0.43);
        if (us >= 0.07 && V <= vr) return k;
        if (k < 0 || (us < 0.013 && V > us)) continue;
        if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
            (-lam + (double)k * loglam - random_loggam((double)k + 1)))
            return k;
    }
}

/* =========================================================================
 * True Poisson PMF  P(k; lam) = exp(-lam + k*log(lam) - lgamma(k+1))
 * ========================================================================= */

static double poisson_pmf(long long k, double lam) {
    if (k < 0) return 0.0;
    return exp(-lam + (double)k * log(lam) - random_loggam((double)k + 1));
}

/* =========================================================================
 * Empirical TV distance
 *
 * TV(P, Q) = 0.5 * sum_k |P(k) - Q(k)|
 *
 * Support: [0, hi)  where  hi = max(max_sample+1, lam + 10*sqrt(lam) + 1)
 * ========================================================================= */

static double tv_distance(const long long *samples, long n, double lam) {
    long long max_k = 0;
    for (long i = 0; i < n; i++)
        if (samples[i] > max_k) max_k = samples[i];

    long long hi_ll = (long long)(lam + 10.0 * sqrt(lam)) + 1;
    if (max_k + 1 > hi_ll) hi_ll = max_k + 1;
    long long hi = hi_ll;

    long long *counts = (long long *)calloc((size_t)hi, sizeof(long long));
    if (!counts) { perror("calloc"); exit(1); }
    for (long i = 0; i < n; i++) counts[samples[i]]++;

    double tv = 0.0;
    for (long long k = 0; k < hi; k++) {
        double emp  = (double)counts[k] / (double)n;
        double true_pmf = poisson_pmf(k, lam);
        tv += fabs(emp - true_pmf);
    }
    free(counts);
    return 0.5 * tv;
}

/* =========================================================================
 * CLI
 * ========================================================================= */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --lam <x> [--n <count>] [--seed <s>]\n"
        "\n"
        "  --lam   <x>   Poisson parameter lambda (required)\n"
        "  --n     <n>   number of samples (default: 100000)\n"
        "  --seed  <s>   RNG seed (default: time-based)\n",
        prog);
}

int main(int argc, char **argv) {
    double   lam   = -1.0;
    long     n     = 100000;
    uint64_t seed  = (uint64_t)time(NULL);
    int      got_seed = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--lam")  && i+1 < argc) lam  = atof(argv[++i]);
        else if (!strcmp(argv[i], "--n")    && i+1 < argc) n    = atol(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) { seed = (uint64_t)atoll(argv[++i]); got_seed = 1; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    (void)got_seed;

    if (lam <= 0.0) { fprintf(stderr, "--lam is required and must be > 0\n"); usage(argv[0]); return 1; }
    if (n   <= 0)   { fprintf(stderr, "--n must be > 0\n"); return 1; }

    bitgen_t bg;
    bitgen_seed(&bg, seed);

    long long *samples = (long long *)malloc((size_t)n * sizeof(long long));
    if (!samples) { perror("malloc"); return 1; }

    for (long i = 0; i < n; i++)
        samples[i] = random_poisson_ptrs(&bg, lam);

    double tv = tv_distance(samples, n, lam);

    /* sample mean and variance */
    double sum = 0.0, sum2 = 0.0;
    for (long i = 0; i < n; i++) {
        sum  += (double)samples[i];
        sum2 += (double)samples[i] * (double)samples[i];
    }
    double mean = sum  / (double)n;
    double var  = sum2 / (double)n - mean * mean;

    printf("lambda      = %g\n",   lam);
    printf("n           = %ld\n",  n);
    printf("seed        = %llu\n", (unsigned long long)seed);
    printf("sample mean = %.6f  (expected %g)\n", mean, lam);
    printf("sample var  = %.6f  (expected %g)\n", var,  lam);
    printf("TV(sampler, Poisson) = %.6e\n", tv);

    free(samples);
    return 0;
}

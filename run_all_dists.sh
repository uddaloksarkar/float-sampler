#!/bin/bash
# Point-mode FPTaylor runs for the distribution frontends.
# Binomial interval mode lives in run_box_dists.sh.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ---- shared flags ----------------------------------------------------------
FP="fp64"
OPT_X_ABS_TOL="0.01"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 1)"
VERBOSITY="-v"                    # "-v", "-vv", or "" for none

COMMON_ARGS=(--fp "$FP" --opt-x-abs-tol "$OPT_X_ABS_TOL")
[ -n "$VERBOSITY" ] && COMMON_ARGS+=("$VERBOSITY")

# # ---- binomial (BTRS) -------------------------------------------------------
# N="10900"
# P="0.1"
# N_TOL="1e5"
# P_TOL="0.1"
# V_TRUNC="1e-20"
# U_TRUNC="1e-5"
# U_TOL_FLOOR="1e-5"
# U_TOL_ACCEPT="0.1"

# BINOMIAL_ARGS=(
#   "${COMMON_ARGS[@]}"
#   --floor-opt-x-abs-tol-vars "u=${U_TOL_FLOOR},n=${N_TOL},p=${P_TOL}"
#   --accept-opt-x-abs-tol-vars "u=${U_TOL_ACCEPT},n=${N_TOL},p=${P_TOL},f=1,fm=1"
#   --v-trunc "$V_TRUNC"
#   --u-trunc "$U_TRUNC"
# )

# python3 main.py "${BINOMIAL_ARGS[@]}" binomial \
#   --n "$N" --p "$P" --jobs "$JOBS"

# # ---- poisson (PTRS) --------------------------------------------------------
# LAM="40"
# POISSON_V_TRUNC="1e-20"
# POISSON_U_TRUNC="1e-3"
# POISSON_OPT_X_ABS_TOL="0.01"

# POISSON_ARGS=(
#   --fp "$FP" --opt-x-abs-tol "$POISSON_OPT_X_ABS_TOL"
#   --v-trunc "$POISSON_V_TRUNC" --u-trunc "$POISSON_U_TRUNC"
# )
# [ -n "$VERBOSITY" ] && POISSON_ARGS+=("$VERBOSITY")

# python3 main.py "${POISSON_ARGS[@]}" poisson --lam "$LAM"

# ---- hypergeometric (HRUA) -------------------------------------------------
# W's declared range (hrua_z_range) scales with N/K/n, same problem as
# binomial's u/n/p/x -- a flat --opt-x-abs-tol alone doesn't scale, so W
# gets its own per-variable override; X and f are always ~[0,1]-scale and
# don't need one.
HYPER_N="100"
HYPER_K="40"
HYPER_n="30"
HYPER_OPT_X_ABS_TOL="0.01"
HYPER_W_TOL="1e3"    # raise alongside HYPER_N (roughly N-scale)

HYPER_ARGS=(
  --fp "$FP" --opt-x-abs-tol "$HYPER_OPT_X_ABS_TOL"
  --opt-x-abs-tol-vars "W=${HYPER_W_TOL}"
)
[ -n "$VERBOSITY" ] && HYPER_ARGS+=("$VERBOSITY")

python3 main.py "${HYPER_ARGS[@]}" hypergeometric \
  --N "$HYPER_N" --K "$HYPER_K" --n "$HYPER_n"

# ---- zipf ------------------------------------------------------------------
ZIPF_S="2.5"

python3 main.py "${COMMON_ARGS[@]}" zipf --s "$ZIPF_S"

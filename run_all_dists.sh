#!/bin/bash
# One-click point-mode FPTaylor runs for the distribution frontends.
# Binomial interval mode lives in run_box_dists.sh.
#
# All FPTaylor optimizer tuning (--approx/--no-approx, --bb-eval,
# --v-trunc/--u-trunc, per-variable --opt-x-abs-tol-vars) is applied
# automatically per distribution from fptaylor_settings.toml (see that
# file's comments for the sweep_fptaylor.py benchmarking behind each
# choice) -- main.py picks it up via dist_common.apply_settings_defaults
# right after argument parsing, so this script only needs to pass each
# distribution's actual parameters. Pass any of the tuning flags below
# explicitly on the command line if you want to override the TOML for a
# one-off run; explicit flags always win.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

FP="fp64"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 1)"
VERBOSITY="-v"                    # "-v", "-vv", or "" for none

COMMON_ARGS=(--fp "$FP")
[ -n "$VERBOSITY" ] && COMMON_ARGS+=("$VERBOSITY")

# # # ---- binomial (BTRS) -------------------------------------------------------
python3 main.py "${COMMON_ARGS[@]}" binomial \
  --n 10900 --p 0.1 --jobs "$JOBS"

python3 main.py "${COMMON_ARGS[@]}" binomial \
  --n 1000000 --p 0.0001 --jobs "$JOBS"

# ---- poisson (PTRS) --------------------------------------------------------
python3 main.py "${COMMON_ARGS[@]}" poisson --lam 1e5

python3 main.py "${COMMON_ARGS[@]}" poisson --lam 1e8

# ---- hypergeometric (HRUA) -------------------------------------------------
python3 main.py "${COMMON_ARGS[@]}" hypergeometric --N 100 --K 40 --n 30

python3 main.py "${COMMON_ARGS[@]}" hypergeometric --N 10000 --K 4000 --n 300

# ---- zipf ------------------------------------------------------------------
# Not yet covered by fptaylor_settings.toml / sweep_fptaylor.py -- runs on
# dist_common's hardcoded defaults (approx=true, bb_eval=false).
python3 main.py "${COMMON_ARGS[@]}" zipf --s 2.5

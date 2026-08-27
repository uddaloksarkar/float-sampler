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
# -vv makes each dist_*.py print the full FPTaylor/CIRE/Gelpia output for
# every query (dist_common.vprint / each module's own "if verbose >= 2"
# checks) -- including the tool's own error text on a failed query, not
# just the terse "FPTaylor ... failed; see <path>.out" a case falls back
# to on error. Drop to "-v" once things are passing again; -vv is very
# noisy on a run with many (n, p)/lambda points.
VERBOSITY="-vv"                   # "-v", "-vv", or "" for none

COMMON_ARGS=(--fp "$FP")
[ -n "$VERBOSITY" ] && COMMON_ARGS+=("$VERBOSITY")

# Print a banner before each call and check its own exit code: main.py's
# per-item try/except (see dist_*.py run()) means a failed (n, p)/lambda
# point is reported as a "WARNING: skipping ..." line and does NOT make
# main.py exit non-zero, so under `set -e` the script would otherwise run
# straight through a run where every single point failed. run_one below
# makes that failure visible immediately instead of only in scrollback.
run_one() {
    echo "=== running: $* ==="
    local status=0
    "$@" || status=$?
    if [ "$status" -ne 0 ]; then
        echo "!!! command failed (exit $status): $*" >&2
    fi
}

# # # ---- binomial (BTRS) -------------------------------------------------------
run_one python3 main.py "${COMMON_ARGS[@]}" binomial \
  --n 10900 --p 0.1 --jobs "$JOBS"

run_one python3 main.py "${COMMON_ARGS[@]}" binomial \
  --n 1000000 --p 0.0001 --jobs "$JOBS"

# ---- poisson (PTRS) --------------------------------------------------------
run_one python3 main.py "${COMMON_ARGS[@]}" poisson --lam 1e5

run_one python3 main.py "${COMMON_ARGS[@]}" poisson --lam 1e8

# ---- poisson-stable (PTRS, cancellation-avoiding templates) ----------------
run_one python3 main.py "${COMMON_ARGS[@]}" poisson-stable --lam 1e5

run_one python3 main.py "${COMMON_ARGS[@]}" poisson-stable --lam 1e8

# ---- hypergeometric (HRUA) -------------------------------------------------
run_one python3 main.py "${COMMON_ARGS[@]}" hypergeometric --N 100 --K 40 --n 30

run_one python3 main.py "${COMMON_ARGS[@]}" hypergeometric --N 10000 --K 4000 --n 300

# ---- zipf ------------------------------------------------------------------
# Not yet covered by fptaylor_settings.toml / sweep_fptaylor.py -- runs on
# dist_common's hardcoded defaults (approx=true, bb_eval=false).
run_one python3 main.py "${COMMON_ARGS[@]}" zipf --s 2.5

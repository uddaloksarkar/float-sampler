#!/usr/bin/env bash
# Compile and run the self-contained test harness (main()) in every
# reference sampler under distributions/, reporting pass/fail.
#
# A run also fails if any "empirical X = A (theory = B)" line in its
# output has |A-B|/max(|B|,1e-12) greater than the tolerance. REL_TOL
# (default 0.5%) is the tolerance for a BASE_N-sample run (default
# 2000000); each source's actual #define N_SAMPLES is read from the
# source and the tolerance is scaled by sqrt(BASE_N / N_SAMPLES) to
# account for sampling noise.
set -u

cd "$(dirname "$0")"

CC=${CC:-cc}
REL_TOL=${REL_TOL:-0.005}
BASE_N=${BASE_N:-2000000}
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

pass=0
fail=0

for src in *.c; do
    name="${src%.c}"
    bin="$TMPDIR/$name"

    printf '=== %s ===\n' "$src"

    if ! "$CC" -O2 -Wall "$src" -lm -o "$bin" 2>"$TMPDIR/$name.compile.log"; then
        echo "  COMPILE FAILED"
        cat "$TMPDIR/$name.compile.log"
        fail=$((fail + 1))
        continue
    fi

    out="$TMPDIR/$name.out"
    if ! "$bin" >"$out"; then
        echo "  RUNTIME FAILED (exit $?)"
        cat "$out"
        fail=$((fail + 1))
        continue
    fi

    cat "$out"

    n_samples=$(grep -oE '#define N_SAMPLES [0-9]+' "$src" | grep -oE '[0-9]+' | head -1)
    n_samples=${n_samples:-$BASE_N}

    if ! REL_TOL="$REL_TOL" BASE_N="$BASE_N" N_SAMPLES="$n_samples" python3 - "$out" <<'EOF'
import math, os, re, sys

rel_tol   = float(os.environ["REL_TOL"])
base_n    = float(os.environ["BASE_N"])
n_samples = float(os.environ["N_SAMPLES"])
tol = rel_tol * math.sqrt(base_n / n_samples)

text = open(sys.argv[1]).read()

pair_re = re.compile(
    r"empirical\s+(\w+)\s*=\s*([-+0-9.eE]+).*?theory\s*=\s*([-+0-9.eE]+)"
)

ok = True
for label, emp_s, theory_s in pair_re.findall(text):
    emp, theory = float(emp_s), float(theory_s)
    rel_err = abs(emp - theory) / max(abs(theory), 1e-12)
    if rel_err > tol:
        print(f"  CHECK FAILED: {label} empirical={emp} theory={theory} "
              f"rel_err={rel_err:.4f} > tol={tol:.4f} (N={n_samples:.0f})")
        ok = False

sys.exit(0 if ok else 1)
EOF
    then
        fail=$((fail + 1))
        continue
    fi

    pass=$((pass + 1))
    echo
done

echo "============================"
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]

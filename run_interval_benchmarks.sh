#!/bin/bash

# Parallel runner for this repo's interval-mode benchmark suites
# (benchmarks/{binomial,poisson,hypergeometric}.tsv -- see
# benchmarks/generate_benchmarks.py). Lives next to main.py and the dist_*.py
# modules at the repo root; the benchmark *.tsv files themselves stay in
# benchmarks/.
# Modeled on ../ttc/run_polytope_ratios.sh: standalone, resumable (re-running
# skips benchmarks already done), one hung/crashed benchmark never takes the
# rest of the batch down with it, and everything merges into one combined CSV
# at the end. Unlike that script, this one is genuinely parallel (xargs -P),
# since main.py invocations are independent per-benchmark subprocesses with no
# shared state between them.
#
# Concurrency note: main.py's box-mode FPTaylor queries default to the
# interpreted --opt bb-eval backend for all three distributions here
# (fptaylor_settings.toml's bb_eval=true), which is safe to run concurrently.
# The *compiled* --opt bb backend is NOT (it writes fixed-named tmp/bb_1.ml,
# tmp/bb files that concurrent runs would clobber) -- do not pass --no-bb-eval
# together with JOBS>1.
#
# Usage:
#   ./run_interval_benchmarks.sh [dist] [outdir]
#     dist    binomial | poisson | hypergeometric | all   (default: all)
#     outdir  output directory                            (default: bench_out)
#
# Env overrides:
#   JOBS      parallel workers                    (default: nproc)
#   TLIMIT    per-benchmark wall-clock cap, sec    (default: 300)
#   MAIN_PY   path to main.py                      (default: first existing of
#             <this script's dir>/main.py, ./main.py, $FLOAT_SAMPLER_ROOT/main.py)
#   BENCH_DIR directory holding the *.tsv suites    (default: <this script's dir>/benchmarks)
#   PYTHON    python interpreter                   (default: python3)
#   MYRND     random-source file for a reproducible shuffle of row order,
#             same role as ttc's bins/myrnd          (default: first existing
#             of <this script's dir>/bins/myrnd, ./bins/myrnd,
#             $FLOAT_SAMPLER_ROOT/bins/myrnd; falls back to file order with
#             a warning if none found)
#   CACHE_DIR directory holding bench_cache/<dist>.csv, the cross-job ledger
#             of tags already known-good from EARLIER jobs/outdirs (see
#             plot_summary.py's update_cache) -- read once per dist, before
#             dispatch, to skip tags a previous run already finished, so a
#             fresh bench_out_<jobid> doesn't recompute the same FPTaylor
#             query twice. Read-only here by design (see plot_summary.py's
#             module docstring for why); this script never writes it.
#                                                   (default: <this script's
#             dir>/bench_cache; no effect if bench_cache/<dist>.csv doesn't
#             exist yet -- run plot_summary.py at least once to seed it)
#
# Examples:
#   ./run_interval_benchmarks.sh                      # everything, nproc workers
#   ./run_interval_benchmarks.sh binomial              # just binomial.tsv
#   JOBS=4 TLIMIT=60 ./run_interval_benchmarks.sh poisson quick_check

set -u

dist="${1:-all}"
outdir="${2:-bench_out}"
jobs="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
tlimit="${TLIMIT:-1800}"
python_bin="${PYTHON:-python3}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bench_dir="${BENCH_DIR:-}"
if [[ -z "$bench_dir" ]]; then
    bench_dir="${script_dir}/benchmarks"
fi

main_py="${MAIN_PY:-}"
if [[ -z "$main_py" ]]; then
    for cand in "${script_dir}/main.py" "./main.py" "${FLOAT_SAMPLER_ROOT:-}/main.py"; do
        if [[ -n "$cand" && -f "$cand" ]]; then
            main_py="$cand"
            break
        fi
    done
fi
if [[ -z "$main_py" || ! -f "$main_py" ]]; then
    echo "error: main.py not found (looked next to this script and ./main.py);" \
         "set MAIN_PY=<path> to override" >&2
    exit 1
fi
main_py="$(cd "$(dirname "$main_py")" && pwd)/$(basename "$main_py")"
repo_root="$(dirname "$main_py")"

# myrnd: a fixed-content random-source file, same role as ttc's
# bins/myrnd (script_ttc.sh's `shuf --random-source=...`) -- generate_
# benchmarks.py writes rows in scale order (small n/N first, extreme
# values last), so processing the file as-is means a time-boxed run that
# doesn't finish only ever covers the easy end and never reaches the hard
# tail. Shuffling with a *fixed* source keeps that reproducible (same
# order every rerun/config, so partial runs are directly comparable) while
# still giving a representative sample across the whole range. Not
# generated here -- same convention as ttc's bins/doalarm, checked in
# separately rather than produced by this script.
myrnd="${MYRND:-}"
if [[ -z "$myrnd" ]]; then
    for cand in "${script_dir}/bins/myrnd" "./bins/myrnd" "${FLOAT_SAMPLER_ROOT:-}/bins/myrnd"; do
        if [[ -n "$cand" && -f "$cand" ]]; then
            myrnd="$cand"
            break
        fi
    done
fi
if [[ -n "$myrnd" && -f "$myrnd" ]]; then
    shuf_cmd=(shuf "--random-source=${myrnd}")
else
    echo "warning: bins/myrnd not found -- processing benchmarks in file order" \
         "(scale-sorted, so a time-boxed run may never reach the hard tail);" \
         "set MYRND=<path> or add bins/myrnd to shuffle reproducibly" >&2
    shuf_cmd=(cat)
fi

cache_dir="${CACHE_DIR:-${script_dir}/bench_cache}"

case "$dist" in
    binomial|poisson|hypergeometric) dist_list=("$dist") ;;
    all) dist_list=(poisson binomial hypergeometric) ;;
    *)
        echo "error: dist must be binomial|poisson|hypergeometric|all, got '$dist'" >&2
        exit 1
        ;;
esac

mkdir -p "$outdir"

# ---------------------------------------------------------------------------
# Worker: run one benchmark row ("tag<TAB>args"), write its own out-dir/log,
# and a one-line status file the merge step reads back. Exported so xargs -P
# can call it as a subshell function per input line.
# ---------------------------------------------------------------------------
run_one() {
    local dist="$1" line="$2"
    local tag="${line%%$'\t'*}"
    local args="${line#*$'\t'}"
    local run_dir="${outdir}/${dist}/runs/${tag}"
    local log="${outdir}/${dist}/logs/${tag}.log"
    local status_file="${outdir}/${dist}/status/${tag}.status"

    if [[ -s "$status_file" ]]; then
        return 0   # already done -- resumable re-run
    fi
    mkdir -p "$(dirname "$log")" "$(dirname "$status_file")"
    rm -rf "$run_dir"

    local t0 t1 rc
    t0=$(date +%s.%N)
    timeout "$tlimit" "$python_bin" "$main_py" "$dist" $args \
        --out-dir "$run_dir" -v > "$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    local dt
    dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')

    local outcome
    if [[ $rc -eq 124 ]]; then
        outcome="TIMEOUT"
    elif [[ $rc -ne 0 ]]; then
        outcome="ERROR"
    else
        outcome="OK"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$outcome" "$rc" "$dt" "$args" > "$status_file"
    echo "  [$outcome] ${dist}/${tag}  (${dt}s, exit ${rc})"
}
export -f run_one
export outdir tlimit python_bin main_py

# ---------------------------------------------------------------------------
# Drive each distribution's suite through xargs -P for real parallelism.
# ---------------------------------------------------------------------------
for d in "${dist_list[@]}"; do
    tsv="${bench_dir}/${d}.tsv"
    if [[ ! -f "$tsv" ]]; then
        echo "error: benchmark file not found: $tsv (run generate_benchmarks.py first?)" >&2
        exit 1
    fi
    n_rows=$(grep -cv '^#' "$tsv")

    # Cache pre-filter: read bench_cache/<d>.csv (if plot_summary.py has
    # ever populated it) exactly once, up front -- not per-row and not from
    # inside run_one, so this stays a pure read done before any parallel
    # worker starts (see CACHE_DIR above for why writes never happen here).
    cache_csv="${cache_dir}/${d}.csv"
    done_tags=""
    n_cached=0
    if [[ -s "$cache_csv" ]]; then
        done_tags=$(mktemp)
        tail -n +2 "$cache_csv" | cut -d',' -f1 > "$done_tags"
        n_cached=$(wc -l < "$done_tags")
    fi

    echo "=== ${d}: ${n_rows} benchmark(s) in suite, ${n_cached} already" \
         "known-good in cache, ${jobs} parallel worker(s), ${tlimit}s cap each ==="

    if [[ -n "$done_tags" ]]; then
        grep -v '^#' "$tsv" | awk -F'\t' -v donefile="$done_tags" '
            BEGIN { while ((getline line < donefile) > 0) done[line] = 1 }
            !($1 in done) { print }
        ' | "${shuf_cmd[@]}" | xargs -d '\n' -P "$jobs" -I{} bash -c 'run_one "$0" "$1"' "$d" {}
        rm -f "$done_tags"
    else
        grep -v '^#' "$tsv" | "${shuf_cmd[@]}" | xargs -d '\n' -P "$jobs" -I{} bash -c 'run_one "$0" "$1"' "$d" {}
    fi
done

# ---------------------------------------------------------------------------
# Merge: one combined CSV per distribution, joining each benchmark's own
# summary.csv row with its status/timing, keyed by tag.
# ---------------------------------------------------------------------------
overall_ok=0
overall_total=0
for d in "${dist_list[@]}"; do
    combined="${outdir}/${d}/combined_summary.csv"
    status_dir="${outdir}/${d}/status"
    [[ -d "$status_dir" ]] || continue

    header_written=0
    : > "$combined"
    d_ok=0
    d_total=0
    for status_file in "$status_dir"/*.status; do
        [[ -e "$status_file" ]] || continue
        d_total=$((d_total + 1))
        IFS=$'\t' read -r tag outcome rc dt args < "$status_file"
        run_csv="${outdir}/${d}/runs/${tag}/summary.csv"
        if [[ "$outcome" == "OK" && -s "$run_csv" ]]; then
            d_ok=$((d_ok + 1))
            if [[ $header_written -eq 0 ]]; then
                head -n1 "$run_csv" | awk -F, -v OFS=, '{print "tag","outcome","time_s_wall",$0}' > "$combined"
                header_written=1
            fi
            tail -n +2 "$run_csv" | while IFS= read -r row; do
                printf '%s,%s,%s,%s\n' "$tag" "$outcome" "$dt" "$row"
            done >> "$combined"
        else
            printf '%s,%s,%s,(no summary -- see logs/%s.log)\n' \
                "$tag" "$outcome" "$dt" "$tag" >> "${outdir}/${d}/failures.csv"
        fi
    done
    echo "${d}: ${d_ok}/${d_total} succeeded -> ${combined}"
    if [[ -f "${outdir}/${d}/failures.csv" ]]; then
        echo "  failures logged in ${outdir}/${d}/failures.csv"
    fi
    overall_ok=$((overall_ok + d_ok))
    overall_total=$((overall_total + d_total))
done

echo "=== done: ${overall_ok}/${overall_total} benchmark(s) succeeded across ${dist_list[*]} ==="
if [[ $overall_ok -eq 0 ]]; then
    echo "error: no benchmark produced a result; check ${outdir}/*/logs/" >&2
    exit 1
fi

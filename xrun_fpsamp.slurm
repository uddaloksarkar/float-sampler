#!/bin/bash

#SBATCH --job-name=fpsampler-bench
# # SBATCH --output=j-fpsamp-corr-%j.out
# #SBATCH --error=j-fpsamp-corr-%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uddalok.sarkar@mail.utoronto.ca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --account=def-ksmeel

module load StdEnv/2023
module load python

# Unlike ttc's run_polytope_ratios.sh (a plain sequential loop, parallelized
# here only by launching many redundant MPI ranks that race and deduplicate
# via its own resumability check), run_interval_benchmarks.sh parallelizes
# itself internally with `xargs -P` -- so this needs one task with many CPUs,
# not ttc's many-node/many-task/openmpi layout, and no mpirun wrapper at all.
export JOBS="${SLURM_CPUS_PER_TASK:-4}"

cd "${SLURM_SUBMIT_DIR}"

# Job-ID-suffixed output dir: bench_out (no suffix) is a shared name every
# run/rerun would otherwise write into, silently mixing rows from different
# jobs (and from any local smoke-testing) into one combined_summary.csv --
# already happened once this way. Each job gets its own bench_out_<jobid>,
# so results from different submissions never collide, and it's obvious
# from the directory name alone which job produced which rows.
outdir="${SLURM_SUBMIT_DIR}/bench_out_${SLURM_JOB_ID}"

# Sequential, not backgrounded: each call already uses all $JOBS cores via
# its own xargs -P, so running the three distributions concurrently here
# would oversubscribe the node 3x for no benefit. Increase --time above (or
# lower TLIMIT per call) if 4h isn't enough to make a useful dent in
# benchmarks/{binomial,hypergeometric}.tsv's ~72K/~52K rows -- most rows
# past roughly n~1e6 (binomial) or the scale where hypergeometric's shapes
# go vacuous are expected to time out or fail by design (see
# benchmarks/generate_benchmarks.py's module docstring), not stall the job.
./run_interval_benchmarks.sh poisson        "${outdir}"
./run_interval_benchmarks.sh binomial       "${outdir}"
./run_interval_benchmarks.sh hypergeometric "${outdir}"

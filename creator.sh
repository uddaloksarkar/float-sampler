#!/bin/bash

source ~/opam-fptaylor-env.sh
source ~/.venvs/udda/bin/activate

# One dataset file per distribution (see generate_benchmark_dataset.py),
# each already in that distribution's own native input-file format:
#   poisson:        "lambda"        one per line
#   binomial:       "n p"           one per line
#   hypergeometric: "N K n"         one per line
# dist_flags maps 1/2/3 space-separated values on a line to main.py's flags.
poisson_file="benchmark_poisson_lambdas.txt"
binomial_file="benchmark_binomial_np.txt"
hypergeometric_file="benchmark_hypergeometric_NKn.txt"

# All FPTaylor optimizer tuning (--approx/--no-approx, --bb-eval,
# --v-trunc/--u-trunc, per-variable --opt-x-abs-tol-vars) is applied
# automatically per distribution from fptaylor_settings.toml (see that
# file's comments for the sweep_fptaylor.py benchmarking behind each
# choice) -- main.py picks it up via dist_common.apply_settings_defaults
# right after argument parsing, so each case below only needs to pass its
# own distribution parameters. Hypergeometric's W tolerance (which scales
# with N, see hrua_z_range) is likewise auto-derived per case inside
# dist_hypergeometric.run() now, not computed here.
FP="fp64"

ulimit -t unlimited
shopt -s nullglob
rm -f todo
touch todo
solver="fp-benchmark"

tlimit="500"

SLURM_JOB_ID=${SLURM_JOB_ID:-local}
SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}

# (dist_name, params_file) pairs; the actual --flag names for however many
# space-separated values one line holds are chosen per distribution below
# (poisson: --lam; binomial: --n --p; hypergeometric: --N --K --n).
dist_files=(
    "poisson:${poisson_file}"
    "binomial:${binomial_file}"
    "hypergeometric:${hypergeometric_file}"
)

output="out"
#10GB mem limit
memlimit="10000000"
numthreads=${OMPI_COMM_WORLD_SIZE:-1}
MYRANK=${OMPI_COMM_WORLD_RANK:-0}
SCRATCH=${SCRATCH:-/tmp}

SERVER=$SLURM_SUBMIT_HOST
WORKDIR="$SCRATCH/scratch/${SLURM_JOB_ID}_${MYRANK}"
output="${output}-${SLURM_JOB_ID}"

# echo ------------------------------------------------------
# echo "Job is running on node ${PBS_NODEFILE}"
# echo ------------------------------------------------------
# echo "Rank is: ${OMPI_COMM_WORLD_RANK}"
# echo "PBS: qsub is running on $PBS_O_HOST"
# echo "PBS: originating queue is $PBS_O_QUEUE"
# echo "PBS: executing queue is $PBS_QUEUE"
# echo "PBS: working directory is $SLURM_SUBMIT_DIR"
# echo "PBS: execution mode is $PBS_ENVIRONMENT"
# echo "PBS: job identifier is ${SLURM_JOB_ID}"
# echo "PBS: job name is $PBS_JOBNAME"
# echo "PBS: node file is $PBS_NODEFILE"
# echo "PBS: current home directory is $PBS_O_HOME"
# echo "PBS: PATH = $PBS_O_PATH"
# echo "server      is ${SERVER}"
# echo "workdir     is ${WORKDIR}"
# echo "permdir     is ${PERMDIR}"
# echo "servpermdir is ${SERVPERMDIR}"
# echo "Output dir  is ${output}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}" || exit

outputdir="${SLURM_SUBMIT_DIR}/${solver}-main"

echo "echo 'copying...'"
echo "echo 'copied'"

# create todo
rm -f todo
mkdir -p ${output}
numlines=0
at_opt=0
for entry in "${dist_files[@]}"
do
    dist="${entry%%:*}"
    dist_file="${entry#*:}"
    lines=$(tr ',' '\n' < "${SLURM_SUBMIT_DIR}/${dist_file}" | grep -v '^$' \
            | shuf --random-source=${SLURM_SUBMIT_DIR}/myrnd)

    mkdir -p "${output}-${at_opt}" || exit
    while IFS= read -r line
    do
        [ -z "$line" ] && continue
        # each line is already "lam" / "n p" / "N K n" -- turn its
        # space-separated tokens into this distribution's own flags.
        # Read line-by-line (not `for line in $lines`): binomial/
        # hypergeometric lines carry 2-3 space-separated values, and
        # unquoted word-splitting would shred each into separate "lines".
        read -ra tok <<< "$line"
        case "$dist" in
            poisson)
                distargs="--fp ${FP} poisson --lam ${tok[0]}"
                tag="poisson_lam_${tok[0]}"
                ;;
            binomial)
                distargs="--fp ${FP} binomial --n ${tok[0]} --p ${tok[1]}"
                tag="binomial_n_${tok[0]}_p_${tok[1]}"
                ;;
            hypergeometric)
                distargs="--fp ${FP} hypergeometric --N ${tok[0]} --K ${tok[1]} --n ${tok[2]}"
                tag="hypergeometric_N_${tok[0]}_K_${tok[1]}_n_${tok[2]}"
                ;;
            *)
                echo "unknown distribution ${dist}, skipping" >&2
                continue
                ;;
        esac
        opts="python3 ${SLURM_SUBMIT_DIR}/main.py --out-dir ${WORKDIR}/${output}-${at_opt}/${tag}_run ${distargs}"
        echo "doing ${dist} (${line})"
        # run
        baseout="${output}-${at_opt}/${tag}"
        mytimeout="timeout -k 2 ${tlimit} "
        echo "/usr/bin/time --verbose -o ${baseout}.timeout ${mytimeout} ${opts} > ${baseout}.out 2>&1" >> todo

        echo "mkdir -p  ${outputdir}/${output}-${at_opt}" >> todo
        echo "xz ${baseout}.out*" >> todo
        echo "xz ${baseout}.timeout*" >> todo
        echo "rm -f core.*" >> todo

        echo "mv ${baseout}.out*      ${outputdir}/${output}-${at_opt}/" >> todo
        echo "mv ${baseout}.timeout*  ${outputdir}/${output}-${at_opt}/" >> todo
        echo "mv core.* ${outputdir}/${output}/" >> todo

        # delete what's left
        echo "rm -f ${baseout}.timeout*" >> todo
        echo "rm -f ${baseout}.out*" >> todo

        # todos: 1+5+3+1 = 10

        numlines=$((numlines+1))
    done <<< "$lines"
    at_opt=$((at_opt+1))
done
todomylines=10

# create per-core todos
numper=$((numlines/numthreads))
remain=$((numlines-numper*numthreads))
if [[ $remain -ge 1 ]]; then
    numper=$((numper+1))
fi
remain=$((numlines-numper*(numthreads-1)))

mystart=0
for ((myi=0; myi < numthreads ; myi++))
do
    rm -f todo_$myi.sh
    touch todo_$myi.sh
    echo "#!/bin/bash" > todo_$myi.sh
    echo "source ~/.venvs/udda/bin/activate" >> todo_$myi.sh
    echo "export PYTHONPATH=${SLURM_SUBMIT_DIR}:\$PYTHONPATH" >> todo_$myi.sh
    echo "ulimit -v $memlimit" >> todo_$myi.sh
    echo "ulimit -c 0" >> todo_$myi.sh
    echo "set -x" >> todo_$myi.sh
    typeset -i myi
    typeset -i numper
    typeset -i mystart
    mystart=$((mystart + numper))
    if [[ $myi -lt $((numthreads-1)) ]]; then
        if [[ $mystart -gt $((numlines+numper)) ]]; then
            # echo "No need, over the limit by more than numper"
            sleep 0
        else
            if [[ $mystart -lt $numlines ]]; then
                myp=$((numper*todomylines))
                mys=$((mystart*todomylines))
                head -n $mys todo | tail -n $myp >> todo_$myi.sh
            else
                #we are at boundary, e.g. numlines is 100, numper is 3, mystart is 102
                #we must only print the last numper-(mystart-numlines) = 3-2 = 1
                mys=$((mystart*todomylines))
                p=$(( numper-mystart+numlines ))
                if [[ $p -gt 0 ]]; then
                    myp=$((p*todomylines))
                    head -n $mys todo | tail -n $myp >> todo_$myi.sh
                fi
            fi
        fi
    else
        if [[ $remain -gt 0 ]]; then
            mys=$((mystart*todomylines))
            mr=$((remain*todomylines))
            head -n $mys todo | tail -n $mr >> todo_$myi.sh
        fi
    fi
    echo "exit 0" >> todo_$myi.sh
    chmod +x todo_$myi.sh
done
# echo "Done."

# Execute todos
echo "This is MPI exec number $MYRANK"
rm -f ${output}/out_${MYRANK}
./todo_${MYRANK}.sh > ${output}/out_${MYRANK}
echo "Finished waiting rank $MYRANK"

exit 0

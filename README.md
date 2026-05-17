# FPSampler (FPTaylor for Sampler)

Tools for bounding the statistical distance between the ideal and finite-precision
Knuth–Poisson sampler and TRS sampler, using rigorous floating-point error analysis via
[FPTaylor](https://github.com/soarlab/FPTaylor) and
[Gelpia](https://github.com/soarlab/gelpia).

---

## Repository layout

```
computeDelta/
├── analyticError.py            # Analytical bound library + CLI
├── fpsampler.py     # Verified bound runner (calls FPTaylor / Gelpia)
├── creator.sh                 # Cluster / MPI worker script
├── lambdas_1_100_step1.txt    # λ = 1, 2, …, 100
├── lambdas_10_100_step10.txt
├── lambdas_100_1000_step100.txt
├── lambdas_100_100000_10_per_decade.txt
├── FPTaylor/                  # git submodule
└── gelpia/                    # git submodule
```

---

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
# or, if already cloned:
git submodule update --init --recursive
```
Build the dependecies `FPTaylor` and `Gelpia`

```bash
cd FPTaylor
make
cd ..
```

FPTaylor requires OCaml (`opam install ocamlfind num`). The binary is expected at
`FPTaylor/fptaylor`; alternatively set `$FPTAYLOR` or pass `--fptaylor <path>`.

```bash
cd gelpia
make requirements
make
cd ..
```

### 2. Fix Python environment

```bash
python3 -m venv ~/.venvs/cdelta
source ~/.venvs/cdelta/bin/activate
pip install matplotlib
```

---

## Overview of the two regimes

| Regime | Condition | Method |
|---|---|---|
| **Low range** | λ < 30 | FPTaylor bounds the product error of K\* = ⌊λ + 10√λ⌋ multiplications and the exp error; combined as Δ ≤ 2E / (e^{−λ} − E) |
| **High range** | λ ≥ 30 | FPTaylor computes ΔE and ΔK; Gelpia minimises h; combined as Δ = ΔE + ΔH |

---

## Analytical bounds (`analyticError.py`)

Quick closed-form bound, no external tools needed (Used as a benchmark).

```bash
# Bound for a single lambda (pass log2(lambda))
python analyticError.py --loglam 6 --fp fp64

# Plot Δ vs log2(λ) for all precisions
python analyticError.py --plot
```

| Flag | Description |
|---|---|
| `--loglam N` | λ = 2^N |
| `--fp {fp32,fp64,fp128}` | floating-point format |
| `--plot` | show Δ and component plots |

---

## FPSampler 

Runs FPTaylor (and Gelpia for λ ≥ 40) to get rigorous numerical bounds, writes
results to a CSV, and optionally plots them.

### Basic usage

```bash
# Single lambda
python fpsampler.py --lam 5

# Batch from a file
python fpsampler.py lambdas_1_100_step1.txt

# With plotting
python fpsampler.py lambdas_1_100_step1.txt --plot --plot-components
```

### Important flags

| Flag | Default | Description |
|---|---|---|
| `lambda_file` | — | File of λ values (one per line, or comma-separated) |
| `--lam N` | — | Single λ value (mutually exclusive with `lambda_file`) |
| `--out-dir PATH` | `total_error_runs_<stem>` | Output directory |
| `--plot` | off | Generate error-vs-lambda plot |

### Output structure

```
total_error_runs_<stem>/
├── summary.csv           # one row per lambda: regime, ΔE, ΔH, total, TV, …
├── total_error_vs_lambda.png   (if --plot)
├── total_error_vs_lambda.pgf   (if --plot --plot-pgf)
├── inputs/               # FPTaylor .txt and Gelpia .dop input files
└── outputs/              # raw .out files from each tool invocation
```



### Lambda input files

Plain text, one value per line (comments with `#`, comma separation also accepted):

```
# lambdas_1_100_step1.txt
1
2
...
100
```

Generate a custom list, e.g. 200–500 step 50:

```bash
python3 -c "print('\n'.join(str(i) for i in range(200, 501, 50)))" > my_lambdas.txt
```


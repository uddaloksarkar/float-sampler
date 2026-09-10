# FPSampler (FPTaylor for Sampler)

Tools for bounding the statistical distance between the ideal and finite-precision
Knuth–Poisson sampler and TRS sampler, using rigorous floating-point error analysis via
[FPTaylor](https://github.com/soarlab/FPTaylor) and
[Gelpia](https://github.com/soarlab/gelpia).

---

## Repository layout

```
computeDelta/
├── analyticError.py                        # Analytical bound library + CLI
├── fpsampler.py                            # Verified bound runner (calls FPTaylor / Gelpia)
├── lambdas_1_100_step1.txt                 # λ = 1, 2, …, 100
├── lambdas_10_100_step10.txt               # λ = 10, 20, …, 100
├── lambdas_100_1000_step100.txt            # λ = 100, 200, …, 1000
├── lambdas_100_100000_10_per_decade.txt
├── distributions/                          # reference C samplers (Knuth, PTRS, BTPE, inversion)
├── FPTaylor/                               # git submodule
└── gelpia/                                 # git submodule
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
| **Low range** | λ < 30 | FPTaylor bounds the product error of K\* = ⌊λ + 10√λ⌋ multiplications and the exp error; combined as Δ ≤ 2E / (e^{−λ} − E) — **computationally expensive; expect long runtimes** |
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

## main.py (multi-distribution CLI)

The general entry point: one dispatcher over all supported distributions, each
implemented as a `dist_<name>.py` module (`poisson`, `poisson-stable`,
`binomial`, `geometric`, `hypergeometric`, `zipf`). It runs the FP-error
analysis (FPTaylor by default, or CIRE via `--backend cire`), writes a
`summary.csv`, and optionally plots TV vs. the distribution's parameter.

```bash
python main.py <distribution> <positional-input-or-flags> [common opts]
```

### Examples

```bash
# Poisson, a file of lambda values
python main.py poisson lambdas_100_1000_step100.txt

# Poisson, single lambda
python main.py poisson --lam 50

# Numerically-stable Poisson reformulation (see distributions/random_poisson_ptrs_stable.c)
python main.py poisson-stable --lam 50

# Binomial, a file of (n, p) pairs
python main.py binomial pairs.txt

# Binomial, a single (n, p) box bound instead of a point
python main.py binomial --n-range 900 1100 --p-range 0.09 0.11

# Geometric / hypergeometric / zipf follow the same pattern
python main.py geometric --p 0.2
python main.py hypergeometric --N 100 --K 40 --n 30
python main.py zipf --s 2.0

# With plotting, verbose FPTaylor output, and a custom output dir
python main.py poisson lambdas_10_100_step10.txt --plot -vv --out-dir poisson_runs
```

Run `python main.py <distribution> --help` to see that distribution's own
positional/`--`-flag input options (e.g. `--lam`, `--n`/`--p`, `--N`/`--K`/`--n`).

### Common flags (all distributions)

| Flag | Default | Description |
|---|---|---|
| `--backend {fptaylor,cire}` | `fptaylor` | FP analysis backend |
| `--fptaylor PATH` | auto-detect / `$FPTAYLOR` | Path to the FPTaylor executable |
| `--cire PATH` | auto-detect | Path to the CIRE_LLVM executable (`--backend cire`, fp64 only) |
| `--fp {fp32,fp64,fp128}` | `fp64` | Floating-point format |
| `--out-dir PATH` | `<dist>_runs[_<stem>]/` | Output directory |
| `--plot` | off | Plot TV vs. the distribution's parameter |
| `--plot-components` | off | Include individual error components in the plot |
| `--plot-pgf` | off | Also save the plot as PGF |
| `--plot-file PATH` | `<out-dir>/tv_vs_param.png` | Plot output path |
| `--cache` | off | Reuse an existing `summary.csv` in `--out-dir` instead of re-running |
| `-v` / `-vv` | off | `-v`: per-problem internal parameters; `-vv`: also raw tool output |
| `--v-trunc FLOAT` | per-distribution (`fptaylor_settings.toml`) | BTRS/PTRS: truncation floor for the `log(v)` domain |
| `--u-trunc FLOAT` | per-distribution (`fptaylor_settings.toml`) | BTRS/PTRS: minimum allowed `us` at the reachable-k boundary |
| `--bb-geometric-ratio-tol FLOAT` | `2.0` | FPTaylor branch-and-bound geometric-splitter ratio |
| `--bb-eval` / `--no-bb-eval` | per-distribution | Use FPTaylor's interpreted `--opt bb-eval` backend instead of the compiling `--opt bb` one |

Per-distribution defaults for several of these (`approx`, `bb_eval`,
`v_trunc`, `u_trunc`, `opt_x_abs_tol`, ...) live in `fptaylor_settings.toml`
and are applied automatically unless overridden on the command line.

### Output structure

Same layout as FPSampler below (`summary.csv`, `inputs/`, `outputs/`, and the
plot if `--plot` is passed), under `--out-dir` (default `<dist>_runs[_<stem>]/`).

---

## FPSampler 

Runs FPTaylor (and Gelpia for λ ≥ 30) to get rigorous numerical bounds, writes
results to a CSV, and optionally plots them.

### Basic usage

```bash
# Single lambda
python fpsampler.py --lam 5

# Batch from a file
python fpsampler.py lambdas_100_1000_step100.txt

# contains Knuth Sampler calls for λ = 10, 20 (expect long run times)
python fpsampler.py lambdas_10_100_step10.txt

# With plotting
python fpsampler.py lambdas_10_100_step10.txt --plot 
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


# WCARPCG

## Requirements

Linux or WSL, Python 3.9+, and a C99 compiler (GCC or Clang).

## Installation

```sh
git clone https://github.com/m478496829/WCARPCG.git
cd WCARPCG
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Check the supplied inputs and operators:

```sh
python reproduce.py check
```

Solve a small system and compare with a sparse direct solution:

```sh
python reproduce.py demo --case ovt_varnu_f1 --beta 0.5 --tol 1e-8
```

Generate the spectral plot from the supplied data:

```sh
python paper_figure/draw_spectrum.py
```

Recompute a complete spectrum (allow 1--2 GiB of memory):

```sh
python reproduce.py spectrum --case ovt_varnu_f1 --beta 0.5
```

Available `--case` values: `ovt_varnu_f1`, `ovt_varnu_f1p5`,
`ovt_nu25_f1`, and `ovt_nu49_f1`. Available `--beta` values: `0`, `0.5`, and `1`.
Use `python reproduce.py --help` for all options.

The C kernel builds automatically on first use. Set `CC=clang` to use Clang.
Runs use one thread by default; set `WCARP_THREADS=4` to use four BLAS threads.

## Outputs

- `results/quick_check.json`: input and operator checks.
- `results/demo/`: convergence histories, complex solutions, and direct-solution comparisons.
- `results/spectra/`: recomputed spectra and numerical checks.
- `paper_figure/Fig_complete_defect_spectrum.pdf`: spectral plot (PNG and SVG are also generated).

Completed spectrum calculations are reused when the same command is repeated.

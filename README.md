# WCARP-CG: small-system reproducibility package

Supporting code and numerical data for:

**Column-Norm-Weighted CARP-CG: A Metric-Consistent Formulation for Frequency-Domain Elastic Wave Systems**

Rui Mao, Yang Li, Zhenwu Fu, Guangyu Gao, and Yong Chen.

This package accompanies the JSC Revision28 manuscript prepared on 26 October
2026. It provides executable weighted row sweeps, component averaging, an
induced-metric CG demonstration, and the numerical inputs and reference outputs
for the manuscript's small three-dimensional spectral tests.

## What can be reproduced

1. **The four small 3-D systems:** 2,187 complex velocity DOFs each, eight fixed
   row blocks, and three weighting exponents (`beta=0, 0.5, 1`). The exported
   local sparse matrices, right-hand sides, copy maps, weights, and production
   operator probes are included.
2. **The complete spectra:** construct the actual double-sweep operator and its
   similarity transform in the induced metric, compute all 2,187 eigenvalues,
   and compare with the archived values for each system and exponent.
3. **A small solve:** run WCARP-CG to an explicitly evaluated row-scaled relative
   residual and compare all complex solution components with a SciPy sparse
   direct solution of the identical linear system.
4. **The manuscript's spectral figure:** regenerate its two panels from the
   supplied full-spectrum and condition-number CSV files.

The input arrays are numerical exports from the original small-system study;
they are not random surrogate matrices. `PROVENANCE.json` records their hashes.
The package starts at these exported operators. It does not regenerate the
finite-difference matrices from the original geological models, and does not
reproduce the manuscript's large MPI timing or memory measurements. No full
Marmousi2/Overthrust model files or proprietary compiler/runtime binaries are
included.

## Requirements

Linux (or WSL), Python 3.9 or newer, a C99 compiler such as GCC, NumPy, SciPy,
Matplotlib, and threadpoolctl. Intel compilers and MPI are not required: the
eight row blocks are applied serially while retaining their original order and
copy maps. No network connection is needed after installing the dependencies.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py check
python reproduce.py demo --case ovt_varnu_f1 --beta 0.5
python paper_figure/draw_spectrum.py
```

The C kernel is compiled automatically on first use. Set `CC=clang` or another
C99 compiler if needed. Generated files go to `results/` and `paper_figure/`;
the archived numerical inputs and reference outputs are never overwritten.

### Fast verification

`python reproduce.py check` verifies all 96 input-file hashes, reconstructs the
weights and induced metric, checks that all three exponents use the same `A,b`,
compares forward/double-sweep actions with the production-exported probes, checks
the new affine kernel against a direct Python row loop, and checks the saved
spectra against the manuscript figure values. This command **does not recompute
the full eigendecompositions**. Its report is `results/quick_check.json`.

### Small solve and direct-solution comparison

The default demonstration uses the variable-Poisson-ratio 1 Hz system with
`beta=0.5`, starts at zero, checks `||b-Ax||_2/||b||_2` at every iteration, and
stops at `1e-8`. The solve is CG on the symmetric similarity transform associated
with the induced diagonal metric M, algebraically equivalent to M-inner-product
CG. Complete complex solutions and the convergence history are saved.

On the packaging validation run this required 445 iterations, with a row-scaled
relative residual of about `9.54e-9` and a relative complex-field difference to
the direct solution of about `1.15e-7`. These are **new portable demonstration
results**, not replacement values for a table in the manuscript. The original
small spectral study did not report these CG trajectories. Last digits and
iteration counts can vary with floating-point libraries.

```sh
python reproduce.py demo --case ovt_varnu_f1 --beta 0.5 --tol 1e-8
```

The `--beta 0` option is the equal-weight formulation with its induced metric.
It is not presented as a reproduction of the historical unweighted production
solver's timing. This package reports no MPI performance speedup.

### Recompute a complete spectrum

```sh
python reproduce.py spectrum --case ovt_varnu_f1 --beta 0.5
```

This assembles dense complex 2187-by-2187 matrices and computes all eigenpairs.
Allow approximately 1--2 GiB of memory and a few minutes depending on hardware.
The implementation checks self-adjointness, the forward/reverse factorization,
operator actions, and eigenpair residuals, then compares the eigenvalues with
the archived reference. Re-running a completed spectrum reuses its output in
`results/spectra/`; use a fresh checkout to perform a clean recomputation.

All four case names:

| Case | Frequency | Poisson's ratio |
| --- | ---: | --- |
| `ovt_varnu_f1` | 1 Hz | spatially varying |
| `ovt_varnu_f1p5` | 1.5 Hz | spatially varying |
| `ovt_nu25_f1` | 1 Hz | 0.25 |
| `ovt_nu49_f1` | 1 Hz | 0.49 |

Each comes from a 5-by-5-by-5 physical crop with two PML cells on each side,
giving a 9-by-9-by-9 computational grid, at 300 m spacing. The row partition is
2-by-2-by-2 and row relaxation is one. See `PLAN.json` for machine-readable
settings. To reproduce all twelve spectra:

```sh
for case in ovt_varnu_f1 ovt_varnu_f1p5 ovt_nu25_f1 ovt_nu49_f1; do
  for beta in 0 0.5 1; do
    python reproduce.py spectrum --case "$case" --beta "$beta"
  done
done
```

Runs use one BLAS thread by default. Set `WCARP_THREADS=4` explicitly if desired.
The full spectral calculation is optional; the fast checks, small solve, and
figure regeneration are the recommended first steps.

## Mathematical conventions

The exported coefficients are already row-scaled. For each local copy j,
`chi_j = sum_i |a_ij|^2`, `epsilon = 1e-8 * max(chi)`, and
`w_j = (chi_j + epsilon)^beta`. Weights stay fixed during the solve. Global
`m_j` is the sum of weights over all local copies of global variable j.

Each local row update is

```text
x_j <- x_j + theta * conj(a_ij)/w_j * (b_i - a_i x)
                  / sum_l (|a_il|^2/w_l).
```

A forward sweep is followed by weighted component averaging; then the reverse
sweep is followed by another weighted component averaging. Structural column
indices with zero stored coefficients are retained. Dropping them can change
the copy map and therefore the operator.

The archived spectral kernel includes the production denominator safeguard;
the fast checks verify that it is inactive on every supplied system.

## Repository contents

- `exporter/runs/`: fixed per-block numerical inputs and production probes.
- `analysis/row_sweeps.c`: archived homogeneous row-sweep kernel.
- `analysis/analyze_spectrum.py`: portable adaptation of the archived complete-spectrum analysis.
- `analysis/affine_sweeps.c`: affine version of the same row update for the demo.
- `reproduce.py`: input checks, small solve, and full-spectrum entry point.
- `reference/`: archived eigenvalues and numerical verification summaries.
- `paper_figure/`: manuscript CSV data and the spectral figure's plotting source.
- `validation/`: results from validating this portable package.
- `PROVENANCE.json`, `SHA256SUMS`: provenance and content checksums.

## Data provenance and reuse

The supplied operator arrays and spectral outputs are derived numerical research
data. The parent geological model is the modified Overthrust model described in
the manuscript and its cited SEG/EAGE model reference. Original geological
datasets remain subject to their own licensing terms; this repository does not
redistribute the full source models or grant rights to those source datasets.

This release makes the included code and derived data available for examining
and reproducing the reported computations. No new general-purpose software or
third-party data licence is asserted. For redistribution or reuse beyond that
purpose, contact the corresponding author, Yang Li (`yang.li.math@hit.edu.cn`).

Please cite the manuscript when using these materials. Publication details and
DOI can be added once available.

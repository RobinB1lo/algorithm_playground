# GMDH Variants Comparison Project — Full Context

## Purpose

Academic research project under Professor Bonakdari implementing, comparing,
and documenting 13 variants of the Group Method of Data Handling (GMDH)
algorithm in Python, following Ivakhnenko's 1971 source paper. Goal: academic
publication (write-up in Typst, in progress).

**Dual objective:**
- Keep implementations as faithful as possible to Ivakhnenko's original
  paper, for the professor's publication.
- Separately investigate optimizations (e.g. Fibonacci threshold search) —
  kept in isolated modules rather than the core class, so optimization work
  never silently compromises the "faithful" implementations reviewers will
  check against the source paper.

**All departures from the source paper must be documented explicitly** —
never silently absorbed into an implementation.

## Environment

- macOS 15, Apple Silicon (arm64)
- Python 3.14, plain **venv** (not conda) — `algo_env/` inside project root
- R 4.6.0 at `/Library/Frameworks/R.framework/`
- Key Python libraries: NumPy, SciPy (Nelder-Mead, BSpline), scikit-learn,
  rpy2 (R bridge)
- R reference: `GMDHreg` (local/custom-built R package), via `rpy2`
- Documentation: Typst

**Key implementation defaults used throughout the codebase:**
- Ridge parameter: `1e-6`
- Train/selection split: 50/50
- Nelder-Mead: `maxiter=100`
- Exponent bounds: `[0.1, 5.0]`
- B-spline defaults: `n_knots=3`, `k=3` (cubic)

## Directory Structure

```
GMDH/
├── vanilla/
│   ├── gmdh.py                      # Base GMDH implementation
│   └── threshold_optimization.py    # Fibonacci threshold search (kept separate
│                                     #   from core class to preserve faithfulness)
├── evaluation_metric_change/
│   ├── aic/
│   │   └── gmdh_aic.py              # GMDH-AIC
│   └── hierarchical/
│       └── hierarchical_gmdh.py     # GMDH-Hierarchical (multi-metric)
├── cv_and_look_back/
│   ├── cv/
│   │   └── cv_gmdh.py               # GMDH-CV
│   └── look_back/
│       └── look_back_gmdh.py        # GMDH-LookBack
├── fractional_polynomials/
│   ├── cfp/
│   │   └── cfp_gmdh.py              # Constrained Fractional Polynomials
│   ├── ufp/
│   │   └── ufp_gmdh.py              # Unconstrained Fractional Polynomials
│   ├── ufp_hierarchical/
│   │   └── ufp_hierarchical_gmdh.py # UFP + Hierarchical Ranking (high novelty)
│   └── ufp_hierarchical_cv/
│       └── ufp_hierarchical_cv_gmdh.py # UFP + Hierarchical + CV (strongest overall)
├── basis_change/
│   ├── spline/
│   │   └── spline_basis_gmdh.py     # B-spline basis
│   ├── radial/
│   │   └── rbf_basis_gmdh.py        # RBF basis
│   ├── sigmoid/
│   │   └── sigmoid_basis_gmdh.py    # Sigmoid basis
│   └── fourier/
│       └── fourier_basis_gmdh.py    # Trigonometric/Fourier basis
├── regular/
│   ├── gmdhref_wrapper.py           # GMDHreg_Wrapper (rpy2 bridge to R reference)
│   └── GMDHreg.R                    # Local reference copy of R implementation
├── comparison.py                    # run_comparison() + build_models() — root-level
├── fibonacci_search.py
├── notes/
├── regressions/
├── current_instructions.md
└── README.md
```

All submodules use explicit imports and empty `__init__.py` files, e.g.:
```python
from basis_change.spline.spline_basis_gmdh import GMDH_Spline
```

## Basis Function Structural Pattern

All basis-change variants (`spline`, `radial`, `sigmoid`, `fourier`) follow an
identical structure: univariate basis functions computed separately for
inputs `a` and `b`, combined as `[1, a_basis, b_basis, interactions]`, fit via
ridge regression.

## Comparison Framework (`comparison.py`)

- `build_models()`: factory producing fresh, unfitted instances of all 13
  GMDH variants + baselines (Linear Regression, Ridge, Random Forest,
  Gradient Boosting) + the R reference (`GMDHreg_Wrapper`, inserted
  conditionally if `R_AVAILABLE`).
- `run_comparison()`: runs the full comparison on one dataset, prints a
  report, saves a labeled plot.
- Two synthetic datasets:
  - **Dataset A ("Wildfire")**: 4 features, 500 samples, additively
    separable, no genuine pairwise interactions. General nonlinear
    regression benchmark, *not* GMDH's target regime.
  - **Dataset B ("High-Dimensional Short-Sequence")**: 30 features, 60
    samples, genuine pairwise interactions among informative features,
    redundant/collinear features, pure-noise irrelevant features. Built
    specifically to match GMDH's intended regime (Ivakhnenko motivates GMDH
    around large dimensionality with very short data sequences).

## R Reference Validation (`GMDHreg_Wrapper`)

**Why:** validate the Python vanilla GMDH against a published, peer-reviewed
R package, since the official `gmdh` PyPI package failed to install (missing
cmake/Boost).

**Which R function:** `gmdh.combi` (classic combinatorial GMDH — matches
Ivakhnenko's original layer-by-layer pairwise polynomial approach), accessed
in Python as `gmdh_pkg.gmdh_combi(...)`, with `predict.combi` /
`predict_combi(...)` for prediction. (Other exports — `gmdh.gia`, `gmdh.mia`,
`gmdh.combi.twice` — use different selection criteria/architectures and were
not chosen, since they depart from the vanilla implementation's approach.)

**Calling convention:** `gmdh.combi` takes `X, y` matrices directly (not a
formula + data.frame). The X matrix **must have column names** or it raises
`X matrix regressors must have names`.

**Conversion approach:** standardize X/y in Python → build a
`pandas.DataFrame` with named columns → convert via `pandas2ri` → `as.matrix()`
in R (reliably preserves `colnames`, unlike direct `numpy2ri` array
conversion, which drops them).

## Debugging History (chronological, for the R integration specifically)

1. Package name typo on install — resolved.
2. Wrong function name (`gmdh_pkg.gmdh()` doesn't exist) — resolved, switched
   to `gmdh_combi`.
3. Wrong calling convention (formula/data.frame vs. direct X,y matrices) —
   resolved, rewrote wrapper.
4. `numpy2rpy` doesn't exist in rpy2 — resolved, switched to
   `numpy2ri` + `localconverter`.
5. Stale `libRblas.dylib` path after R was upgraded from `4.5-arm64` to
   `4.6` — rpy2's compiled `.so` had the old path hardcoded from its original
   build. **Resolved as non-blocking**: rpy2 automatically falls back to ABI
   mode after the warning and works correctly afterward (confirmed via
   `ro.r('R.version.string')` returning `"R version 4.6.0 (2026-04-24)"`).
   This warning can be treated as noise.
6. `X matrix regressors must have names` — resolved via the
   pandas/`as.matrix()` conversion approach described above.
7. `R_X11.so` / missing `libSM.6.dylib` warning — cosmetic, caused by no
   XQuartz installed; confirmed harmless (`library(GMDHreg)` loads fine
   regardless).
8. Initial fixes assumed conda; corrected once clarified the project uses a
   plain Python `venv` (`algo_env/`). Recommended fix for `R_HOME`
   persistence: set it in Python itself
   (`os.environ.setdefault("R_HOME", "/Library/Frameworks/R.framework/Resources")`)
   at the top of `comparison.py`, rather than relying on shell exports or
   conda env vars, since this is portable regardless of how the script is
   launched.

## Current Open Problem (unresolved as of last session)

`comparison.py` reports:
```
⚠ rpy2 not installed. Reference comparison will be skipped.
```

This is printed by a generic `except ImportError:` fallback that does not log
the actual exception, so the true cause is currently unconfirmed.

**Key contradiction:** a direct terminal test in the (presumed) same activated
venv succeeds:
```bash
python -c "import rpy2.robjects as ro; print(ro.r('R.version.string'))"
# -> (after the harmless dlopen warning) "R version 4.6.0 (2026-04-24)"
```
This proves rpy2 imports and reaches R correctly in that shell. Yet
`comparison.py` still hits the `except ImportError:` branch. Possible causes,
not yet distinguished:

- `comparison.py` is being run by a **different Python** than the one just
  verified (different venv / no venv activated in that terminal / IDE run
  configuration using a different interpreter) — not yet conclusively ruled
  out.
- The real failure occurs **later**, inside `importr("GMDHreg")` (which can
  raise `rpy2.rinterface_lib.embedded.RRuntimeError`, not `ImportError`), and
  is being caught by an overly broad `except Exception` elsewhere, then
  reported under the same generic message.
- `R_HOME` is not actually set in the process that runs `comparison.py` (e.g.
  IDE "Run" button doesn't inherit exported shell variables).

## Next Diagnostic Steps (where the debugging session left off)

1. In the exact terminal/environment used to run `comparison.py`, confirm as
   two separate commands:
   ```bash
   which python
   python -c "import sys; print(sys.executable)"
   ```
   Both must resolve inside the project's `algo_env/bin/python`.

2. Make the exception handling in `gmdhref_wrapper.py` log the real error
   instead of swallowing it:
   ```python
   import sys
   try:
       import rpy2.robjects as ro
       ...
       R_AVAILABLE = True
   except ImportError as e:
       print(f"rpy2 import failed: {e}", file=sys.stderr)
       R_AVAILABLE = False
   ```

3. Add, at the very top of `comparison.py` (before any other imports):
   ```python
   import os
   os.environ.setdefault("R_HOME", "/Library/Frameworks/R.framework/Resources")
   ```

4. Re-run `comparison.py` and capture full stdout + stderr, especially
   anything printed immediately before the "rpy2 not installed" line.

## Other Active Project Challenges (broader, not R-integration-specific)

- Vanilla GMDH outperforming more complex variants — suspected to be linked
  to dataset properties (Dataset A lacks genuine interactions; Dataset B was
  built specifically to address this).
- `n_keep=8` is a fixed value, inconsistent with Ivakhnenko's fractional
  threshold scheme — motivates the Fibonacci threshold optimization work.
- Variable-dropping failures observed after the academically faithful
  refactor of vanilla GMDH; investigating whether Fibonacci threshold
  optimization resolves this.
- GMDH-CV overfitting fix: CV must be applied *within* the selection set
  after the external train/selection criterion is enforced — not as a
  replacement for it.

## Potential Future Work

- Additional basis function variants under consideration: Exponential,
  Wavelet.
- Continued critical evaluation of faithfulness to Ivakhnenko (1971) across
  all variants, with explicit documentation of departures.



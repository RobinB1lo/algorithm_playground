# GMDH Variants: Implementation, Benchmarking & Validation

A research implementation of the Group Method of Data Handling (GMDH)
algorithm and 13 custom variants, built for academic publication under
Professor Bonakdari. Follows Ivakhnenko's foundational 1971 paper as the
methodological reference point, with all departures from the original
algorithm explicitly documented rather than silently absorbed.

---

## Overview

GMDH is a self-organizing inductive modeling method: starting from pairs of
input variables, it builds successive layers of low-order polynomial
"neurons," keeps only the best-performing candidates at each layer (judged
against a held-out selection set), and stops once performance stops
improving. It was designed for regimes where the number of candidate
features is large relative to the number of observations — where standard
multiple regression struggles but GMDH's low-order partial polynomials
remain individually well-estimable.

This project has two goals, kept strictly separate:

1. **Faithfulness** — implement GMDH and its variants as close to
   Ivakhnenko's original formulation as possible, for publication and
   methodological comparison.
2. **Exploration** — investigate optimizations and extensions (e.g.
   Fibonacci-based threshold search) in clearly isolated modules that never silently alter the faithful implementations.

## What's Included

- **13 GMDH variants**, spanning evaluation-metric changes (AIC,
hierarchical ranking), cross-validation, fractional-polynomial neurons
(constrained and unconstrained), alternative basis functions (spline, RBF, sigmoid, Fourier), and architectural changes (look-back connections), plus combined variants.
- **4 baselines** for context: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting.
- **5 synthetic benchmark datasets**: Since these implementations are focused on improvement for environmental problems we create 4 synthetic environmental datasets and one low dimentional one
- **An independent R reference implementation** 

See the accompanying technical documentation for algorithm details,
variant-by-variant design notes, and dataset generation specifics.

## Setup

```bash
python3 -m venv algo_env
source algo_env/bin/activate
cd GMDH
pip install -r requirements.txt
```

## Running

```bash
python data_generation.py   # produces benchmark CSVs
python comparison.py        # evaluates all variants + baselines + R reference
```

## Status

Actively developed toward manuscript preparation under Professor
Bonakdari. See `current_instructions.md` and project notes for the current
state of open issues and next steps.
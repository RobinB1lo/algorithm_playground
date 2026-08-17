"""
Comprehensive Comparison Script: All GMDH Variants

Compares 13 GMDH implementations PLUS the official GMDHreg reference
across TWO synthetic datasets.

REFERENCE IMPLEMENTATION:
- GMDHreg: Official CRAN package by Villacorta Tilve (2024).
  Used to validate that the Python vanilla GMDH matches published behavior.
  https://CRAN.R-project.org/package=GMDHreg

DATASET A - "Wildfire" (low-dimensional, data-rich, no interactions):
    4 features, 500 samples, additively separable fractional-polynomial
    target. Represents a well-behaved nonlinear regression problem where
    GMDH acts as a general-purpose flexible regressor.

DATASET B - "High-Dimensional Short-Sequence" (GMDH's namesake regime):
    30 features, 60 samples, with genuine pairwise interactions among a
    subset of informative features, several redundant/collinear features,
    and several pure-noise irrelevant features for the selection mechanism
    to discard.

GMDH variants tested:
1. Vanilla GMDH (baseline error-based)
2. GMDH with AIC (information criterion)
3. GMDH with Hierarchical Ranking (multi-metric)
4. GMDH with Cross-Validation
5. GMDH with Constrained Fractional Polynomials
6. GMDH with Unconstrained Fractional Polynomials
7. GMDH with Spline Basis Functions
8. GMDH with RBF Basis Functions
9. GMDH with Sigmoid Basis Functions
10. GMDH with Trigonometric (Fourier) Basis Functions
11. GMDH with Look-Back Connections
12. GMDH with UFP + Hierarchical (combined)
13. GMDH with UFP + Hierarchical + CV (combined)

Plus baselines: Linear Regression, Ridge, Random Forest, Gradient Boosting
"""
#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("R_HOME", "/Library/Frameworks/R.framework/Resources")

# Diagnostic: prove which interpreter and R_HOME this process sees
print(f"[comparison.py] Python executable: {sys.executable}", file=sys.stderr)
print(f"[comparison.py] R_HOME: {os.environ.get('R_HOME', 'NOT SET')}", file=sys.stderr)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import time
import warnings

warnings.filterwarnings('ignore')

# Import all GMDH variants
from vanilla.gmdh import GMDH
from evaluation_metric_change.aic.gmdh_aic import GMDH_AIC
from evaluation_metric_change.hierarchical.hierarchical_gmdh import GMDH_Hierarchical
from cv_and_look_back.cv.cv_gmdh import GMDH_CV
from fractional_polynomials.cfp.cfp_gmdh import GMDH_Constrained
from fractional_polynomials.ufp.ufp_gmdh import GMDH_Unconstrained
from basis_change.spline.spline_basis_gmdh import GMDH_Spline
from basis_change.radial.rbf_basis_gmdh import GMDH_RBF
from basis_change.sigmoid.sigmoid_basis_gmdh import GMDH_Sigmoid
from basis_change.fourier.fourier_basis_gmdh import GMDH_Trig
from cv_and_look_back.look_back.look_back_gmdh import GMDH_LookBack
from fractional_polynomials.ufp_hierarchical.ufp_hierarchical_gmdh import GMDH_UFP_Hierarchical
from fractional_polynomials.ufp_hierarchical_cv.ufp_hierarchical_cv_gmdh import GMDH_UFP_Hierarchical_CV

# Import R reference wrapper (in regular/ directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'regular'))
try:
    # R_ERROR_MSG will contain the REAL reason if R_AVAILABLE is False
    from gmdhref_wrapper import GMDHreg_Wrapper, R_AVAILABLE, R_ERROR_MSG
except Exception as e:
    R_AVAILABLE = False
    GMDHreg_Wrapper = None
    R_ERROR_MSG = f"{type(e).__name__}: {e}"
    print(f"[comparison.py] Failed to import gmdhref_wrapper: {R_ERROR_MSG}", file=sys.stderr)

# =============================================================================
# DATASET A: Wildfire (low-dimensional, data-rich, no interactions)
# =============================================================================

def generate_synthetic_data(n_samples=500, random_state=42):
    """Generate synthetic non-linear wildfire behavior data.

    Features (X) represent environmental factors (All strictly positive):
    - X[:, 0] : Ambient Temperature (°C) [Range: 15 to 45]
    - X[:, 1] : Relative Humidity (%)   [Range: 5 to 60]
    - X[:, 2] : Wind Speed (km/h)       [Range: 5 to 40]
    - X[:, 3] : Fuel Moisture (%)       [Range: 2 to 25]

    True target (y): Rate of Fire Spread (meters/minute)

    True Mathematical Function:
    y = 0.2*Temp + (40 / Humidity) + 0.01*(Wind^2) + (15 / sqrt(FuelMoisture)) + noise
    Alternately written using fractional notation:
    y = 0.2*X0 + 40*X1^(-1) + 0.01*X2^(2) + 15*X3^(-0.5) + noise

    NOTE ON REGIME: this dataset is additively separable (no genuine pairwise
    interactions) and has ~125 samples per feature - a data-rich, low-
    dimensional regime. It benchmarks GMDH variants as general nonlinear
    regressors, not the "large dimensionality, short data sequence" regime
    the algorithm was originally designed for (see Dataset B below).
    """
    rng = np.random.default_rng(random_state)

    temp = rng.uniform(15, 45, size=n_samples)          # X[:, 0]
    humidity = rng.uniform(5, 60, size=n_samples)       # X[:, 1]
    wind_speed = rng.uniform(5, 40, size=n_samples)     # X[:, 2]
    fuel_moisture = rng.uniform(2, 25, size=n_samples)  # X[:, 3]

    X = np.column_stack([temp, humidity, wind_speed, fuel_moisture])

    y = (0.2 * X[:, 0]) + (40.0 / X[:, 1]) + (0.01 * X[:, 2]**2) + (15.0 / np.sqrt(X[:, 3]))

    noise = rng.normal(0, 0.5, size=n_samples)
    y = y + noise

    return X, y


# =============================================================================
# DATASET B: High-Dimensional Short-Sequence (GMDH's namesake regime)
# =============================================================================

def generate_high_dimensional_data(n_samples=60, n_informative=6, n_redundant=6,
                                     n_irrelevant=18, random_state=42):
    """Generate a synthetic dataset in the regime GMDH was explicitly designed
    for: large dimensionality relative to a short data sequence, with genuine
    pairwise interactions, redundant (collinear) features, and pure-noise
    irrelevant features for the selection mechanism to discard.

    Ivakhnenko (1970) motivates GMDH around exactly this regime: complex
    problems with large dimensionality when the data sequence is very short.
    A 10-input complete Kolmogorov-Gabor polynomial already has on the order
    of 200,000 terms - far more than can be estimated from a short sequence
    by ordinary regression. This dataset is built so that:

    - n_samples is deliberately small relative to total feature count
      (default 60 samples, 30 features - roughly 2 samples per feature),
      putting ordinary multiple regression on thin ice while GMDH's low-order
      partial polynomials remain individually well-estimable.
    - n_informative features drive the target through genuine PAIRWISE
      INTERACTIONS (not just additive univariate terms), giving the
      pairwise-combination mechanism something structurally necessary to
      discover, unlike Dataset A.
    - n_redundant features are noisy near-duplicates of informative features
      (collinear), testing whether selection prefers a "clean" informative
      variable over a redundant copy.
    - n_irrelevant features are pure noise, uncorrelated with the target -
      Ivakhnenko's "harmful" features, which a well-functioning selection
      criterion should learn to discard rather than incorporate.

    All features are kept strictly positive to remain compatible with the
    fractional-polynomial (CFP/UFP) variants elsewhere in this suite.

    Parameters:
    -----------
    n_samples : int
        Total number of observations. Kept small relative to n_features.
    n_informative : int
        Number of features that genuinely drive the target.
    n_redundant : int
        Number of noisy near-duplicates of informative features.
    n_irrelevant : int
        Number of pure-noise, uninformative features.
    random_state : int, optional
        Random seed.

    Returns:
    --------
    X : np.ndarray, shape (n_samples, n_informative + n_redundant + n_irrelevant)
    y : np.ndarray, shape (n_samples,)
    """
    rng = np.random.default_rng(random_state)
    n_features = n_informative + n_redundant + n_irrelevant

    # 1. Informative features - strictly positive, moderate range
    X_informative = rng.uniform(1.0, 10.0, size=(n_samples, n_informative))

    # 2. Redundant features - noisy near-duplicates of a randomly chosen
    #    informative feature each, kept strictly positive via clipping
    redundant_sources = rng.integers(0, n_informative, size=n_redundant)
    redundant_noise_scale = 0.5
    X_redundant = np.column_stack([
        np.clip(
            X_informative[:, src] + rng.normal(0, redundant_noise_scale, size=n_samples),
            0.1, None
        )
        for src in redundant_sources
    ])

    # 3. Irrelevant features - pure positive noise, uncorrelated with target
    X_irrelevant = rng.uniform(1.0, 10.0, size=(n_samples, n_irrelevant))

    # Assemble full feature matrix; shuffle column order so "informative"
    # features are not trivially the first columns
    X_full = np.column_stack([X_informative, X_redundant, X_irrelevant])
    col_order = rng.permutation(n_features)
    X = X_full[:, col_order]

    # Track where each informative feature ended up post-shuffle, for
    # constructing y from genuine interactions among them
    original_informative_idx = np.arange(n_informative)
    new_positions = np.argsort(col_order)
    informative_new_idx = new_positions[original_informative_idx]

    a, b, c, d, e, f = [X[:, informative_new_idx[k]] for k in range(6)]

    # 4. True target: genuine pairwise interactions + one univariate
    #    nonlinear term, so the pairwise-combination mechanism has real
    #    interaction structure to discover, unlike Dataset A.
    y = (
        0.5 * a * b          # genuine interaction: a and b only matter together
        + 0.3 * c * d        # genuine interaction: c and d only matter together
        + 0.05 * e**2         # univariate nonlinear term
        - 0.4 * f             # univariate linear term
    )

    noise = rng.normal(0, y.std() * 0.05, size=n_samples)
    y = y + noise

    return X, y


# =============================================================================
# Shared evaluation / plotting / reporting logic
# =============================================================================

def build_models():
    """Fresh, unfitted model instances - called once per dataset."""
    models = [
        # GMDH Variants - Basic
        (GMDH(n_keep=8, max_layers=10, patience=1, random_state=42), "Vanilla GMDH"),
        (GMDH_AIC(n_keep=8, max_layers=10, patience=1, random_state=42), "GMDH-AIC"),
        (GMDH_Hierarchical(n_keep=8, max_layers=10, patience=1, random_state=42), "GMDH-Hierarchical"),
        (GMDH_CV(n_keep=8, max_layers=10, k_folds=5, patience=1, random_state=42), "GMDH-CV"),

        # GMDH Variants - Fractional Polynomials
        (GMDH_Constrained(n_keep=8, max_layers=10, penalty_type='l2', penalty_lambda=0.01, random_state=42), "GMDH-CFP-L2"),
        (GMDH_Constrained(n_keep=8, max_layers=10, penalty_type='l1', penalty_lambda=0.01, random_state=42), "GMDH-CFP-L1"),
        (GMDH_Unconstrained(n_keep=8, max_layers=10, k_folds=5, penalty_type='l2', penalty_lambda=0.01, random_state=42), "GMDH-UFP-L2"),
        (GMDH_Unconstrained(n_keep=8, max_layers=10, k_folds=5, penalty_type='l1', penalty_lambda=0.01, random_state=42), "GMDH-UFP-L1"),

        # GMDH Variants - Alternative Basis Functions
        (GMDH_Spline(n_keep=8, max_layers=10, ridge=1e-6, n_knots=3, k=3, patience=1, random_state=42), "GMDH-Spline"),
        (GMDH_RBF(n_keep=8, max_layers=10, ridge=1e-6, n_centers=5, gamma=5.0, patience=1, random_state=42), "GMDH-RBF"),
        (GMDH_Sigmoid(n_keep=8, max_layers=10, ridge=1e-6, n_centers=5, k=8.0, patience=1, random_state=42), "GMDH-Sigmoid"),
        (GMDH_Trig(n_keep=8, max_layers=10, ridge=1e-6, n_harmonics=3, patience=1, random_state=42), "GMDH-Fourier"),

        # GMDH Variants - Architecture
        (GMDH_LookBack(n_keep=8, max_layers=10, ridge=1e-6, patience=1, lookback_scope='all', random_state=42), "GMDH-LookBack"),

        # GMDH Variants - Combined
        (GMDH_UFP_Hierarchical(n_keep=8, max_layers=10, penalty_type='l2', penalty_lambda=0.01, patience=1, random_state=42), "GMDH-UFP-Hierarchical"),
        (GMDH_UFP_Hierarchical_CV(n_keep=8, max_layers=10, penalty_type='l2', penalty_lambda=0.01, k_folds=5, patience=1, random_state=42), "GMDH-UFP-Hierarchical-CV"),
    ]

    # Reference implementation (official CRAN GMDHreg if available)
    if R_AVAILABLE and GMDHreg_Wrapper is not None:
        try:
            models.insert(1, (
                GMDHreg_Wrapper(criteria="test", G=2, random_state=42),
                "★ GMDHreg (CRAN Reference)"
            ))
        except Exception as e:
            print(f"⚠ Reference instantiation failed: {type(e).__name__}: {e}", file=sys.stderr)

    # Baselines
    models.extend([
        (LinearRegression(), "Linear Regression"),
        (Ridge(alpha=1.0), "Ridge Regression"),
        (RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10), "Random Forest"),
        (GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3), "Gradient Boosting"),
    ])

    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model, return metrics."""
    start = time.time()
    try:
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        elapsed = time.time() - start

        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        return {
            'name': model_name,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'r2_test': r2_test,
            'time': elapsed,
            'y_pred_test': y_pred_test,
            'status': 'success'
        }
    except Exception as e:
        return {
            'name': model_name,
            'rmse_test': np.inf,
            'mae_test': np.inf,
            'r2_test': -np.inf,
            'time': 0,
            'y_pred_test': None,
            'status': f'failed: {str(e)[:50]}'
        }


def run_comparison(X, y, dataset_label, output_suffix, true_function_desc):
    """Run the full model comparison on one dataset, print a labeled report,
    and save a labeled plot."""
    print("=" * 140)
    print(f"COMPREHENSIVE GMDH COMPARISON: {dataset_label}")
    print("=" * 140)

    print(f"\n1. Dataset: {dataset_label}")
    print(f"   Data shape: {X.shape}")
    print(f"   True function: {true_function_desc}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"   Samples-to-features ratio (train): {X_train.shape[0] / X_train.shape[1]:.2f}")

    print("\n2. Training all models...")
    models = build_models()

    results = []
    for model, name in models:
        print(f"   {name:<45}", end=" ", flush=True)
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)
        status_str = "✓" if result['status'] == 'success' else f"✗ ({result['status']})"
        print(f"{status_str} (time: {result['time']:.2f}s)")

    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] != 'success']

    print(f"\n3. Results Summary (All Models) - {dataset_label}")
    print("=" * 140)
    print(f"{'Model':<45} {'Test RMSE':<15} {'Test MAE':<15} {'Test R²':<15} {'Time (s)':<12}")
    print("-" * 140)

    for result in successful_results:
        print(f"{result['name']:<45} {result['rmse_test']:<15.4f} "
              f"{result['mae_test']:<15.4f} {result['r2_test']:<15.4f} {result['time']:<12.3f}")

    if failed_results:
        print("\n❌ Failed Models:")
        for result in failed_results:
            print(f"   {result['name']}: {result['status']}")

    non_vanilla_successful_results = successful_results.copy()
    # Remove vanilla GMDH (position depends on whether reference is present)
    if R_AVAILABLE and len(non_vanilla_successful_results) > 0 and "★" in non_vanilla_successful_results[0]['name']:
        non_vanilla_successful_results.pop(1)  # Remove vanilla after reference
    else:
        non_vanilla_successful_results.pop(0)  # Remove vanilla as first

    best_idx = np.argmin([r['rmse_test'] for r in non_vanilla_successful_results])
    print("-" * 140)
    print(f"Best model (lowest test RMSE): {non_vanilla_successful_results[best_idx]['name']}")

    # Validation: compare reference to vanilla if both present
    reference_result = None
    vanilla_result = None
    for r in successful_results:
        if "★ GMDHreg" in r['name']:
            reference_result = r
        elif r['name'] == "Vanilla GMDH":
            vanilla_result = r

    if reference_result and vanilla_result:
        rmse_diff = abs(reference_result['rmse_test'] - vanilla_result['rmse_test'])
        rmse_pct_diff = 100 * rmse_diff / vanilla_result['rmse_test']
        print(f"\n🔍 VALIDATION: GMDHreg (CRAN) vs. Vanilla GMDH (Python)")
        print(f"   GMDHreg RMSE:     {reference_result['rmse_test']:.4f}")
        print(f"   Vanilla RMSE:     {vanilla_result['rmse_test']:.4f}")
        print(f"   Absolute diff:    {rmse_diff:.4f}")
        print(f"   Percent diff:     {rmse_pct_diff:.2f}%")
        if rmse_pct_diff < 10:
            print(f"   ✓ Implementations match well (< 10% difference)")
        elif rmse_pct_diff < 20:
            print(f"   ⚠ Implementations differ moderately (10–20%)")
        else:
            print(f"   ✗ Significant difference (> 20%) — investigate algorithm details")
        print(f"   NOTE: GMDHreg uses its own internal train/validation split,")
        print(f"         which differs from vanilla GMDH's 50/50 split.")

    print("=" * 140)

    print(f"\n4. GMDH Variant Details - {dataset_label}")
    gmdh_results = [r for r in successful_results if 'GMDH' in r['name'] and r['name'] != 'Gradient Boosting']
    for result in gmdh_results:
        print(f"\n   {result['name']}:")
        print(f"      Test RMSE: {result['rmse_test']:.4f}")
        print(f"      Test R²: {result['r2_test']:.4f}")
        print(f"      Time: {result['time']:.3f}s")

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(f'GMDH Variants Comparison — {dataset_label}', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    names = [r['name'] for r in successful_results]
    rmses = [r['rmse_test'] for r in successful_results]
    colors = ['#FF6B6B' if '★' in name else '#2ecc71' if 'GMDH' in name else '#3498db' for name in names]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, rmses, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Test RMSE (lower is better)')
    ax.set_title('RMSE Comparison')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    ax = axes[0, 1]
    r2s = [r['r2_test'] for r in successful_results]
    ax.barh(y_pos, r2s, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Test R² (higher is better)')
    ax.set_title('R² Comparison')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    ax = axes[0, 2]
    times = [r['time'] for r in successful_results]
    ax.barh(y_pos, times, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    ax = axes[1, 0]
    gmdh_names = [r['name'] for r in gmdh_results]
    gmdh_rmses = [r['rmse_test'] for r in gmdh_results]
    gmdh_colors = ['#FF6B6B' if '★' in name else '#2ecc71' for name in gmdh_names]
    gmdh_y_pos = np.arange(len(gmdh_names))
    ax.barh(gmdh_y_pos, gmdh_rmses, color=gmdh_colors, alpha=0.7)
    ax.set_yticks(gmdh_y_pos)
    ax.set_yticklabels(gmdh_names, fontsize=8)
    ax.set_xlabel('Test RMSE')
    ax.set_title('GMDH Variants - RMSE Only')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    ax = axes[1, 1]
    gmdh_r2s = [r['r2_test'] for r in gmdh_results]
    ax.barh(gmdh_y_pos, gmdh_r2s, color=gmdh_colors, alpha=0.7)
    ax.set_yticks(gmdh_y_pos)
    ax.set_yticklabels(gmdh_names, fontsize=8)
    ax.set_xlabel('Test R²')
    ax.set_title('GMDH Variants - R² Only')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    ax = axes[1, 2]
    best_result = non_vanilla_successful_results[best_idx]
    ax.scatter(y_test, best_result['y_pred_test'], alpha=0.6, s=30)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Best Model: {best_result["name"]}\n(R² = {best_result["r2_test"]:.4f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_filename = f'gmdh_comprehensive_comparison_{output_suffix}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n5. Plots saved to '{output_filename}'")

    non_vanilla_gmdh_results = gmdh_results.copy()
    if R_AVAILABLE and len(non_vanilla_gmdh_results) > 1 and "★" in non_vanilla_gmdh_results[0]['name']:
        non_vanilla_gmdh_results.pop(0)
    else:
        non_vanilla_gmdh_results.pop(0)

    if len(non_vanilla_gmdh_results) > 0:
        best_gmdh_idx = np.argmin([r['rmse_test'] for r in non_vanilla_gmdh_results])
        print(f"\n6. Summary Statistics - {dataset_label}")
        print(f"   Best GMDH variant (excluding vanilla): {non_vanilla_gmdh_results[best_gmdh_idx]['name']}")

    print(f"   Best overall model: {best_result['name']}")
    print(f"   GMDH variants tested: {len(gmdh_results)}")
    print(f"   Baseline models tested: {len([r for r in successful_results if 'GMDH' not in r['name']])}")
    print(f"   Models failed: {len(failed_results)}")

    print("\n" + "=" * 140 + "\n")

    return successful_results, gmdh_results


def main():
    print("Checking for reference implementation...")
    if not R_AVAILABLE:
        print(f"⚠ R reference unavailable: {R_ERROR_MSG}")
        print("  Reference comparison will be skipped.\n")
    else:
        print("✓ R reference wrapper loaded successfully.\n")
    print("Checking for reference implementation...")

    # ---- Comparison 1: Wildfire ----
    X_a, y_a = generate_synthetic_data(n_samples=500, random_state=42)
    run_comparison(
        X_a, y_a,
        dataset_label="Dataset A - Wildfire (Low-Dimensional, Data-Rich)",
        output_suffix="dataset_a_wildfire",
        true_function_desc="y = 0.2*x0 + 40/x1 + 0.01*x2^2 + 15/sqrt(x3) + noise"
    )

    # ---- Comparison 2: High-Dimensional ----
    X_b, y_b = generate_high_dimensional_data(
        n_samples=60, n_informative=6, n_redundant=6, n_irrelevant=18, random_state=42
    )
    run_comparison(
        X_b, y_b,
        dataset_label="Dataset B - High-Dimensional Short-Sequence (GMDH's Target Regime)",
        output_suffix="dataset_b_highdim",
        true_function_desc=(
            "y = 0.5*a*b + 0.3*c*d + 0.05*e^2 - 0.4*f + noise "
            "(genuine pairwise interactions; redundant & irrelevant features mixed in)"
        )
    )


if __name__ == "__main__":
    main()
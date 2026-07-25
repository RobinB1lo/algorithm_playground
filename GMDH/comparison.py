"""
Comprehensive Comparison Script: All GMDH Variants

Compares 9 GMDH implementations:
1. Vanilla GMDH (baseline error-based)
2. GMDH with AIC (information criterion)
3. GMDH with Hierarchical Ranking (multi-metric)
4. GMDH with Cross-Validation
5. GMDH with Constrained Fractional Polynomials
6. GMDH with Unconstrained Fractional Polynomials
7. GMDH with Spline Basis Functions
8. GMDH with UFP + Hierarchical (combined)
9. GMDH with UFP + Hierarchical + CV (combined)

Plus baselines: Linear Regression, Ridge, Random Forest, Gradient Boosting
"""

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
from gmdh import GMDH
from gmdh_aic import GMDH_AIC
from hierarchical_gmdh import GMDH_Hierarchical
from cv_gmdh import GMDH_CV
from cfp_gmdh import GMDH_Constrained
from ufp_gmdh import GMDH_Unconstrained
from spline_basis_gmdh import GMDH_Spline
from ufp_hierarchical_gmdh import GMDH_UFP_Hierarchical
from ufp_hierarchical_cv_gmdh import GMDH_UFP_Hierarchical_CV


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
    """
    rng = np.random.default_rng(random_state)
    
    # 1. Generate realistic, strictly positive weather & fuel parameters
    temp = rng.uniform(15, 45, size=n_samples)          # X[:, 0]
    humidity = rng.uniform(5, 60, size=n_samples)       # X[:, 1]
    wind_speed = rng.uniform(5, 40, size=n_samples)     # X[:, 2]
    fuel_moisture = rng.uniform(2, 25, size=n_samples)  # X[:, 3]
    
    X = np.column_stack([temp, humidity, wind_speed, fuel_moisture])
    
    # 2. Compute Rate of Spread (y) using explicit fractional polynomial terms
    # - Humidity has a pure inverse relationship (X1^-1)
    # - Wind Speed has a quadratic relationship (X2^2)
    # - Fuel Moisture has an inverse square-root relationship (X3^-0.5)
    y = (0.2 * X[:, 0]) + (40.0 / X[:, 1]) + (0.01 * X[:, 2]**2) + (15.0 / np.sqrt(X[:, 3]))
    
    # 3. Add realistic normal variance (noise)
    noise = rng.normal(0, 0.5, size=n_samples)
    y = y + noise
    
    return X, y


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


def main():
    print("=" * 90)
    print("COMPREHENSIVE GMDH COMPARISON: All Variants + Baselines")
    print("=" * 90)
    
    # Generate synthetic data
    print("\n1. Generating synthetic nonlinear data...")
    X, y = generate_synthetic_data(n_samples=500, random_state=42)
    print(f"   Data shape: {X.shape}")
    print(f"   True function: y = 0.2*x0 + 40/x1 + 0.01*x2^2 + 15/sqrt(x3) + noise")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    
    # Initialize all models
    print("\n2. Training all models...")
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
        
        # GMDH Variants - Spline Basis (NEW)
        (GMDH_Spline(n_keep=8, max_layers=10, ridge=1e-6, n_knots=3, k=3, patience=1, random_state=42), "GMDH-Spline"),
        
        # GMDH Variants - Combined
        (GMDH_UFP_Hierarchical(n_keep=8, max_layers=10, penalty_type='l2', penalty_lambda=0.01, patience=1, random_state=42), "GMDH-UFP-Hierarchical"),
        (GMDH_UFP_Hierarchical_CV(n_keep=8, max_layers=10, penalty_type='l2', penalty_lambda=0.01, k_folds=5, patience=1, random_state=42), "GMDH-UFP-Hierarchical-CV"),
        
        # Baselines
        (LinearRegression(), "Linear Regression"),
        (Ridge(alpha=1.0), "Ridge Regression"),
        (RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10), "Random Forest"),
        (GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3), "Gradient Boosting"),
    ]
    
    results = []
    for model, name in models:
        print(f"   {name:<35}", end=" ")
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)
        status_str = "✓" if result['status'] == 'success' else f"✗ ({result['status']})"
        print(f"{status_str} (time: {result['time']:.2f}s)")
    
    # Filter successful results for display
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] != 'success']
    
    # Print results
    print("\n3. Results Summary (All Models)")
    print("=" * 120)
    print(f"{'Model':<35} {'Test RMSE':<15} {'Test MAE':<15} {'Test R²':<15} {'Time (s)':<12}")
    print("-" * 120)
    
    for result in successful_results:
        print(f"{result['name']:<35} {result['rmse_test']:<15.4f} "
              f"{result['mae_test']:<15.4f} {result['r2_test']:<15.4f} {result['time']:<12.3f}")
    
    if failed_results:
        print("\n❌ Failed Models:")
        for result in failed_results:
            print(f"   {result['name']}: {result['status']}")
    
    non_vanilla_successful_results = successful_results.copy()
    non_vanilla_successful_results.pop(0)
    best_idx = np.argmin([r['rmse_test'] for r in non_vanilla_successful_results])
    print("-" * 120)
    print(f"Best model (lowest test RMSE): {non_vanilla_successful_results[best_idx]['name']}")
    print("=" * 120)
    
    # GMDH-specific details
    print("\n4. GMDH Variant Details")
    gmdh_variants = [r for r in successful_results if 'GMDH' in r['name'] and r['name'] != 'Gradient Boosting']
    for result in gmdh_variants:
        print(f"\n   {result['name']}:")
        print(f"      Test RMSE: {result['rmse_test']:.4f}")
        print(f"      Test R²: {result['r2_test']:.4f}")
        print(f"      Time: {result['time']:.3f}s")
    
    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('GMDH Variants Comparison on Synthetic Nonlinear Data', fontsize=16, fontweight='bold')
    
    # Plot 1: RMSE Comparison (All)
    ax = axes[0, 0]
    names = [r['name'] for r in successful_results]
    rmses = [r['rmse_test'] for r in successful_results]
    colors = ['#2ecc71' if 'GMDH' in name else '#3498db' for name in names]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, rmses, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Test RMSE (lower is better)')
    ax.set_title('RMSE Comparison')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Plot 2: R² Comparison (All)
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
    
    # Plot 3: Time Comparison
    ax = axes[0, 2]
    times = [r['time'] for r in successful_results]
    ax.barh(y_pos, times, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Plot 4: GMDH Variants Only - RMSE
    ax = axes[1, 0]
    gmdh_results = [r for r in successful_results if 'GMDH' in r['name'] and r['name'] != 'Gradient Boosting']
    gmdh_names = [r['name'] for r in gmdh_results]
    gmdh_rmses = [r['rmse_test'] for r in gmdh_results]
    gmdh_y_pos = np.arange(len(gmdh_names))
    ax.barh(gmdh_y_pos, gmdh_rmses, color='#2ecc71', alpha=0.7)
    ax.set_yticks(gmdh_y_pos)
    ax.set_yticklabels(gmdh_names, fontsize=8)
    ax.set_xlabel('Test RMSE')
    ax.set_title('GMDH Variants - RMSE Only')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Plot 5: GMDH Variants Only - R²
    ax = axes[1, 1]
    gmdh_r2s = [r['r2_test'] for r in gmdh_results]
    ax.barh(gmdh_y_pos, gmdh_r2s, color='#2ecc71', alpha=0.7)
    ax.set_yticks(gmdh_y_pos)
    ax.set_yticklabels(gmdh_names, fontsize=8)
    ax.set_xlabel('Test R²')
    ax.set_title('GMDH Variants - R² Only')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Plot 6: Best Model Predictions vs Actual
    ax = axes[1, 2]
    best_result = non_vanilla_successful_results[best_idx]
    ax.scatter(y_test, best_result['y_pred_test'], alpha=0.6, s=30)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Best Model: {best_result["name"]}\n(R² = {best_result["r2_test"]:.4f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmdh_comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n5. Plots saved to 'gmdh_comprehensive_comparison.png'")
    
    # Summary statistics
    non_vanilla_gmdh_results = gmdh_results.copy()
    non_vanilla_gmdh_results.pop(0)
    best_gmdh_idx = np.argmin([r['rmse_test'] for r in non_vanilla_gmdh_results])
    print("\n6. Summary Statistics")
    print(f"   Best GMDH variant: {non_vanilla_gmdh_results[best_gmdh_idx]['name']}")
    print(f"   Best overall model: {best_result['name']}")
    print(f"   GMDH variants tested: {len(gmdh_results)}")
    print(f"   Baseline models tested: {len([r for r in successful_results if 'GMDH' not in r['name']])}")
    print(f"   Models failed: {len(failed_results)}")
    print(f"\n   New variants:")
    print(f"   - GMDH-Spline: RMSE={[r['rmse_test'] for r in gmdh_results if r['name'] == 'GMDH-Spline'][0]:.4f}")
    print(f"   - GMDH-UFP-Hierarchical: RMSE={[r['rmse_test'] for r in gmdh_results if r['name'] == 'GMDH-UFP-Hierarchical'][0]:.4f}")
    print(f"   - GMDH-UFP-Hierarchical-CV: RMSE={[r['rmse_test'] for r in gmdh_results if r['name'] == 'GMDH-UFP-Hierarchical-CV'][0]:.4f}")
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
import numpy as np
from cfp_gmdh import GMDH_CFP
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    """Create a synthetic dataset and run GMDH-CFP."""
    
    print("=" * 80)
    print("GMDH-CFP (Constrained Fractional Polynomials): Synthetic Dataset Example")
    print("=" * 80)
    
    # === Generate synthetic data ===
    np.random.seed(42)
    n_samples = 200
    
    # Create features with positive values (good for fractional powers)
    x0 = np.random.uniform(0.5, 5, n_samples)
    x1 = np.random.uniform(0.5, 5, n_samples)
    x2 = np.random.uniform(0.5, 5, n_samples)
    
    # Create target with power-law and fractional polynomial relationships
    # y = 2*sqrt(x0) + 3/x1 + 2*x2^0.5 + noise
    # These kinds of relationships benefit from fractional polynomial modeling
    y = (
        2.0 * np.sqrt(x0) +           # Power 0.5
        3.0 / x1 +                     # Power -1 (inverse)
        2.0 * np.power(x2, 0.5) +      # Power 0.5
        np.random.normal(0, 0.5, n_samples)  # noise
    )
    
    X = np.column_stack([x0, x1, x2])
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTrue underlying relationship:")
    print(f"  y = 2*√(x0) + 3/x1 + 2*√(x2) + noise")
    print(f"\nNote: Fractional polynomials excel at modeling power laws,")
    print(f"      inverse relationships, and square root functions!")
    
    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # === Fit GMDH-CFP ===
    print("\n" + "=" * 80)
    print("Training GMDH-CFP Network...")
    print("=" * 80)
    
    # Define fractional powers to search over
    # Includes common useful powers: negative (inverse), fractional (roots), integer
    powers = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    gmdh_cfp = GMDH_CFP(
        n_keep=3,
        max_layers=5,
        ridge=1e-6,
        training_split=0.5,
        patience=1,
        powers_a=powers,
        powers_b=powers,
        random_state=42
    )
    
    gmdh_cfp.fit(X_train, y_train)
    
    # === Print learned equation ===
    gmdh_cfp.print_equation(feature_names=['x0', 'x1', 'x2'])
    
    # === Evaluate ===
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    y_train_pred = gmdh_cfp.predict(X_train)
    y_test_pred = gmdh_cfp.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTrain RMSE: {train_rmse:.6f}")
    print(f"Test RMSE:  {test_rmse:.6f}")
    print(f"Train R²:   {train_r2:.6f}")
    print(f"Test R²:    {test_r2:.6f}")
    
    print(f"\n--- Constrained Fractional Polynomials ---")
    print(f"CFP extends GMDH by allowing fractional and negative powers:")
    print(f"  Power 2:    x² (quadratic)")
    print(f"  Power 1:    x (linear)")
    print(f"  Power 0.5:  √x (square root)")
    print(f"  Power 0:    1 (constant)")
    print(f"  Power -0.5: 1/√x (inverse square root)")
    print(f"  Power -1:   1/x (inverse)")
    print(f"  Power -2:   1/x² (inverse quadratic)")
    print(f"\nThis flexibility allows GMDH-CFP to discover power-law relationships,")
    print(f"scaling laws, and inverse dependencies that standard polynomial GMDH")
    print(f"cannot easily capture.")
    
    # === Sample predictions ===
    print("\n" + "=" * 80)
    print("Sample Predictions (Test Set)")
    print("=" * 80)
    
    for i in range(min(5, len(X_test))):
        print(f"\nSample {i+1}:")
        print(f"  Input:  x0={X_test[i, 0]:.4f}, x1={X_test[i, 1]:.4f}, x2={X_test[i, 2]:.4f}")
        print(f"  True:   y={y_test.iloc[i]:.4f}" if hasattr(y_test, 'iloc') else f"  True:   y={y_test[i]:.4f}")
        print(f"  Pred:   y={y_test_pred[i]:.4f}")


if __name__ == "__main__":
    main()
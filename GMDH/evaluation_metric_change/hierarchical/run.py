import numpy as np
from hierarchical_gmdh import GMDH_Hierarchical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    """Create a synthetic dataset and run GMDH-Hierarchical."""
    
    print("=" * 80)
    print("GMDH-Hierarchical: Synthetic Dataset Example")
    print("=" * 80)
    
    # === Generate synthetic data ===
    np.random.seed(42)
    n_samples = 200
    
    # Create features
    x0 = np.random.uniform(-5, 5, n_samples)
    x1 = np.random.uniform(-5, 5, n_samples)
    x2 = np.random.uniform(-5, 5, n_samples)
    
    # Create target: y = 2*x0 + 3*x1^2 - 0.5*x0*x1 + 0.1*x2 + noise
    y = 2*x0 + 3*(x1**2) - 0.5*x0*x1 + 0.1*x2 + np.random.normal(0, 2, n_samples)
    
    X = np.column_stack([x0, x1, x2])
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTrue underlying relationship:")
    print(f"  y = 2*x0 + 3*x1² - 0.5*x0*x1 + 0.1*x2 + noise")
    
    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # === Fit GMDH-Hierarchical ===
    print("\n" + "=" * 80)
    print("Training GMDH-Hierarchical Network...")
    print("=" * 80)
    
    gmdh_hier = GMDH_Hierarchical(
        n_keep=3,
        max_layers=5,
        ridge=1e-6,
        training_split=0.5,
        patience=1,
        random_state=42
    )
    
    gmdh_hier.fit(X_train, y_train)
    
    # === Print learned equation ===
    gmdh_hier.print_equation(feature_names=['x0', 'x1', 'x2'])
    
    # === Evaluate ===
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    y_train_pred = gmdh_hier.predict(X_train)
    y_test_pred = gmdh_hier.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTrain RMSE: {train_rmse:.6f}")
    print(f"Test RMSE:  {test_rmse:.6f}")
    print(f"Train R²:   {train_r2:.6f}")
    print(f"Test R²:    {test_r2:.6f}")
    
    print(f"\n--- Hierarchical Selection Insight ---")
    print(f"Neurons in each layer are selected using a hierarchical (lexicographic)")
    print(f"ordering of multiple criteria:")
    print(f"  1. RMSE (Root Mean Squared Error) — primary criterion")
    print(f"  2. MAE (Mean Absolute Error) — tie-breaker 1")
    print(f"  3. -R² (Negative R²) — tie-breaker 2")
    print(f"  4. AIC (Akaike Information Criterion) — tie-breaker 3")
    print(f"  5. -Adjusted R² (Negative Adjusted R²) — tie-breaker 4")
    print(f"\nThis multi-metric approach balances predictive accuracy, robustness,")
    print(f"model complexity, and generalization better than a single metric.")
    
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
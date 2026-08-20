import numpy as np
from spline_basis_gmdh import GMDH_Spline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    """Create a synthetic dataset and run GMDH-Spline."""
    
    print("=" * 80)
    print("GMDH-Spline: Synthetic Dataset Example")
    print("=" * 80)
    
    # === Generate synthetic data ===
    np.random.seed(42)
    n_samples = 200
    
    # Create features
    x0 = np.random.uniform(-5, 5, n_samples)
    x1 = np.random.uniform(-5, 5, n_samples)
    x2 = np.random.uniform(-5, 5, n_samples)
    
    # Create target with nonlinear relationship (good for splines)
    # y = sin(x0) + cos(x1) + 0.5*x2 + noise
    y = (np.sin(x0) + np.cos(x1) + 0.5*x2 + 
         np.random.normal(0, 0.5, n_samples))
    
    X = np.column_stack([x0, x1, x2])
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTrue underlying relationship:")
    print(f"  y = sin(x0) + cos(x1) + 0.5*x2 + noise")
    print(f"\nNote: Splines are great for learning smooth nonlinear functions!")
    
    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # === Fit GMDH-Spline ===
    print("\n" + "=" * 80)
    print("Training GMDH-Spline Network...")
    print("=" * 80)
    
    gmdh_spline = GMDH_Spline(
        n_keep=3,
        max_layers=5,
        ridge=1e-6,
        training_split=0.5,
        patience=1,
        n_knots=3,      # 3 interior knots for smoother fitting
        k=3,            # cubic splines
        random_state=42
    )
    
    gmdh_spline.fit(X_train, y_train)
    
    # === Print learned equations ===
    gmdh_spline.print_equation(feature_names=['x0', 'x1', 'x2'])
    
    # === Evaluate ===
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    y_train_pred = gmdh_spline.predict(X_train)
    y_test_pred = gmdh_spline.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTrain RMSE: {train_rmse:.6f}")
    print(f"Test RMSE:  {test_rmse:.6f}")
    print(f"Train R²:   {train_r2:.6f}")
    print(f"Test R²:    {test_r2:.6f}")
    
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
import numpy as np
from fourier_basis_gmdh import GMDH_Trig
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    """Create a synthetic dataset and run GMDH-Fourier."""
    
    print("=" * 80)
    print("GMDH-Fourier (Trigonometric): Synthetic Dataset Example")
    print("=" * 80)
    
    # === Generate synthetic data ===
    np.random.seed(42)
    n_samples = 200
    
    # Create features
    x0 = np.random.uniform(-np.pi, np.pi, n_samples)
    x1 = np.random.uniform(-np.pi, np.pi, n_samples)
    x2 = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Create target with periodic behavior (good for Fourier/Trigonometric)
    # y = A*sin(x0) + B*cos(2*x1) + C*sin(x0)*cos(x1) + linear term + noise
    y = (
        2.0 * np.sin(x0) +                      # First harmonic on x0
        1.5 * np.cos(2 * x1) +                  # Second harmonic on x1
        1.0 * np.sin(x0) * np.cos(x1) +         # Interaction between x0 and x1
        0.3 * x2 +                              # Linear trend
        np.random.normal(0, 0.4, n_samples)     # noise
    )
    
    X = np.column_stack([x0, x1, x2])
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTrue underlying relationship:")
    print(f"  y = 2*sin(x0) + 1.5*cos(2*x1) + 1*sin(x0)*cos(x1) + 0.3*x2 + noise")
    print(f"\nNote: Fourier/trigonometric basis excels at learning periodic patterns!")
    
    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # === Fit GMDH-Fourier ===
    print("\n" + "=" * 80)
    print("Training GMDH-Fourier Network...")
    print("=" * 80)
    
    gmdh_trig = GMDH_Trig(
        n_keep=3,
        max_layers=5,
        ridge=1e-6,
        training_split=0.5,
        patience=1,
        n_harmonics=3,  # 3 sin/cos harmonic pairs per input (frequencies 1, 2, 3)
        random_state=42
    )
    
    gmdh_trig.fit(X_train, y_train)
    
    # === Print learned equations ===
    gmdh_trig.print_equation(feature_names=['x0', 'x1', 'x2'])
    
    # === Evaluate ===
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    y_train_pred = gmdh_trig.predict(X_train)
    y_test_pred = gmdh_trig.predict(X_test)
    
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
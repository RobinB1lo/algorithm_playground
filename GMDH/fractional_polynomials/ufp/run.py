import numpy as np
from ufp_gmdh import GMDH_UFP
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    """Create a synthetic dataset and run GMDH-UFP."""
    
    print("=" * 80)
    print("GMDH-UFP (Unconstrained Fractional Polynomials): Synthetic Dataset Example")
    print("=" * 80)
    
    # === Generate synthetic data ===
    np.random.seed(42)
    n_samples = 200
    
    # Create features with positive values
    x0 = np.random.uniform(0.5, 5, n_samples)
    x1 = np.random.uniform(0.5, 5, n_samples)
    x2 = np.random.uniform(0.5, 5, n_samples)
    
    # Create target with mixed power-law relationships
    # y = 1.5*x0^1.5 + 0.8*x1^0.5 + 2.0*x2^(-0.5) + noise
    # UFP should discover these optimal powers automatically
    y = (
        1.5 * np.power(x0, 1.5) +
        0.8 * np.power(x1, 0.5) +
        2.0 * np.power(x2, -0.5) +
        np.random.normal(0, 0.8, n_samples)
    )
    
    X = np.column_stack([x0, x1, x2])
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTrue underlying relationship:")
    print(f"  y = 1.5*x0^1.5 + 0.8*x1^0.5 + 2.0*x2^(-0.5) + noise")
    print(f"\nNote: UFP learns that x0 needs power 1.5, x1 needs power 0.5,")
    print(f"      and x2 needs power -0.5 to fit the data best!")
    
    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # === Fit GMDH-UFP ===
    print("\n" + "=" * 80)
    print("Training GMDH-UFP Network (learning optimal powers)...")
    print("=" * 80)
    
    # Powers to search over (UFP will find the best ones)
    powers = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    gmdh_ufp = GMDH_UFP(
        n_keep=3,
        max_layers=5,
        ridge=1e-6,
        training_split=0.5,
        patience=1,
        learnable_powers=True,  # Learn powers via grid search
        powers_to_try=powers,
        random_state=42
    )
    
    gmdh_ufp.fit(X_train, y_train)
    
    # === Print learned equation ===
    gmdh_ufp.print_equation(feature_names=['x0', 'x1', 'x2'])
    
    # === Evaluate ===
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    y_train_pred = gmdh_ufp.predict(X_train)
    y_test_pred = gmdh_ufp.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTrain RMSE: {train_rmse:.6f}")
    print(f"Test RMSE:  {test_rmse:.6f}")
    print(f"Train R²:   {train_r2:.6f}")
    print(f"Test R²:    {test_r2:.6f}")
    
    print(f"\n--- Unconstrained Fractional Polynomials ---")
    print(f"UFP vs CFP (Constrained Fractional Polynomials):")
    print(f"")
    print(f"CFP (Constrained):")
    print(f"  - Searches ALL combinations of predefined powers for a and b")
    print(f"  - More feature combinations (n_powers_a × n_powers_b)")
    print(f"  - Broader search space, but more computationally expensive")
    print(f"")
    print(f"UFP (Unconstrained):")
    print(f"  - Learns a SINGLE optimal power for each input per neuron")
    print(f"  - Form: w₀ + w₁*a^(p_a) + w₂*b^(p_b)")
    print(f"  - Much fewer candidate neurons (only powers_to_try combinations)")
    print(f"  - More efficient, discovers simplest power law structures")
    print(f"")
    print(f"UFP discovered:")
    if gmdh_ufp.layers_:
        best_neuron = gmdh_ufp.layers_[0].neurons[0]
        print(f"  - Power for input a: {gmdh_ufp._format_power(best_neuron.power_a)}")
        print(f"  - Power for input b: {gmdh_ufp._format_power(best_neuron.power_b)}")
    
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
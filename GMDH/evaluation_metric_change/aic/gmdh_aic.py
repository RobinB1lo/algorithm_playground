import numpy as np
from itertools import combinations
from typing import Optional, List


class GMDHNeuron_AIC:
    __slots__ = ('i', 'j', 'w')

    def __init__(self, i: int, j: int) -> None:
        self.i = i                          # column index into the layer's inputs
        self.j = j                          # column index
        self.w: Optional[np.ndarray] = None # coefficients, set by fit()

    @staticmethod
    def _features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(a), a, b, a * b, a ** 2, b ** 2])

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray,
            ridge: float = 1e-6) -> 'GMDHNeuron_AIC':
        X = self._features(a, b)
        A = X.T @ X + ridge * np.eye(6)
        self.w = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._features(a, b) @ self.w

    def compute_aic(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
        """Compute AIC for model selection.
        
        AIC = 2k + n*ln(RSS/n)
        where k = number of parameters (6 for polynomial)
              n = sample size
              RSS = residual sum of squares
        
        Lower AIC is better (balances fit quality with model complexity).
        """
        y_pred = self.predict(a, b)
        rss = np.sum((y - y_pred) ** 2)
        n = len(y)
        k = 6  # number of parameters in _features
        
        rss_safe = max(rss, 1e-12)  # avoid log(0)
        aic = 2 * k + n * np.log(rss_safe / n)
        return float(aic)

    def compute_rmse(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
        """Compute RMSE for reference/comparison."""
        y_pred = self.predict(a, b)
        return float(np.sqrt(np.mean((y - y_pred) ** 2)))


class GMDHLayer_AIC:
    """One GMDH layer: fit a neuron for every pair of input columns, score each
    on the selection set using AIC, and keep the best `n_keep`."""

    def __init__(self, n_keep: int, ridge: float = 1e-6) -> None:
        self.n_keep = n_keep
        self.ridge = ridge
        self.neurons: List[GMDHNeuron_AIC] = []   # survivors, sorted best-first by AIC
        self.best_aic: float = np.inf             # lowest AIC in this layer
        self.best_rmse: float = np.inf            # RMSE of best neuron (for reference)

    def fit(self, Z_tr: np.ndarray, Z_se: np.ndarray,
            y_tr: np.ndarray, y_se: np.ndarray):
        """Z_tr, Z_se have shape (n_samples, n_inputs) for the train / selection
        splits. Returns the survivors' output columns on both splits, ready to be
        fed in as the next layer's inputs."""
        m = Z_tr.shape[1]
        # each entry: (aic_score, rmse_score, neuron, train_output, selection_output)
        candidates = []

        for i, j in combinations(range(m), 2):
            nu = GMDHNeuron_AIC(i, j).fit(Z_tr[:, i], Z_tr[:, j], y_tr, self.ridge)
            aic = nu.compute_aic(Z_se[:, i], Z_se[:, j], y_se)  # score on SELECTION
            rmse = nu.compute_rmse(Z_se[:, i], Z_se[:, j], y_se)
            out_tr = nu.predict(Z_tr[:, i], Z_tr[:, j])       # outputs to pass on
            out_se = nu.predict(Z_se[:, i], Z_se[:, j])
            candidates.append((aic, rmse, nu, out_tr, out_se))

        # keep the n_keep lowest-AIC neurons, best first
        candidates.sort(key=lambda c: c[0])
        survivors = candidates[:self.n_keep]

        self.neurons = [c[2] for c in survivors]
        self.best_aic = survivors[0][0]
        self.best_rmse = survivors[0][1]
        Z_tr_next = np.column_stack([c[3] for c in survivors])
        Z_se_next = np.column_stack([c[4] for c in survivors])
        return Z_tr_next, Z_se_next

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Replay the survivors on a new input matrix -> this layer's outputs."""
        return np.column_stack([
            nu.predict(Z[:, nu.i], Z[:, nu.j]) for nu in self.neurons
        ])


class GMDH_AIC:
    """GMDH network using AIC (Akaike Information Criterion) for neuron selection."""
    
    def __init__(self, n_keep: int, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0, 
                 method: str = "diminishing_returns", threshold: Optional[float] = None,
                 random_state: Optional[int] = None) -> None:
        """Initialize GMDH-AIC network with hyperparameters.
        
        Parameters:
        -----------
        n_keep : int
            Number of best neurons to keep per layer
        max_layers : int
            Maximum number of layers to build
        ridge : float
            Ridge regression parameter (for numerical stability)
        training_split : float
            Fraction of data to use for training (rest goes to selection)
        patience : int
            Number of non-improving layers to tolerate before stopping
        method : str
            Stopping method: "diminishing_returns" or "threshold"
        threshold : float, optional
            Target AIC threshold (only used if method="threshold")
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.method = method
        self.threshold = threshold
        self.random_state = random_state
        
        # Will be populated during fit
        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.layers_: List[GMDHLayer_AIC] = []
        self.best_aic_: float = np.inf
        self.best_rmse_: float = np.inf

    def _standardize(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Standardize X and y, storing statistics for later de-standardization."""
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-12  # avoid division by zero
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() + 1e-12

        X_std = (X - self.x_mean_) / self.x_std_
        y_std = (y - self.y_mean_) / self.y_std_
        return X_std, y_std
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GMDH-AIC network on data X and target y.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of length n_samples
        """
        # Standardize data
        X_std, y_std = self._standardize(X, y)

        # Split into train and selection sets
        rng = np.random.default_rng(self.random_state)
        n = X_std.shape[0]
        indices = rng.permutation(n)
        split = int(n * self.training_split)
        tr_idx = indices[:split]
        se_idx = indices[split:]

        Z_tr = X_std[tr_idx]
        Z_se = X_std[se_idx]
        y_tr = y_std[tr_idx]
        y_se = y_std[se_idx]

        # Layer loop
        self.layers_ = []
        best_aic = np.inf
        best_rmse = np.inf
        best_layer_idx = -1
        no_improve_count = 0

        for layer_num in range(self.max_layers):
            # Stop if we can't form a pair of inputs
            if Z_tr.shape[1] < 2:
                break

            # Build and fit the layer
            layer = GMDHLayer_AIC(self.n_keep, self.ridge)
            Z_tr, Z_se = layer.fit(Z_tr, Z_se, y_tr, y_se)
            self.layers_.append(layer)

            # Check if this layer improved (using AIC)
            if layer.best_aic < best_aic - 1e-12:
                best_aic = layer.best_aic
                best_rmse = layer.best_rmse
                best_layer_idx = layer_num
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Stop if patience exhausted (diminishing returns)
            if no_improve_count > self.patience:
                break

            # Stop if threshold reached
            if self.method == "threshold" and best_aic < self.threshold:
                break
        
        # Trim to best layer (discard layers built after the best one)
        self.layers_ = self.layers_[:best_layer_idx + 1]
        self.best_aic_ = best_aic
        self.best_rmse_ = best_rmse

    def _neuron_equation(self, neuron: GMDHNeuron_AIC, feature_names: List[str]) -> str:
        """Build string representation of a neuron's polynomial equation."""
        a_name = feature_names[neuron.i]
        b_name = feature_names[neuron.j]
        w = neuron.w
        
        terms = []
        # Constant term
        if abs(w[0]) > 1e-10:
            terms.append(f"{w[0]:+.6f}")
        # Linear terms
        if abs(w[1]) > 1e-10:
            terms.append(f"{w[1]:+.6f}*{a_name}")
        if abs(w[2]) > 1e-10:
            terms.append(f"{w[2]:+.6f}*{b_name}")
        # Interaction term
        if abs(w[3]) > 1e-10:
            terms.append(f"{w[3]:+.6f}*{a_name}*{b_name}")
        # Quadratic terms
        if abs(w[4]) > 1e-10:
            terms.append(f"{w[4]:+.6f}*{a_name}²")
        if abs(w[5]) > 1e-10:
            terms.append(f"{w[5]:+.6f}*{b_name}²")
        
        # Join terms, cleaning up leading +
        eq = " ".join(terms) if terms else "0"
        eq = eq.lstrip("+").strip()
        return eq

    def print_equation(self, feature_names: Optional[List[str]] = None) -> None:
        """Print the GMDH-AIC network's final polynomial equation after fitting."""
        if not self.layers_:
            print("No layers fitted.")
            return
        
        n_features = len(self.x_mean_)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        
        print("\n" + "=" * 80)
        print("GMDH-AIC Network: Final Polynomial Equation")
        print("(Neurons selected via Akaike Information Criterion)")
        print("=" * 80)
        
        # Track feature names through layers
        current_features = feature_names.copy()
        layer_outputs = {}  # For reference if needed
        
        for layer_idx, layer in enumerate(self.layers_):
            print(f"\n--- Layer {layer_idx + 1} ---")
            print(f"Best Neuron:")
            print(f"  AIC:  {layer.best_aic:.6f}")
            print(f"  RMSE: {layer.best_rmse:.6f}")
            print(f"Neurons kept: {len(layer.neurons)}")
            
            next_features = []
            for neuron_idx, neuron in enumerate(layer.neurons):
                eq = self._neuron_equation(neuron, current_features)
                var_name = f"z{layer_idx}_{neuron_idx}"
                next_features.append(var_name)
                print(f"\n  {var_name} = {eq}")
            
            current_features = next_features
            layer_outputs[layer_idx] = current_features
        
        # Final equation
        final_var = current_features[0]
        print(f"\n" + "=" * 80)
        print("FINAL PREDICTION (standardized):")
        print(f"  y_standardized = {final_var}")
        
        print("\nDE-STANDARDIZE TO ORIGINAL SCALE:")
        print(f"  y_predicted = {final_var} × {self.y_std_:.6f} + {self.y_mean_:.6f}")
        print("=" * 80 + "\n")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray
            Predicted target values of length n_samples
        """
        # Standardize using stored statistics
        X_std = (X - self.x_mean_) / self.x_std_

        # Replay through each kept layer
        Z = X_std
        for layer in self.layers_:
            Z = layer.predict(Z)
        
        # Extract best neuron output (first column) and de-standardize
        y_pred = Z[:, 0] * self.y_std_ + self.y_mean_
        return y_pred
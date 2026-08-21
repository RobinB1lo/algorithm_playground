import numpy as np
from itertools import combinations
from typing import Optional, List


class GMDHNeuron_UFP:
    """Neuron with unconstrained fractional polynomial basis functions.
    
    Unlike CFP which uses a predefined set of powers, UFP learns the optimal
    powers for each input during fitting via optimization.
    """
    __slots__ = ('i', 'j', 'w', 'power_a', 'power_b', 'learnable_powers')

    def __init__(self, i: int, j: int, learnable_powers: bool = True) -> None:
        self.i = i                          # column index
        self.j = j                          # column index
        self.learnable_powers = learnable_powers
        self.power_a: float = 1.0           # power for input a (learned)
        self.power_b: float = 1.0           # power for input b (learned)
        self.w: Optional[np.ndarray] = None # coefficients [const, coeff_a, coeff_b]

    def _safe_power(self, x: np.ndarray, p: float, epsilon: float = 1e-10) -> np.ndarray:
        """Compute x^p safely."""
        if p == 0:
            return np.ones_like(x)
        
        x_safe = np.where(np.abs(x) < epsilon, epsilon, x)
        
        if p == int(p):
            return np.power(x_safe, int(p))
        else:
            x_pos = np.abs(x_safe)
            return np.power(x_pos, p)

    def _features(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Build fractional polynomial features with learned powers."""
        a_transformed = self._safe_power(a, self.power_a)
        b_transformed = self._safe_power(b, self.power_b)
        return np.column_stack([np.ones_like(a), a_transformed, b_transformed])

    def _fit_powers_grid_search(self, a: np.ndarray, b: np.ndarray, y: np.ndarray,
                                 powers_to_try: List[float], ridge: float = 1e-6) -> tuple:
        """Grid search over powers to find best combination."""
        best_error = np.inf
        best_pa, best_pb = 1.0, 1.0
        
        for pa in powers_to_try:
            for pb in powers_to_try:
                a_t = self._safe_power(a, pa)
                b_t = self._safe_power(b, pb)
                X = np.column_stack([np.ones_like(a), a_t, b_t])
                
                try:
                    A = X.T @ X + ridge * np.eye(3)
                    w = np.linalg.solve(A, X.T @ y)
                    y_pred = X @ w
                    error = np.sqrt(np.mean((y - y_pred) ** 2))
                    
                    if error < best_error:
                        best_error = error
                        best_pa, best_pb = pa, pb
                except:
                    continue
        
        return best_pa, best_pb

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray,
            ridge: float = 1e-6, powers_to_try: List[float] = None) -> 'GMDHNeuron_UFP':
        """Fit both powers and coefficients."""
        if powers_to_try is None:
            powers_to_try = [-2, -1, -0.5, 0, 0.5, 1, 2]
        
        if self.learnable_powers:
            # Grid search to find best powers
            self.power_a, self.power_b = self._fit_powers_grid_search(
                a, b, y, powers_to_try, ridge
            )
        
        # Fit coefficients with learned powers
        X = self._features(a, b)
        A = X.T @ X + ridge * np.eye(3)
        self.w = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._features(a, b) @ self.w

    def error(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - self.predict(a, b)) ** 2)))


class GMDHLayer_UFP:
    """GMDH layer using unconstrained fractional polynomials."""

    def __init__(self, n_keep: int, ridge: float = 1e-6,
                 learnable_powers: bool = True,
                 powers_to_try: List[float] = None) -> None:
        self.n_keep = n_keep
        self.ridge = ridge
        self.learnable_powers = learnable_powers
        self.powers_to_try = powers_to_try or [-2, -1, -0.5, 0, 0.5, 1, 2]
        self.neurons: List[GMDHNeuron_UFP] = []
        self.best_error: float = np.inf

    def fit(self, Z_tr: np.ndarray, Z_se: np.ndarray,
            y_tr: np.ndarray, y_se: np.ndarray):
        """Fit layer with unconstrained fractional polynomials."""
        m = Z_tr.shape[1]
        candidates = []

        for i, j in combinations(range(m), 2):
            nu = GMDHNeuron_UFP(i, j, self.learnable_powers).fit(
                Z_tr[:, i], Z_tr[:, j], y_tr, self.ridge, self.powers_to_try
            )
            err = nu.error(Z_se[:, i], Z_se[:, j], y_se)
            out_tr = nu.predict(Z_tr[:, i], Z_tr[:, j])
            out_se = nu.predict(Z_se[:, i], Z_se[:, j])
            candidates.append((err, nu, out_tr, out_se))

        candidates.sort(key=lambda c: c[0])
        survivors = candidates[:self.n_keep]

        self.neurons = [c[1] for c in survivors]
        self.best_error = survivors[0][0]
        Z_tr_next = np.column_stack([c[2] for c in survivors])
        Z_se_next = np.column_stack([c[3] for c in survivors])
        return Z_tr_next, Z_se_next

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.column_stack([
            nu.predict(Z[:, nu.i], Z[:, nu.j]) for nu in self.neurons
        ])


class GMDH_UFP:
    """GMDH with Unconstrained Fractional Polynomials (learns optimal powers)."""
    
    def __init__(self, n_keep: int, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 learnable_powers: bool = True,
                 powers_to_try: List[float] = None,
                 random_state: Optional[int] = None) -> None:
        """
        Parameters:
        -----------
        n_keep : int
            Number of best neurons to keep per layer
        max_layers : int
            Maximum number of layers
        ridge : float
            Ridge regression parameter
        training_split : float
            Train/selection split ratio
        patience : int
            Early stopping patience
        learnable_powers : bool
            Whether to learn powers (True) or fix at 1.0 (False)
        powers_to_try : list of float
            Powers to search over during grid search
        random_state : int, optional
            Random seed
        """
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.learnable_powers = learnable_powers
        self.powers_to_try = powers_to_try or [-2, -1, -0.5, 0, 0.5, 1, 2]
        self.random_state = random_state
        
        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.layers_: List[GMDHLayer_UFP] = []
        self.best_error_: float = np.inf

    def _standardize(self, X: np.ndarray, y: np.ndarray) -> tuple:
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-12
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() + 1e-12

        X_std = (X - self.x_mean_) / self.x_std_
        y_std = (y - self.y_mean_) / self.y_std_
        return X_std, y_std
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GMDH-UFP network."""
        X_std, y_std = self._standardize(X, y)

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

        self.layers_ = []
        best_error = np.inf
        best_layer_idx = -1
        no_improve_count = 0

        for layer_num in range(self.max_layers):
            if Z_tr.shape[1] < 2:
                break

            layer = GMDHLayer_UFP(
                self.n_keep, self.ridge, self.learnable_powers, self.powers_to_try
            )
            Z_tr, Z_se = layer.fit(Z_tr, Z_se, y_tr, y_se)
            self.layers_.append(layer)

            if layer.best_error < best_error - 1e-12:
                best_error = layer.best_error
                best_layer_idx = layer_num
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count > self.patience:
                break

        self.layers_ = self.layers_[:best_layer_idx + 1]
        self.best_error_ = best_error

    def _format_power(self, p: float) -> str:
        """Format power nicely."""
        if p == int(p):
            return str(int(p))
        else:
            return f"{p:.2g}"

    def _neuron_equation(self, neuron: GMDHNeuron_UFP, feature_names: List[str]) -> str:
        """Build string representation of unconstrained fractional polynomial."""
        a_name = feature_names[neuron.i]
        b_name = feature_names[neuron.j]
        w = neuron.w
        
        terms = []
        
        # Constant term
        if abs(w[0]) > 1e-10:
            terms.append(f"{w[0]:+.6f}")
        
        # Power of a (learned)
        if abs(w[1]) > 1e-10:
            p_str = self._format_power(neuron.power_a)
            if neuron.power_a == 1.0:
                terms.append(f"{w[1]:+.6f}*{a_name}")
            else:
                terms.append(f"{w[1]:+.6f}*{a_name}^{p_str}")
        
        # Power of b (learned)
        if abs(w[2]) > 1e-10:
            p_str = self._format_power(neuron.power_b)
            if neuron.power_b == 1.0:
                terms.append(f"{w[2]:+.6f}*{b_name}")
            else:
                terms.append(f"{w[2]:+.6f}*{b_name}^{p_str}")
        
        eq = " ".join(terms) if terms else "0"
        eq = eq.lstrip("+").strip()
        return eq

    def print_equation(self, feature_names: Optional[List[str]] = None) -> None:
        """Print the GMDH-UFP network's final unconstrained fractional polynomial equation."""
        if not self.layers_:
            print("No layers fitted.")
            return
        
        n_features = len(self.x_mean_)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        
        print("\n" + "=" * 80)
        print("GMDH-UFP Network: Final Unconstrained Fractional Polynomial Equation")
        print("=" * 80)
        print(f"Powers learned via grid search over: {self.powers_to_try}")
        print("Note: Each neuron learns its own optimal power for each input.")
        
        # Track feature names through layers
        current_features = feature_names.copy()
        
        for layer_idx, layer in enumerate(self.layers_):
            print(f"\n--- Layer {layer_idx + 1} ---")
            print(f"Best Selection Error: {layer.best_error:.6f}")
            print(f"Neurons kept: {len(layer.neurons)}")
            
            next_features = []
            for neuron_idx, neuron in enumerate(layer.neurons):
                eq = self._neuron_equation(neuron, current_features)
                var_name = f"z{layer_idx}_{neuron_idx}"
                next_features.append(var_name)
                print(f"\n  {var_name} = {eq}")
                print(f"       [learned powers: a^{self._format_power(neuron.power_a)}, b^{self._format_power(neuron.power_b)}]")
            
            current_features = next_features
        
        # Final equation
        final_var = current_features[0]
        print(f"\n" + "=" * 80)
        print("FINAL PREDICTION (standardized):")
        print(f"  y_standardized = {final_var}")
        
        print("\nDE-STANDARDIZE TO ORIGINAL SCALE:")
        print(f"  y_predicted = {final_var} × {self.y_std_:.6f} + {self.y_mean_:.6f}")
        print("=" * 80 + "\n")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        X_std = (X - self.x_mean_) / self.x_std_

        Z = X_std
        for layer in self.layers_:
            Z = layer.predict(Z)
        
        return Z[:, 0] * self.y_std_ + self.y_mean_
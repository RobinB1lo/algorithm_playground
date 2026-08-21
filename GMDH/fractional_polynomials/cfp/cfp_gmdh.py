import numpy as np
from itertools import combinations
from typing import Optional, List, Tuple


class GMDHNeuron_CFP:
    """Neuron with constrained fractional polynomial basis functions."""
    __slots__ = ('i', 'j', 'w', 'powers_a', 'powers_b')

    def __init__(self, i: int, j: int, powers_a: List[float] = None, 
                 powers_b: List[float] = None) -> None:
        self.i = i                          # column index
        self.j = j                          # column index
        if powers_a is None:
            powers_a = [-2, -1, -0.5, 0, 0.5, 1, 2]
        if powers_b is None:
            powers_b = [-2, -1, -0.5, 0, 0.5, 1, 2]
        self.powers_a = powers_a
        self.powers_b = powers_b
        self.w: Optional[np.ndarray] = None # coefficients

    def _safe_power(self, x: np.ndarray, p: float, epsilon: float = 1e-10) -> np.ndarray:
        """Compute x^p safely, handling negative/zero x."""
        if p == 0:
            return np.ones_like(x)
        
        x_safe = np.where(np.abs(x) < epsilon, epsilon, x)
        
        if p == int(p):
            return np.power(x_safe, int(p))
        else:
            # For fractional powers, ensure x_safe is positive
            x_pos = np.abs(x_safe)
            return np.power(x_pos, p)

    def _features(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Build fractional polynomial features."""
        features = [np.ones_like(a)]
        
        # Powers of a
        for p in self.powers_a:
            features.append(self._safe_power(a, p))
        
        # Powers of b
        for p in self.powers_b:
            features.append(self._safe_power(b, p))
        
        return np.column_stack(features)

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray,
            ridge: float = 1e-6) -> 'GMDHNeuron_CFP':
        """Fit fractional polynomial coefficients via ridge regression."""
        X = self._features(a, b)
        A = X.T @ X + ridge * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._features(a, b) @ self.w

    def error(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - self.predict(a, b)) ** 2)))


class GMDHLayer_CFP:
    """GMDH layer using constrained fractional polynomials."""

    def __init__(self, n_keep: int, ridge: float = 1e-6, 
                 powers_a: List[float] = None, powers_b: List[float] = None) -> None:
        self.n_keep = n_keep
        self.ridge = ridge
        self.powers_a = powers_a or [-2, -1, -0.5, 0, 0.5, 1, 2]
        self.powers_b = powers_b or [-2, -1, -0.5, 0, 0.5, 1, 2]
        self.neurons: List[GMDHNeuron_CFP] = []
        self.best_error: float = np.inf

    def fit(self, Z_tr: np.ndarray, Z_se: np.ndarray,
            y_tr: np.ndarray, y_se: np.ndarray):
        """Fit layer with constrained fractional polynomials."""
        m = Z_tr.shape[1]
        candidates = []

        for i, j in combinations(range(m), 2):
            nu = GMDHNeuron_CFP(i, j, self.powers_a, self.powers_b).fit(
                Z_tr[:, i], Z_tr[:, j], y_tr, self.ridge
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


class GMDH_CFP:
    """GMDH with Constrained Fractional Polynomials."""
    
    def __init__(self, n_keep: int, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 powers_a: List[float] = None, powers_b: List[float] = None,
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
        powers_a : list of float
            Fractional powers to use for input a (e.g., [-2, -1, -0.5, 0, 0.5, 1, 2])
        powers_b : list of float
            Fractional powers to use for input b
        random_state : int, optional
            Random seed
        """
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.powers_a = powers_a or [-2, -1, -0.5, 0, 0.5, 1, 2]
        self.powers_b = powers_b or [-2, -1, -0.5, 0, 0.5, 1, 2]
        self.random_state = random_state
        
        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.layers_: List[GMDHLayer_CFP] = []
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
        """Fit GMDH-CFP network."""
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

            layer = GMDHLayer_CFP(
                self.n_keep, self.ridge, self.powers_a, self.powers_b
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
        """Format fractional power nicely."""
        if p == int(p):
            return str(int(p))
        else:
            return f"{p:.2g}"

    def _neuron_equation(self, neuron: GMDHNeuron_CFP, feature_names: List[str]) -> str:
        """Build string representation of a fractional polynomial equation."""
        a_name = feature_names[neuron.i]
        b_name = feature_names[neuron.j]
        w = neuron.w
        
        terms = []
        idx = 0
        
        # Constant term
        if abs(w[idx]) > 1e-10:
            terms.append(f"{w[idx]:+.6f}")
        idx += 1
        
        # Powers of a
        for p in neuron.powers_a:
            if abs(w[idx]) > 1e-10:
                p_str = self._format_power(p)
                terms.append(f"{w[idx]:+.6f}*{a_name}^{p_str}")
            idx += 1
        
        # Powers of b
        for p in neuron.powers_b:
            if abs(w[idx]) > 1e-10:
                p_str = self._format_power(p)
                terms.append(f"{w[idx]:+.6f}*{b_name}^{p_str}")
            idx += 1
        
        # Join terms
        eq = " ".join(terms) if terms else "0"
        eq = eq.lstrip("+").strip()
        return eq

    def print_equation(self, feature_names: Optional[List[str]] = None) -> None:
        """Print the GMDH-CFP network's final fractional polynomial equation."""
        if not self.layers_:
            print("No layers fitted.")
            return
        
        n_features = len(self.x_mean_)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        
        print("\n" + "=" * 80)
        print("GMDH-CFP Network: Final Constrained Fractional Polynomial Equation")
        print("=" * 80)
        print(f"Fractional powers used:")
        print(f"  Powers for input a: {self.powers_a}")
        print(f"  Powers for input b: {self.powers_b}")
        
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
import numpy as np
from itertools import combinations
from typing import Optional, List


class GMDHNeuronTrig:
    """Neuron with trigonometric (Fourier) basis functions instead of polynomials."""
    __slots__ = ('i', 'j', 'w', 'n_harmonics', 'a_min', 'a_max', 'b_min', 'b_max')

    def __init__(self, i: int, j: int, n_harmonics: int = 3) -> None:
        self.i = i                          # column index
        self.j = j                          # column index
        self.n_harmonics = n_harmonics      # number of sin/cos harmonic pairs per input
        self.w: Optional[np.ndarray] = None # coefficients
        # Normalization parameters (set during fit)
        self.a_min, self.a_max = 0.0, 1.0
        self.b_min, self.b_max = 0.0, 1.0

    @staticmethod
    def _create_trig_basis(x: np.ndarray, n_harmonics: int = 3) -> tuple:
        """Create trigonometric (Fourier) basis functions for input x.

        Returns:
        --------
        basis_functions : list of arrays
            Evaluated basis functions at x (sin/cos pairs, increasing frequency)
        harmonics : array
            Harmonic numbers used
        """
        # Normalize x to [0, 2*pi] so period 1 maps to a full cycle
        x_min, x_max = x.min(), x.max()
        x_norm = (x - x_min) / (x_max - x_min + 1e-10) * 2.0 * np.pi

        harmonics = np.arange(1, n_harmonics + 1)

        # Evaluate sin/cos basis functions at each harmonic
        basis_functions = []
        for h in harmonics:
            basis_functions.append(np.sin(h * x_norm))
            basis_functions.append(np.cos(h * x_norm))

        return basis_functions, x_min, x_max, harmonics

    def _features(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Build features from trigonometric basis functions of a and b."""
        # Normalize using stored normalization parameters, onto [0, 2*pi]
        a_norm = (a - self.a_min) / (self.a_max - self.a_min + 1e-10) * 2.0 * np.pi
        b_norm = (b - self.b_min) / (self.b_max - self.b_min + 1e-10) * 2.0 * np.pi

        harmonics = np.arange(1, self.n_harmonics + 1)

        # Evaluate sin/cos basis functions
        a_basis = []
        b_basis = []
        for h in harmonics:
            a_basis.append(np.sin(h * a_norm))
            a_basis.append(np.cos(h * a_norm))
            b_basis.append(np.sin(h * b_norm))
            b_basis.append(np.cos(h * b_norm))

        # Build feature matrix: [1, a_basis, b_basis, interactions]
        features = [np.ones_like(a)]
        features.extend(a_basis)
        features.extend(b_basis)

        # Add interaction terms between a and b basis functions
        for a_bf in a_basis:
            for b_bf in b_basis:
                features.append(a_bf * b_bf)

        return np.column_stack(features)

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray,
            ridge: float = 1e-6) -> 'GMDHNeuronTrig':
        """Fit trigonometric coefficients via ridge regression."""
        # Store normalization parameters
        self.a_min, self.a_max = a.min(), a.max()
        self.b_min, self.b_max = b.min(), b.max()

        # Build feature matrix
        X = self._features(a, b)

        # Ridge regression
        A = X.T @ X + ridge * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._features(a, b) @ self.w

    def error(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - self.predict(a, b)) ** 2)))


class GMDHLayerTrig:
    """GMDH layer using trigonometric (Fourier) basis functions."""

    def __init__(self, n_keep: int, ridge: float = 1e-6,
                 n_harmonics: int = 3) -> None:
        self.n_keep = n_keep
        self.ridge = ridge
        self.n_harmonics = n_harmonics    # sin/cos harmonic pairs per input
        self.neurons: List[GMDHNeuronTrig] = []
        self.best_error: float = np.inf

    def fit(self, Z_tr: np.ndarray, Z_se: np.ndarray,
            y_tr: np.ndarray, y_se: np.ndarray):
        """Fit layer: fit trig neurons on train, score on selection."""
        m = Z_tr.shape[1]
        candidates = []

        for i, j in combinations(range(m), 2):
            nu = GMDHNeuronTrig(i, j, self.n_harmonics).fit(
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


class GMDH_Trig:
    """GMDH with trigonometric (Fourier) basis functions."""

    def __init__(self, n_keep: int, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 n_harmonics: int = 3, random_state: Optional[int] = None) -> None:
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
        n_harmonics : int
            Number of sin/cos harmonic pairs per input (default 3).
            Each input contributes 2*n_harmonics basis functions.
        random_state : int, optional
            Random seed
        """
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.n_harmonics = n_harmonics
        self.random_state = random_state

        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.layers_: List[GMDHLayerTrig] = []
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
        """Fit GMDH-Trig network."""
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

            layer = GMDHLayerTrig(
                self.n_keep, self.ridge, self.n_harmonics
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

    def _neuron_equation(self, neuron: GMDHNeuronTrig, feature_names: List[str]) -> str:
        """Build description of trigonometric neuron's learned equation."""
        a_name = feature_names[neuron.i]
        b_name = feature_names[neuron.j]
        w = neuron.w
        
        harmonics = np.arange(1, neuron.n_harmonics + 1)
        
        terms = []
        # Constant term
        if abs(w[0]) > 1e-10:
            terms.append(f"{w[0]:+.6f}*const")
        
        # a sin/cos terms
        idx = 1
        for h in harmonics:
            if abs(w[idx]) > 1e-10:
                terms.append(f"{w[idx]:+.6f}*sin({h}*{a_name})")
            idx += 1
            if abs(w[idx]) > 1e-10:
                terms.append(f"{w[idx]:+.6f}*cos({h}*{a_name})")
            idx += 1
        
        # b sin/cos terms
        for h in harmonics:
            if abs(w[idx]) > 1e-10:
                terms.append(f"{w[idx]:+.6f}*sin({h}*{b_name})")
            idx += 1
            if abs(w[idx]) > 1e-10:
                terms.append(f"{w[idx]:+.6f}*cos({h}*{b_name})")
            idx += 1
        
        # Interaction terms (only show significant ones)
        for ai_h in harmonics:
            for ai_type in ['sin', 'cos']:
                for bi_h in harmonics:
                    for bi_type in ['sin', 'cos']:
                        if abs(w[idx]) > 1e-10:
                            a_term = f"{ai_type}({ai_h}*{a_name})"
                            b_term = f"{bi_type}({bi_h}*{b_name})"
                            terms.append(f"{w[idx]:+.6f}*{a_term}×{b_term}")
                        idx += 1
        
        eq = " ".join(terms) if terms else "0"
        eq = eq.lstrip("+").strip()
        return eq

    def print_equation(self, feature_names: Optional[List[str]] = None) -> None:
        """Print the GMDH-Trig network's learned equations."""
        if not self.layers_:
            print("No layers fitted.")
            return
        
        n_features = len(self.x_mean_)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        
        print("\n" + "=" * 80)
        print("GMDH-Fourier Network: Final Trigonometric (Sin/Cos) Equations")
        print(f"Number of harmonics (n_harmonics): {self.layers_[0].n_harmonics}")
        print("Note: Input range is mapped to [0, 2π], so each harmonic h represents")
        print("      frequencies that complete h full cycles across the input range")
        print("=" * 80)
        
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
                print(f"       (combines {current_features[neuron.i]} and {current_features[neuron.j]} via Fourier)")
            
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
import numpy as np
from itertools import combinations
from typing import Optional, List


class GMDHNeuronRBF:
    """Neuron with Gaussian radial basis functions instead of polynomials."""
    __slots__ = ('i', 'j', 'w', 'n_centers', 'gamma', 'a_min', 'a_max', 'b_min', 'b_max')

    def __init__(self, i: int, j: int, n_centers: int = 5, gamma: float = 5.0) -> None:
        self.i = i                          # column index
        self.j = j                          # column index
        self.n_centers = n_centers          # number of RBF centers per input
        self.gamma = gamma                  # RBF width parameter (1/(2*sigma^2))
        self.w: Optional[np.ndarray] = None # coefficients
        # Normalization parameters (set during fit)
        self.a_min, self.a_max = 0.0, 1.0
        self.b_min, self.b_max = 0.0, 1.0

    @staticmethod
    def _create_rbf_basis(x: np.ndarray, n_centers: int = 5, gamma: float = 5.0) -> tuple:
        """Create Gaussian RBF basis functions for input x.

        Returns:
        --------
        basis_functions : list of arrays
            Evaluated basis functions at x
        centers : array
            Center locations used
        """
        # Normalize x to [0, 1]
        x_min, x_max = x.min(), x.max()
        x_norm = (x - x_min) / (x_max - x_min + 1e-10)

        # Place centers evenly across [0, 1]
        centers = np.linspace(0, 1, n_centers)

        # Evaluate each Gaussian RBF basis function
        basis_functions = []
        for c in centers:
            basis_functions.append(np.exp(-gamma * (x_norm - c) ** 2))

        return basis_functions, x_min, x_max, centers

    def _features(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Build features from RBF basis functions of a and b."""
        # Normalize using stored normalization parameters
        a_norm = (a - self.a_min) / (self.a_max - self.a_min + 1e-10)
        b_norm = (b - self.b_min) / (self.b_max - self.b_min + 1e-10)

        # Recreate centers (same as in fit)
        centers = np.linspace(0, 1, self.n_centers)

        # Evaluate RBF basis functions
        a_basis = []
        b_basis = []
        for c in centers:
            a_basis.append(np.exp(-self.gamma * (a_norm - c) ** 2))
            b_basis.append(np.exp(-self.gamma * (b_norm - c) ** 2))

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
            ridge: float = 1e-6) -> 'GMDHNeuronRBF':
        """Fit RBF coefficients via ridge regression."""
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


class GMDHLayerRBF:
    """GMDH layer using Gaussian radial basis functions."""

    def __init__(self, n_keep: int, ridge: float = 1e-6,
                 n_centers: int = 5, gamma: float = 5.0) -> None:
        self.n_keep = n_keep
        self.ridge = ridge
        self.n_centers = n_centers    # RBF centers per input
        self.gamma = gamma            # RBF width parameter
        self.neurons: List[GMDHNeuronRBF] = []
        self.best_error: float = np.inf

    def fit(self, Z_tr: np.ndarray, Z_se: np.ndarray,
            y_tr: np.ndarray, y_se: np.ndarray):
        """Fit layer: fit RBF neurons on train, score on selection."""
        m = Z_tr.shape[1]
        candidates = []

        for i, j in combinations(range(m), 2):
            nu = GMDHNeuronRBF(i, j, self.n_centers, self.gamma).fit(
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


class GMDH_RBF:
    """GMDH with Gaussian radial basis functions."""

    def __init__(self, n_keep: int, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 n_centers: int = 5, gamma: float = 5.0,
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
        n_centers : int
            Number of RBF centers per input (default 5)
        gamma : float
            RBF width parameter, 1/(2*sigma^2) (default 5.0)
        random_state : int, optional
            Random seed
        """
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.n_centers = n_centers
        self.gamma = gamma
        self.random_state = random_state

        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.layers_: List[GMDHLayerRBF] = []
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
        """Fit GMDH-RBF network."""
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

            layer = GMDHLayerRBF(
                self.n_keep, self.ridge, self.n_centers, self.gamma
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

    def _neuron_equation(self, neuron: GMDHNeuronRBF, feature_names: List[str]) -> str:
        """Build description of RBF neuron's learned equation."""
        a_name = feature_names[neuron.i]
        b_name = feature_names[neuron.j]
        w = neuron.w
        
        # Centers for RBF functions
        centers = np.linspace(0, 1, neuron.n_centers)
        sigma = np.sqrt(1.0 / (2.0 * neuron.gamma))  # convert gamma back to sigma
        
        terms = []
        # Constant term
        if abs(w[0]) > 1e-10:
            terms.append(f"{w[0]:+.6f}*const")
        
        # a RBF basis functions
        for i, c in enumerate(centers):
            if abs(w[1 + i]) > 1e-10:
                terms.append(f"{w[1 + i]:+.6f}*φ({a_name}; c={c:.2f}, σ={sigma:.2f})")
        
        # b RBF basis functions
        for i, c in enumerate(centers):
            if abs(w[1 + neuron.n_centers + i]) > 1e-10:
                terms.append(f"{w[1 + neuron.n_centers + i]:+.6f}*φ({b_name}; c={c:.2f}, σ={sigma:.2f})")
        
        # Interaction terms (only show active ones)
        interaction_start = 1 + 2 * neuron.n_centers
        for ai, ca in enumerate(centers):
            for bi, cb in enumerate(centers):
                idx = interaction_start + ai * neuron.n_centers + bi
                if abs(w[idx]) > 1e-10:
                    terms.append(f"{w[idx]:+.6f}*φ({a_name}; {ca:.2f})×φ({b_name}; {cb:.2f})")
        
        eq = " ".join(terms) if terms else "0"
        eq = eq.lstrip("+").strip()
        return eq

    def print_equation(self, feature_names: Optional[List[str]] = None) -> None:
        """Print the GMDH-RBF network's learned equations."""
        if not self.layers_:
            print("No layers fitted.")
            return
        
        n_features = len(self.x_mean_)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        
        sigma = np.sqrt(1.0 / (2.0 * self.layers_[0].gamma))
        
        print("\n" + "=" * 80)
        print("GMDH-RBF Network: Final Gaussian RBF Equations")
        print(f"Number of centers (n_centers): {self.layers_[0].n_centers}")
        print(f"Gamma (width parameter): {self.layers_[0].gamma}")
        print(f"Sigma (standard deviation): {sigma:.4f}")
        print("Note: φ(x; c, σ) = exp(-γ*(x_normalized - c)²) where γ = 1/(2*σ²)")
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
                print(f"       (combines {current_features[neuron.i]} and {current_features[neuron.j]} via Gaussian RBFs)")
            
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
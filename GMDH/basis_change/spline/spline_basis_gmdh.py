import numpy as np
from itertools import combinations
from scipy.interpolate import BSpline
from typing import Optional, List


class GMDHNeuronSpline:
    """Neuron with B-spline basis functions instead of polynomials."""
    __slots__ = ('i', 'j', 'w', 'n_knots', 'k', 'a_min', 'a_max', 'b_min', 'b_max')

    def __init__(self, i: int, j: int, n_knots: int = 3, k: int = 3) -> None:
        self.i = i                          # column index
        self.j = j                          # column index
        self.n_knots = n_knots              # interior knots for spline
        self.k = k                          # spline degree (3 = cubic)
        self.w: Optional[np.ndarray] = None # coefficients
        # Normalization parameters (set during fit)
        self.a_min, self.a_max = 0.0, 1.0
        self.b_min, self.b_max = 0.0, 1.0

    @staticmethod
    def _create_spline_basis(x: np.ndarray, n_knots: int = 3, k: int = 3) -> tuple:
        """Create B-spline basis functions for input x.
        
        Returns:
        --------
        basis_functions : list of arrays
            Evaluated basis functions at x
        knots : array
            Knot vector used
        """
        # Normalize x to [0, 1]
        x_min, x_max = x.min(), x.max()
        x_norm = (x - x_min) / (x_max - x_min + 1e-10)
        
        # Create knot vector: [0, 0, 0, 0, interior_knots, 1, 1, 1, 1] for k=3
        interior_knots = np.linspace(0, 1, n_knots)
        knots = np.concatenate([[0] * (k + 1), interior_knots, [1] * (k + 1)])
        
        # Number of basis functions
        n_basis = len(knots) - k - 1
        
        # Evaluate each B-spline basis function
        basis_functions = []
        for i in range(n_basis):
            coeff = np.zeros(n_basis)
            coeff[i] = 1.0
            spl = BSpline(knots, coeff, k)
            basis_functions.append(spl(x_norm))
        
        return basis_functions, x_min, x_max, knots

    def _features(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Build features from spline basis functions of a and b."""
        # Normalize using stored normalization parameters
        a_norm = (a - self.a_min) / (self.a_max - self.a_min + 1e-10)
        b_norm = (b - self.b_min) / (self.b_max - self.b_min + 1e-10)
        
        # Recreate knots (same as in fit)
        interior_knots = np.linspace(0, 1, self.n_knots)
        knots = np.concatenate([[0] * (self.k + 1), interior_knots, [1] * (self.k + 1)])
        n_basis = len(knots) - self.k - 1
        
        # Evaluate spline basis functions
        a_basis = []
        b_basis = []
        for i in range(n_basis):
            coeff = np.zeros(n_basis)
            coeff[i] = 1.0
            spl = BSpline(knots, coeff, self.k)
            a_basis.append(spl(a_norm))
            b_basis.append(spl(b_norm))
        
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
            ridge: float = 1e-6) -> 'GMDHNeuronSpline':
        """Fit spline coefficients via ridge regression."""
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


class GMDHLayerSpline:
    """GMDH layer using spline basis functions."""

    def __init__(self, n_keep: int, ridge: float = 1e-6,
                 n_knots: int = 3, k: int = 3) -> None:
        self.n_keep = n_keep
        self.ridge = ridge
        self.n_knots = n_knots        # interior knots for spline
        self.k = k                     # spline degree
        self.neurons: List[GMDHNeuronSpline] = []
        self.best_error: float = np.inf

    def fit(self, Z_tr: np.ndarray, Z_se: np.ndarray,
            y_tr: np.ndarray, y_se: np.ndarray):
        """Fit layer: fit spline neurons on train, score on selection."""
        m = Z_tr.shape[1]
        candidates = []

        for i, j in combinations(range(m), 2):
            nu = GMDHNeuronSpline(i, j, self.n_knots, self.k).fit(
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


class GMDH_Spline:
    """GMDH with B-spline basis functions."""
    
    def __init__(self, n_keep: int, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 n_knots: int = 3, k: int = 3, random_state: Optional[int] = None) -> None:
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
        n_knots : int
            Number of interior knots for splines (default 3)
        k : int
            Spline degree (default 3 = cubic)
        random_state : int, optional
            Random seed
        """
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.n_knots = n_knots
        self.k = k
        self.random_state = random_state
        
        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.layers_: List[GMDHLayerSpline] = []
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
        """Fit GMDH-Spline network."""
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

            layer = GMDHLayerSpline(
                self.n_keep, self.ridge, self.n_knots, self.k
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        X_std = (X - self.x_mean_) / self.x_std_

        Z = X_std
        for layer in self.layers_:
            Z = layer.predict(Z)
        
        return Z[:, 0] * self.y_std_ + self.y_mean_
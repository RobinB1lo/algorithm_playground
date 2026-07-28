import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from typing import Optional, List, Tuple

class GMDHNeuronConstrained:
    """Neuron with constrained fractional exponents (e1, e2 only)."""
    __slots__ = ('i', 'j', 'e1', 'e2', 'w')

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j
        self.e1: float = 1.0  # exponent for input a
        self.e2: float = 1.0  # exponent for input b
        self.w: Optional[np.ndarray] = None

    @staticmethod
    def _features(a: np.ndarray, b: np.ndarray, e1: float, e2: float) -> np.ndarray:
        """Build features: [1, a^e1, b^e2, (a^e1)*(b^e2), (a^e1)^2, (b^e2)^2]"""
        a_e1 = np.maximum(a ** e1, 1e-10)  # avoid negative/zero values
        b_e2 = np.maximum(b ** e2, 1e-10)
        return np.column_stack([
            np.ones_like(a),
            a_e1,
            b_e2,
            a_e1 * b_e2,
            a_e1 ** 2,
            b_e2 ** 2
        ])

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray,
            ridge: float = 1e-6, penalty_type: str = 'l2',
            penalty_lambda: float = 0.01) -> 'GMDHNeuronConstrained':
        """Optimize e1, e2 via Nelder-Mead, then fit coefficients w."""
        
        def objective(exponents):
            e1, e2 = exponents
            # Bound exponents to reasonable range
            if e1 < 0.1 or e1 > 5.0 or e2 < 0.1 or e2 > 5.0:
                return 1e10  # penalty for out-of-bounds
            
            X = self._features(a, b, e1, e2)
            try:
                A = X.T @ X + ridge * np.eye(6)
                w = np.linalg.solve(A, X.T @ y)
                residuals = y - X @ w
                rmse = np.sqrt(np.mean(residuals ** 2))
            except np.linalg.LinAlgError:
                return 1e10
            
            # Add regularization penalty on exponents
            if penalty_type == 'l2':
                penalty = penalty_lambda * (e1**2 + e2**2)
            elif penalty_type == 'l1':
                penalty = penalty_lambda * (abs(e1) + abs(e2))
            else:
                penalty = 0
            
            return rmse + penalty
        
        # Optimize exponents
        result = minimize(objective, x0=[1.0, 1.0], method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
        self.e1, self.e2 = result.x
        
        # Refit w with optimal exponents
        X = self._features(a, b, self.e1, self.e2)
        A = X.T @ X + ridge * np.eye(6)
        self.w = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._features(a, b, self.e1, self.e2) @ self.w

    def error(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - self.predict(a, b)) ** 2)))


class GMDHLayerConstrained:
    """Layer with constrained fractional exponents."""

    def __init__(self, n_keep: int, ridge: float = 1e-6,
                 penalty_type: str = 'l2', penalty_lambda: float = 0.01) -> None:
        self.n_keep = n_keep
        self.ridge = ridge
        self.penalty_type = penalty_type
        self.penalty_lambda = penalty_lambda
        self.neurons: List[GMDHNeuronConstrained] = []
        self.best_error: float = np.inf

    def fit(self, Z_tr: np.ndarray, Z_se: np.ndarray,
            y_tr: np.ndarray, y_se: np.ndarray):
        """Fit layer with exponent optimization on train, score on selection."""
        m = Z_tr.shape[1]
        candidates = []

        for i, j in combinations(range(m), 2):
            # Optimize exponents on training data
            nu = GMDHNeuronConstrained(i, j).fit(
                Z_tr[:, i], Z_tr[:, j], y_tr, 
                self.ridge, self.penalty_type, self.penalty_lambda
            )
            # Score on selection data
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


class GMDH_Constrained:
    """GMDH with constrained fractional exponents (e1, e2) + penalty."""
    
    def __init__(self, n_keep: int, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 penalty_type: str = 'l2', penalty_lambda: float = 0.01,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.penalty_type = penalty_type
        self.penalty_lambda = penalty_lambda
        self.random_state = random_state
        
        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.layers_: List[GMDHLayerConstrained] = []
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
        X_std, y_std = self._standardize(X, y)
        rng = np.random.default_rng(self.random_state)
        n = X_std.shape[0]
        indices = rng.permutation(n)
        split = int(n * self.training_split)
        tr_idx, se_idx = indices[:split], indices[split:]

        Z_tr, Z_se = X_std[tr_idx], X_std[se_idx]
        y_tr, y_se = y_std[tr_idx], y_std[se_idx]

        self.layers_ = []
        best_error = np.inf
        best_layer_idx = -1
        no_improve_count = 0

        for layer_num in range(self.max_layers):
            if Z_tr.shape[1] < 2:
                break

            layer = GMDHLayerConstrained(
                self.n_keep, self.ridge, self.penalty_type, self.penalty_lambda
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
        X_std = (X - self.x_mean_) / self.x_std_
        Z = X_std
        for layer in self.layers_:
            Z = layer.predict(Z)
        return Z[:, 0] * self.y_std_ + self.y_mean_
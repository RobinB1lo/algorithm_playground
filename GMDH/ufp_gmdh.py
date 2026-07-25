import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from typing import Optional, List, Tuple


class GMDHNeuronUnconstrained:
    """Neuron with 6 independent fractional exponents."""
    __slots__ = ('i', 'j', 'exponents', 'w')

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j
        self.exponents: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.w: Optional[np.ndarray] = None

    @staticmethod
    def _features(a: np.ndarray, b: np.ndarray, exponents: np.ndarray) -> np.ndarray:
        """Build features: [1, a^e1, b^e2, a^e3*b^e4, a^e5, b^e6]"""
        e1, e2, e3, e4, e5, e6 = exponents
        a_e1 = np.maximum(a ** e1, 1e-10)
        b_e2 = np.maximum(b ** e2, 1e-10)
        a_e3 = np.maximum(a ** e3, 1e-10)
        b_e4 = np.maximum(b ** e4, 1e-10)
        a_e5 = np.maximum(a ** e5, 1e-10)
        b_e6 = np.maximum(b ** e6, 1e-10)
        return np.column_stack([
            np.ones_like(a),
            a_e1,
            b_e2,
            a_e3 * b_e4,
            a_e5,
            b_e6
        ])

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray,
            ridge: float = 1e-6, penalty_type: str = 'l2',
            penalty_lambda: float = 0.01) -> 'GMDHNeuronUnconstrained':
        """Optimize 6 exponents via Nelder-Mead, then fit w."""
        
        def objective(exponents):
            # Bound exponents
            if np.any(exponents < 0.1) or np.any(exponents > 5.0):
                return 1e10
            
            X = self._features(a, b, exponents)
            try:
                A = X.T @ X + ridge * np.eye(6)
                w = np.linalg.solve(A, X.T @ y)
                residuals = y - X @ w
                rmse = np.sqrt(np.mean(residuals ** 2))
            except np.linalg.LinAlgError:
                return 1e10
            
            # Penalty on all exponents
            if penalty_type == 'l2':
                penalty = penalty_lambda * np.sum(exponents ** 2)
            elif penalty_type == 'l1':
                penalty = penalty_lambda * np.sum(np.abs(exponents))
            else:
                penalty = 0
            
            return rmse + penalty
        
        # Optimize all 6 exponents
        result = minimize(objective, x0=self.exponents, method='Nelder-Mead',
                         options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-4})
        self.exponents = result.x
        
        # Refit w with optimal exponents
        X = self._features(a, b, self.exponents)
        A = X.T @ X + ridge * np.eye(6)
        self.w = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._features(a, b, self.exponents) @ self.w

    def error(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - self.predict(a, b)) ** 2)))


class GMDHLayerUnconstrained:
    """Layer with 6 independent exponents, using k-fold CV."""

    def __init__(self, n_keep: int, ridge: float = 1e-6,
                 penalty_type: str = 'l2', penalty_lambda: float = 0.01,
                 k_folds: int = 5) -> None:
        self.n_keep = n_keep
        self.ridge = ridge
        self.penalty_type = penalty_type
        self.penalty_lambda = penalty_lambda
        self.k_folds = k_folds
        self.neurons: List[GMDHNeuronUnconstrained] = []
        self.best_error: float = np.inf

    def _get_kfold_indices(self, n: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate k-fold split indices."""
        indices = rng.permutation(n)
        fold_sizes = np.full(self.k_folds, n // self.k_folds, dtype=int)
        fold_sizes[:n % self.k_folds] += 1
        
        current = 0
        folds = []
        for fold_size in fold_sizes:
            val_idx = indices[current:current + fold_size]
            train_idx = np.concatenate((indices[:current], indices[current + fold_size:]))
            folds.append((train_idx, val_idx))
            current += fold_size
        return folds

    def fit(self, Z: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        """Fit layer using k-fold CV for exponent optimization."""
        m = Z.shape[1]
        n = Z.shape[0]
        folds = self._get_kfold_indices(n, rng)
        
        candidates = []

        for i, j in combinations(range(m), 2):
            avg_error = 0.0
            
            # Use CV to estimate generalization error with optimized exponents
            for tr_idx, val_idx in folds:
                nu = GMDHNeuronUnconstrained(i, j).fit(
                    Z[tr_idx, i], Z[tr_idx, j], y[tr_idx],
                    self.ridge, self.penalty_type, self.penalty_lambda
                )
                avg_error += nu.error(Z[val_idx, i], Z[val_idx, j], y[val_idx])
            
            avg_error /= self.k_folds
            candidates.append((avg_error, i, j))

        candidates.sort(key=lambda c: c[0])
        survivors = candidates[:self.n_keep]
        
        self.best_error = survivors[0][0]
        self.neurons = []
        Z_next = []
        
        # Retrain survivors on full dataset
        for err, i, j in survivors:
            final_nu = GMDHNeuronUnconstrained(i, j).fit(
                Z[:, i], Z[:, j], y,
                self.ridge, self.penalty_type, self.penalty_lambda
            )
            self.neurons.append(final_nu)
            Z_next.append(final_nu.predict(Z[:, i], Z[:, j]))
        
        return np.column_stack(Z_next)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.column_stack([
            nu.predict(Z[:, nu.i], Z[:, nu.j]) for nu in self.neurons
        ])


class GMDH_Unconstrained:
    """GMDH with 6 independent fractional exponents + CV + penalty."""
    
    def __init__(self, n_keep: int, max_layers: int = 10, ridge: float = 1e-6,
                 penalty_type: str = 'l2', penalty_lambda: float = 0.01,
                 k_folds: int = 5, patience: int = 0,
                 random_state: Optional[int] = None) -> None:
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.penalty_type = penalty_type
        self.penalty_lambda = penalty_lambda
        self.k_folds = k_folds
        self.patience = patience
        self.random_state = random_state
        
        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.layers_: List[GMDHLayerUnconstrained] = []
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

        Z = X_std
        y_target = y_std

        self.layers_ = []
        best_error = np.inf
        best_layer_idx = -1
        no_improve_count = 0

        for layer_num in range(self.max_layers):
            if Z.shape[1] < 2:
                break

            layer = GMDHLayerUnconstrained(
                self.n_keep, self.ridge, self.penalty_type, 
                self.penalty_lambda, self.k_folds
            )
            Z = layer.fit(Z, y_target, rng)
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
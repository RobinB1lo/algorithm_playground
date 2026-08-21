import numpy as np
from itertools import combinations
from typing import Optional, List, Tuple


class GMDHNeuron_CV:
    __slots__ = ('i', 'j', 'w')

    def __init__(self, i: int, j: int) -> None:
        self.i = i
        self.j = j
        self.w: Optional[np.ndarray] = None

    @staticmethod
    def _features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(a), a, b, a * b, a ** 2, b ** 2])

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray,
            ridge: float = 1e-6) -> 'GMDHNeuron_CV':
        X = self._features(a, b)
        A = X.T @ X + ridge * np.eye(6)
        self.w = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._features(a, b) @ self.w

    def error(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - self.predict(a, b)) ** 2)))


class GMDHLayer_CV:
    """Layer using k-fold CV for robust neuron evaluation."""

    def __init__(self, n_keep: int, ridge: float = 1e-6, k_folds: int = 5) -> None:
        self.n_keep = n_keep
        self.ridge = ridge
        self.k_folds = k_folds
        self.neurons: List[GMDHNeuron_CV] = []
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

    def fit(self, Z_tr: np.ndarray, Z_se: np.ndarray,
            y_tr: np.ndarray, y_se: np.ndarray, rng: np.random.Generator):
        """Fit layer: use CV on selection set to evaluate neurons robustly."""
        m = Z_tr.shape[1]
        candidates = []
        
        # Get CV folds from SELECTION set only (holds out from everything)
        folds = self._get_kfold_indices(Z_se.shape[0], rng)

        for i, j in combinations(range(m), 2):
            # Fit on TRAINING set
            nu = GMDHNeuron_CV(i, j).fit(Z_tr[:, i], Z_tr[:, j], y_tr, self.ridge)
            
            # Evaluate with k-fold CV on SELECTION set
            cv_errors = []
            for tr_fold_idx, val_fold_idx in folds:
                # Refit on this fold's train portion
                nu_fold = GMDHNeuron_CV(i, j).fit(
                    Z_se[tr_fold_idx, i], Z_se[tr_fold_idx, j], y_se[tr_fold_idx], 
                    self.ridge
                )
                # Evaluate on validation fold
                fold_error = nu_fold.error(Z_se[val_fold_idx, i], Z_se[val_fold_idx, j], y_se[val_fold_idx])
                cv_errors.append(fold_error)
            
            avg_cv_error = np.mean(cv_errors)
            
            # Now refit on full training set for final predictions
            nu_final = GMDHNeuron_CV(i, j).fit(Z_tr[:, i], Z_tr[:, j], y_tr, self.ridge)
            out_tr = nu_final.predict(Z_tr[:, i], Z_tr[:, j])
            out_se = nu_final.predict(Z_se[:, i], Z_se[:, j])
            
            candidates.append((avg_cv_error, nu_final, out_tr, out_se))

        # Keep n_keep best by CV error
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


class GMDH_CV:
    """GMDH with k-fold CV for robust neuron selection (with proper train/selection split)."""
    
    def __init__(self, n_keep: int, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, k_folds: int = 5, patience: int = 0,
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
            Fraction for train vs selection split (e.g., 0.5 = 50/50)
        k_folds : int
            Number of folds for CV evaluation
        patience : int
            Patience for early stopping
        random_state : int, optional
            Random seed
        """
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.k_folds = k_folds
        self.patience = patience
        self.random_state = random_state
        
        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.layers_: List[GMDHLayer_CV] = []
        self.best_error_: float = np.inf

    def _standardize(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Standardize X and y."""
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-12
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() + 1e-12

        X_std = (X - self.x_mean_) / self.x_std_
        y_std = (y - self.y_mean_) / self.y_std_
        return X_std, y_std
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GMDH-CV network."""
        # Standardize
        X_std, y_std = self._standardize(X, y)

        # CRITICAL: Split data once at the beginning (like vanilla GMDH)
        # This maintains the external criterion principle
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
        best_error = np.inf
        best_layer_idx = -1
        no_improve_count = 0

        for layer_num in range(self.max_layers):
            # Stop if we can't form pairs
            if Z_tr.shape[1] < 2:
                break

            # Build and fit layer with CV
            layer = GMDHLayer_CV(self.n_keep, self.ridge, self.k_folds)
            Z_tr, Z_se = layer.fit(Z_tr, Z_se, y_tr, y_se, rng)
            self.layers_.append(layer)

            # Check improvement
            if layer.best_error < best_error - 1e-12:
                best_error = layer.best_error
                best_layer_idx = layer_num
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count > self.patience:
                break
        
        # Trim to best layer
        self.layers_ = self.layers_[:best_layer_idx + 1]
        self.best_error_ = best_error

    def _neuron_equation(self, neuron: GMDHNeuron_CV, feature_names: List[str]) -> str:
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
        """Print the GMDH-CV network's final polynomial equation after fitting."""
        if not self.layers_:
            print("No layers fitted.")
            return
        
        n_features = len(self.x_mean_)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        
        print("\n" + "=" * 80)
        print("GMDH-CV Network: Final Polynomial Equation")
        print(f"(Neurons selected via {self.layers_[0].k_folds}-fold CV on selection set)")
        print("=" * 80)
        
        # Track feature names through layers
        current_features = feature_names.copy()
        layer_outputs = {}  # For reference if needed
        
        for layer_idx, layer in enumerate(self.layers_):
            print(f"\n--- Layer {layer_idx + 1} ---")
            print(f"Best CV Error: {layer.best_error:.6f}")
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
        """Predict on new data."""
        X_std = (X - self.x_mean_) / self.x_std_

        # Replay through layers
        Z = X_std
        for layer in self.layers_:
            Z = layer.predict(Z)
        
        # De-standardize
        return Z[:, 0] * self.y_std_ + self.y_mean_
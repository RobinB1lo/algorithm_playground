import numpy as np
from itertools import combinations, count
from typing import Optional, List, Tuple


class LBNode:
    """Base class for any node in the look-back GMDH DAG (raw input or neuron)."""
    _id_counter = count()

    def __init__(self, ancestors: frozenset) -> None:
        self.id = next(LBNode._id_counter)
        self.ancestors = ancestors  # frozenset of ancestor node ids (transitive)

    def evaluate(self, X_std: np.ndarray, cache: dict) -> np.ndarray:
        raise NotImplementedError


class RawInputNode(LBNode):
    """A raw (standardized) input feature - a leaf of the DAG."""

    def __init__(self, feature_index: int) -> None:
        super().__init__(ancestors=frozenset())
        self.feature_index = feature_index

    def evaluate(self, X_std: np.ndarray, cache: dict) -> np.ndarray:
        if self.id in cache:
            return cache[self.id]
        val = X_std[:, self.feature_index]
        cache[self.id] = val
        return val


class LookBackNeuron(LBNode):
    """A quadratic GMDH neuron whose two parents can be ANY earlier node -
    a raw input, a previous layer's neuron, or a node from several layers back -
    as long as neither parent was used to build the other."""

    def __init__(self, parent_a: LBNode, parent_b: LBNode) -> None:
        ancestors = (parent_a.ancestors | parent_b.ancestors
                     | {parent_a.id, parent_b.id})
        super().__init__(ancestors=ancestors)
        self.parent_a = parent_a
        self.parent_b = parent_b
        self.w: Optional[np.ndarray] = None

    @staticmethod
    def _features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(a), a, b, a * b, a ** 2, b ** 2])

    def fit(self, a_tr: np.ndarray, b_tr: np.ndarray, y_tr: np.ndarray,
            ridge: float = 1e-6) -> 'LookBackNeuron':
        X = self._features(a_tr, b_tr)
        A = X.T @ X + ridge * np.eye(6)
        self.w = np.linalg.solve(A, X.T @ y_tr)
        return self

    def _predict_from_values(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._features(a, b) @ self.w

    def error_from_values(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - self._predict_from_values(a, b)) ** 2)))

    def evaluate(self, X_std: np.ndarray, cache: dict) -> np.ndarray:
        if self.id in cache:
            return cache[self.id]
        a = self.parent_a.evaluate(X_std, cache)
        b = self.parent_b.evaluate(X_std, cache)
        val = self._predict_from_values(a, b)
        cache[self.id] = val
        return val


class GMDH_LookBack:
    """GMDH variant allowing later layers to connect back to nodes from any
    earlier layer (including raw inputs), provided the look-back node was not
    itself used to construct the forward node it's pairing with."""

    def __init__(self, n_keep: int, max_layers: int = 10, ridge: float = 1e-6,
                 training_split: float = 0.5, patience: int = 0,
                 lookback_scope: str = 'all',
                 max_lookback_candidates: Optional[int] = None,
                 random_state: Optional[int] = None) -> None:
        """
        Parameters:
        -----------
        n_keep : int
            Number of best neurons to keep per layer (the new "frontier")
        max_layers : int
            Maximum number of layers to build
        ridge : float
            Ridge regression parameter
        training_split : float
            Train/selection split ratio
        patience : int
            Early stopping patience (non-improving layers tolerated)
        lookback_scope : str
            'all' - frontier nodes may pair with ANY earlier node (raw inputs
                    or any previous layer's surviving neurons)
            'inputs_only' - frontier nodes may only look back to raw inputs
                    (cheaper, and closest to the diagram's v4 example)
        max_lookback_candidates : int, optional
            If set, randomly subsample the look-back pool to this size per
            layer to control combinatorial blow-up as the DAG grows.
        random_state : int, optional
            Random seed
        """
        self.n_keep = n_keep
        self.max_layers = max_layers
        self.ridge = ridge
        self.training_split = training_split
        self.patience = patience
        self.lookback_scope = lookback_scope
        self.max_lookback_candidates = max_lookback_candidates
        self.random_state = random_state

        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.output_node_: Optional[LBNode] = None
        self.best_error_: float = np.inf
        self.all_nodes_: List[LBNode] = []  # kept for introspection/debugging

    def _standardize(self, X: np.ndarray, y: np.ndarray) -> tuple:
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-12
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() + 1e-12

        X_std = (X - self.x_mean_) / self.x_std_
        y_std = (y - self.y_mean_) / self.y_std_
        return X_std, y_std

    def _lookback_pool(self, all_nodes: List[LBNode], frontier: List[LBNode]) -> List[LBNode]:
        frontier_ids = {n.id for n in frontier}
        pool = [n for n in all_nodes if n.id not in frontier_ids]
        if self.lookback_scope == 'inputs_only':
            pool = [n for n in pool if isinstance(n, RawInputNode)]
        return pool

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_std, y_std = self._standardize(X, y)

        rng = np.random.default_rng(self.random_state)
        n = X_std.shape[0]
        indices = rng.permutation(n)
        split = int(n * self.training_split)
        tr_idx = indices[:split]
        se_idx = indices[split:]

        # Layer 0: raw input nodes
        raw_nodes = [RawInputNode(k) for k in range(X_std.shape[1])]
        values_tr = {node.id: X_std[tr_idx, node.feature_index] for node in raw_nodes}
        values_se = {node.id: X_std[se_idx, node.feature_index] for node in raw_nodes}
        y_tr, y_se = y_std[tr_idx], y_std[se_idx]

        all_nodes: List[LBNode] = list(raw_nodes)
        frontier: List[LBNode] = list(raw_nodes)

        best_error = np.inf
        best_output_node = None
        no_improve_count = 0

        for layer_num in range(self.max_layers):
            if len(frontier) < 1:
                break

            # Standard same-layer combos (vanilla behavior)
            candidate_pairs: List[Tuple[LBNode, LBNode]] = list(combinations(frontier, 2))

            # Look-back combos: frontier node x eligible earlier node
            lookback_pool = self._lookback_pool(all_nodes, frontier)
            if self.max_lookback_candidates is not None and len(lookback_pool) > self.max_lookback_candidates:
                idx = rng.choice(len(lookback_pool), self.max_lookback_candidates, replace=False)
                lookback_pool = [lookback_pool[i] for i in idx]

            for f in frontier:
                for l in lookback_pool:
                    if l.id in f.ancestors:      # l was used to build f - skip, would be circular
                        continue
                    candidate_pairs.append((f, l))

            if not candidate_pairs:
                break

            scored = []
            for a, b in candidate_pairs:
                a_tr, b_tr = values_tr[a.id], values_tr[b.id]
                a_se, b_se = values_se[a.id], values_se[b.id]
                neuron = LookBackNeuron(a, b).fit(a_tr, b_tr, y_tr, self.ridge)
                err = neuron.error_from_values(a_se, b_se, y_se)
                scored.append((err, neuron, a, b))

            scored.sort(key=lambda c: c[0])
            survivors = scored[:self.n_keep]
            layer_best_error = survivors[0][0]

            new_frontier = []
            for err, neuron, a, b in survivors:
                values_tr[neuron.id] = neuron._predict_from_values(values_tr[a.id], values_tr[b.id])
                values_se[neuron.id] = neuron._predict_from_values(values_se[a.id], values_se[b.id])
                all_nodes.append(neuron)
                new_frontier.append(neuron)

            frontier = new_frontier

            if layer_best_error < best_error - 1e-12:
                best_error = layer_best_error
                best_output_node = survivors[0][1]
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count > self.patience:
                break

        self.output_node_ = best_output_node
        self.best_error_ = best_error
        self.all_nodes_ = all_nodes

    def _build_node_name_map(self) -> dict:
        """Map node IDs to human-readable names for equation printing."""
        name_map = {}
        for node in self.all_nodes_:
            if isinstance(node, RawInputNode):
                name_map[node.id] = f"x{node.feature_index}"
            else:
                name_map[node.id] = f"n{node.id}"
        return name_map

    def _node_equation_string(self, node: LBNode, name_map: dict, memo: dict) -> str:
        """Recursively build equation string for a node."""
        if node.id in memo:
            return memo[node.id]
        
        if isinstance(node, RawInputNode):
            result = name_map[node.id]
        else:
            # It's a LookBackNeuron
            a_str = self._node_equation_string(node.parent_a, name_map, memo)
            b_str = self._node_equation_string(node.parent_b, name_map, memo)
            w = node.w
            
            # Build polynomial terms
            terms = []
            if abs(w[0]) > 1e-10:
                terms.append(f"{w[0]:+.6f}")
            if abs(w[1]) > 1e-10:
                terms.append(f"{w[1]:+.6f}*({a_str})")
            if abs(w[2]) > 1e-10:
                terms.append(f"{w[2]:+.6f}*({b_str})")
            if abs(w[3]) > 1e-10:
                terms.append(f"{w[3]:+.6f}*({a_str})*({b_str})")
            if abs(w[4]) > 1e-10:
                terms.append(f"{w[4]:+.6f}*({a_str})²")
            if abs(w[5]) > 1e-10:
                terms.append(f"{w[5]:+.6f}*({b_str})²")
            
            expr = " ".join(terms) if terms else "0"
            expr = expr.lstrip("+").strip()
            result = f"({expr})"
        
        memo[node.id] = result
        return result

    def print_equation(self, feature_names: Optional[List[str]] = None) -> None:
        """Print the GMDH-LookBack network's final DAG-based equation."""
        if self.output_node_ is None:
            print("No output node fitted.")
            return
        
        n_features = len(self.x_mean_)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]
        
        # Build name map
        name_map = {}
        for node in self.all_nodes_:
            if isinstance(node, RawInputNode):
                name_map[node.id] = feature_names[node.feature_index]
            else:
                name_map[node.id] = f"n{node.id}"
        
        print("\n" + "=" * 80)
        print("GMDH-LookBack Network: Final DAG-Based Equation")
        print(f"Scope: {self.lookback_scope}")
        print("=" * 80)
        
        print(f"\nDAG Structure ({len(self.all_nodes_)} total nodes):")
        print(f"  Raw inputs: {sum(1 for n in self.all_nodes_ if isinstance(n, RawInputNode))}")
        print(f"  Neurons: {sum(1 for n in self.all_nodes_ if isinstance(n, LookBackNeuron))}")
        
        print(f"\n--- Neuron Definitions (selected for DAG) ---")
        for node in self.all_nodes_:
            if isinstance(node, LookBackNeuron):
                a_name = name_map[node.parent_a.id]
                b_name = name_map[node.parent_b.id]
                w = node.w
                
                terms = []
                if abs(w[0]) > 1e-10:
                    terms.append(f"{w[0]:+.6f}")
                if abs(w[1]) > 1e-10:
                    terms.append(f"{w[1]:+.6f}*{a_name}")
                if abs(w[2]) > 1e-10:
                    terms.append(f"{w[2]:+.6f}*{b_name}")
                if abs(w[3]) > 1e-10:
                    terms.append(f"{w[3]:+.6f}*{a_name}*{b_name}")
                if abs(w[4]) > 1e-10:
                    terms.append(f"{w[4]:+.6f}*{a_name}²")
                if abs(w[5]) > 1e-10:
                    terms.append(f"{w[5]:+.6f}*{b_name}²")
                
                eq = " ".join(terms) if terms else "0"
                eq = eq.lstrip("+").strip()
                print(f"\n  {name_map[node.id]} = {eq}")
        
        print(f"\n" + "=" * 80)
        print("FINAL PREDICTION (standardized, expanded form):")
        memo = {}
        expanded = self._node_equation_string(self.output_node_, name_map, memo)
        print(f"  y_standardized = {expanded}")
        
        print("\nDE-STANDARDIZE TO ORIGINAL SCALE:")
        print(f"  y_predicted = y_standardized × {self.y_std_:.6f} + {self.y_mean_:.6f}")
        print("=" * 80 + "\n")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data by recursively walking the DAG from the
        chosen output node back to raw inputs, memoizing shared ancestors."""
        X_std = (X - self.x_mean_) / self.x_std_
        cache: dict = {}
        y_pred_std = self.output_node_.evaluate(X_std, cache)
        return y_pred_std * self.y_std_ + self.y_mean_
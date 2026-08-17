"""
Wrapper around the official GMDHreg package from CRAN.
"""

import os
import sys
import numpy as np
from typing import Optional

os.environ.setdefault("R_HOME", "/Library/Frameworks/R.framework/Resources")

R_AVAILABLE = False
R_ERROR_MSG = None

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    _r_version = ro.r('R.version.string')[0]
    print(f"[gmdhref_wrapper] R version: {_r_version}", file=sys.stderr)

    gmdh_pkg = importr("GMDHreg")
    assert hasattr(gmdh_pkg, 'gmdh_combi')
    assert hasattr(gmdh_pkg, 'predict_combi')

    R_AVAILABLE = True
    print("[gmdhref_wrapper] GMDHreg loaded successfully.", file=sys.stderr)

except Exception as e:
    R_ERROR_MSG = f"{type(e).__name__}: {e}"
    print(f"[gmdhref_wrapper] R/GMDHreg NOT available: {R_ERROR_MSG}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    gmdh_pkg = None


class GMDHreg_Wrapper:
    def __init__(self, criteria: str = "test", G: int = 2,
                 random_state: Optional[int] = None):
        if not R_AVAILABLE:
            raise RuntimeError(
                f"R reference not available.\n"
                f"Reason: {R_ERROR_MSG}\n"
                f"Python: {sys.executable}\n"
                f"R_HOME: {os.environ.get('R_HOME', 'NOT SET')}"
            )
        self.criteria = criteria
        self.G = G
        self.random_state = random_state
        self.model_ = None
        self.x_mean_ = None
        self.x_std_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self.feature_names_ = None

    def _numpy_to_r_matrix(self, arr: np.ndarray, colnames: list):
        """Convert a 2-D numpy array to an R matrix with column names."""
        # R's matrix() fills by column by default; byrow=TRUE matches NumPy C-order
        r_matrix = ro.r.matrix(
            arr.flatten().tolist(),
            nrow=arr.shape[0],
            ncol=arr.shape[1],
            byrow=True
        )
        # Set column names using R's colnames<- assignment function
        ro.r['colnames<-'](r_matrix, ro.vectors.StrVector(colnames))
        return r_matrix

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GMDHreg_Wrapper':
        if self.random_state is not None:
            ro.r(f'set.seed({self.random_state})')

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-12
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() + 1e-12

        X_std = (X - self.x_mean_) / self.x_std_
        y_std = (y - self.y_mean_) / self.y_std_

        n_features = X_std.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(n_features)]

        # Build R objects directly — no pandas, no py2rpy conversion
        X_r = self._numpy_to_r_matrix(X_std, self.feature_names_)
        y_r = ro.vectors.FloatVector(y_std.tolist())

        self.model_ = gmdh_pkg.gmdh_combi(
            X_r, y_r, G=self.G, criteria=self.criteria
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model has not been fitted.")

        X_std = (X - self.x_mean_) / self.x_std_
        X_r = self._numpy_to_r_matrix(X_std, self.feature_names_)

        y_pred_r = gmdh_pkg.predict_combi(self.model_, newdata=X_r)
        y_pred_std = np.asarray(y_pred_r, dtype=float)

        y_pred = y_pred_std * self.y_std_ + self.y_mean_
        return y_pred
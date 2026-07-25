import numpy as np

class LinearRegression1:
    def __init__(self, X_raw: list, y_raw: list) -> None:
        self.X = self._features(X_raw)
        self.y = y_raw
        self.w = None 
        
    def fit(self, X_raw: list, y: list) -> None:
        X = self._add_intercept_column(X_raw)
        A = X.T @ X
        b = X.T @ y
        self.w = np.linalg.solve(A, b)
        return self.w


    def predict(self, x: list) -> None:
        pass


class LinearRegression2:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y
    
    def cost_function(self, w, b):
        m = len(self.x)
        cost_sum = 0

        for i in range(m):
            f = w * self.x[i] + b
            cost = (f - self.y[i]) ** 2
            cost_sum += cost
        
        total_cost = (1/(2*m)) * cost_sum
        return total_cost

    def gradient_function(self, w, b):
        m = len(self.x)
        dc_dw = 0
        dc_db = 0

        for i in range(m):
            f = w * self.x[i] + b

            dc_dw += (f - self.y[i]) * self.x[i]
            dc_db += (f - self.y[i])

        dc_dw = (1/m) * dc_dw
        dc_db = (1/m) * dc_db

        return dc_dw, dc_db

    def gradient_descent(alpha, iters):
        pass

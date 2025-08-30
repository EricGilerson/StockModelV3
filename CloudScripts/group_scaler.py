
import numpy as np


class GroupScaler:
    #Mimics sklearn's interface but shares one center_/scale_ across a group of features.

    def __init__(self, center: float, scale: float, n_cols: int, mode: str = "standard", pre: str = None):
        self.center_ = np.full(n_cols, center, dtype=np.float32)
        self.scale_ = np.full(n_cols, scale, dtype=np.float32)
        self.mode = mode
        self.pre = pre  # pre-transform type (e.g., "log")
        self.n_cols = n_cols

    def transform(self, X):
        # Handle NaN and Inf values during transformation
        X = np.nan_to_num(X, nan=self.center_[0], posinf=self.center_[0] + 10 * self.scale_[0],
                          neginf=self.center_[0] - 10 * self.scale_[0])

        # Apply pre-transform if specified
        if self.pre == "log":
            X = np.sign(X) * np.log1p(np.abs(X))

        return (X - self.center_) / self.scale_

    def inverse_transform(self, X):
        # Handle NaN and Inf values during inverse transformation
        X = np.nan_to_num(X, nan=0, posinf=10, neginf=-10)
        result = X * self.scale_ + self.center_

        # Apply inverse of pre-transform if specified
        if self.pre == "log":
            result = np.sign(result) * (np.exp(np.abs(result)) - 1)

        return result

    def fit_transform(self, X):
        return self.transform(X)
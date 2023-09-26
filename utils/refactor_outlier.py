import numpy as np

def refactor_outlier(X, col, top, factor):

    # Finds outlier
    outlier = max(X[:, col]) if top else min(X[:, col])
    outlier_row = np.argwhere(X[:, col] == outlier)

    # Refactors assumed outlier error from data
    X[outlier_row] = outlier / factor
    return None

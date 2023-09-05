import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def pca_analysis(X, y):
    # Subtract mean value from data
    N = len(y)
    # Standardize Data
    X_std = X - X.mean(axis=0)
    X_std = X_std / X_std.std(axis=0)

    # PCA by computing SVD of Y
    U, S, V = svd(X_std, full_matrices=False)


    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()

    threshold = 0.9

    # Plot variance explained
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()

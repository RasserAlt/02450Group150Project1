import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def pca_analysis(X_std):

    # PCA by computing SVD of Y
    U, S, Vh = svd(X_std, full_matrices=False)


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

    # of the vector V. So, for us to obtain the correct V, we transpose:
    V = Vh.T

    # Project the centered data onto principal component space
    Z = Y @ V

    # Indices of the principal components to be plotted
    i = 0
    j = 1

    # Plot PCA of the data
    f = plt.figure()
    plt.title('NanoNose data: PCA')
    # Z = array(Z)
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y == c
        plt.plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
    plt.legend(categoryNames)
    plt.xlabel('PC{0}'.format(i + 1))
    plt.ylabel('PC{0}'.format(j + 1))

    # Output result to screen
    plt.show()

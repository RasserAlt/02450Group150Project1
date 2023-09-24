import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def pca_analysis(x0, Xy_std, category_names, category_dict):
    # PCA by computing SVD of Y
    U, S, Vh = svd(Xy_std, full_matrices=False)

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
    Z = Xy_std @ V

    # Indices of the principal components to be plotted
    i = 0
    j = 1

    # Plot PCA of the data
    f = plt.figure()
    plt.title('Abalone data: PCA')
    # Z = array(Z)
    x0 = x0.flatten()
    for c in range(len(category_dict)):
        # select indices belonging to class c:
        class_mask = x0 == c
        plt.plot(Z[class_mask, i], Z[class_mask, j], 'x', alpha=.5)
    plt.legend(category_names)
    plt.xlabel('PC{0}'.format(i + 1))
    plt.ylabel('PC{0}'.format(j + 1))

    # Output result to screen
    plt.show()

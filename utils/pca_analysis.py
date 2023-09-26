import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def pca_analysis(x0, Xy_std, category_names, attribute_names):
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

    # From the plot we can see only two principal components are needed to explain 90% of the variance
    # So for projecting the data we use teo first indices of the principal components
    i = 0
    j = 1

    # of the vector V. So, for us to obtain the correct V, we transpose:
    V = Vh.T

    # Project the centered data onto principal component space
    Z = Xy_std @ V

    # Plot PCA of the data
    f = plt.figure()
    plt.title('Abalone data: PCA')
    # Z = array(Z)
    x0 = x0.flatten()
    # Makes a different colors plot for each category, aka each sex
    for c in range(len(category_names)):
        # select indices belonging to category c:
        category_mask = x0 == c
        plt.plot(Z[category_mask, i], Z[category_mask, j], '.', alpha=.75-.25*c)
    plt.legend(category_names)
    plt.xlabel('PC{0}'.format(i + 1))
    plt.ylabel('PC{0}'.format(j + 1))

    # Output result to screen
    plt.show()

    # Plot to show the first two principal components coefficient on each continues attributes
    N, M = Xy_std.shape

    pcs = [0, 1]
    legendStrs = ['PC' + str(e + 1) for e in pcs]
    c = ['r', 'g']
    bw = .2
    r = np.arange(1, M + 1)
    for i in pcs:
        plt.bar(r + i * bw, V[:, i], width=bw)
    plt.xticks(r + bw, attribute_names)
    plt.ylabel('Component coefficients')
    plt.legend(legendStrs)
    plt.grid()
    plt.title('Abalone: PCA Component Coefficients')
    plt.show()

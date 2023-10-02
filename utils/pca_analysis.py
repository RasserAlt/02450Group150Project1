import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def pca_analysis(X_std, y, ring_class, variable_names):
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

    # From the plot we can see only two principal components are needed to explain 90% of the variance
    # So for projecting the data we use teo first indices of the principal components
    i = 0
    j = 1

    # of the vector V. So, for us to obtain the correct V, we transpose:
    V = Vh.T

    # Project the centered data onto principal component space
    Z = X_std @ V

    # Plot PCA of the data
    plt.figure()
    plt.title('Abalone data: PCA')
    y = y.flatten()
    ring_class_names = []
    for c in range(len(ring_class)-1):
        ring_class_names.append("[" + str(ring_class[c]) + ":" + str(ring_class[c + 1]) + "]")
        class_mask = []
        for h in range(len(y)):
            class_mask.append(ring_class[c] <= y[h] < ring_class[c+1])
        plt.plot(Z[class_mask, i], Z[class_mask, j], '.', alpha=0.5)
    ring_class_names.append("[" + str(ring_class[len(ring_class)-1]) + ":]")
    class_mask = []
    for h in range(len(y)):
        class_mask.append(ring_class[len(ring_class)-1] <= y[h])
    plt.plot(Z[class_mask, i], Z[class_mask, j], '.', alpha=0.5)
    plt.legend(ring_class_names)
    plt.xlabel('PC{0}'.format(i + 1))
    plt.ylabel('PC{0}'.format(j + 1))

    # Output result to screen
    plt.show()

    # Plot to show the first two principal components coefficient on each continuous attributes
    N, M = X_std.shape

    pcs = [0, 1, 2]
    legendStrs = ['PC' + str(e + 1) for e in pcs]
    c = ['r', 'g', 'b']
    bw = .2
    r = np.arange(1, M + 1)
    for i in pcs:
        plt.bar(r + i * bw, V[:, i], width=bw)
    plt.xticks(r + bw, variable_names)
    plt.ylabel('Component coefficients')
    plt.legend(legendStrs)
    plt.grid()
    plt.title('Abalone: PCA Component Coefficients')
    plt.show()

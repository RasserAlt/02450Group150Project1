# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)
import matplotlib.pyplot as plt
import numpy as np

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from .toolbox import rlr_validate


def baseline(y):
    y.mean()


def regularized_linear_regression(X, y, variable_names, lambdas):
    N, M = X.shape
    y = y.squeeze()
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
    variable_names = [u'Offset'] + variable_names
    M = M + 1
    K = 10

    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X,
                                                                                                      y,
                                                                                                      lambdas,
                                                                                                      K)
    figure(K, figsize=(12, 8))
    subplot(1, 2, 1)
    semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
    xlabel('Regularization factor')
    ylabel('Mean Coefficient Values')
    grid()
    # You can choose to display the legend, but it's omitted for a cleaner
    # plot, since there are many attributes
    # legend(attributeNames[1:], loc='best')

    subplot(1, 2, 2)
    title('Optimal lambda: {:}'.format(opt_lambda))
    loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
    xlabel('Regularization factor')
    ylabel('Squared error (crossvalidation)')
    legend(['Train error', 'Validation error'])
    grid()
    show()

    Xty = X.T @ y
    XtX = X.T @ X
    w_rlr = np.empty(M)

    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    for m in range(M):
        print('{:>11} {:>11}'.format(variable_names[m], np.round(w_rlr[m], 2)))
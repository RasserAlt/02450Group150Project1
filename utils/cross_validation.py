import torch
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)
import matplotlib.pyplot as plt
import numpy as np
import utils as ut

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from .toolbox import rlr_validate
from .toolbox import correlated_ttest

def two_layer_cross_validation(X, y, attribute_names, lambdas, h_range):
    N, M = X.shape
    y = y.squeeze()

    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 10
    k = 10
    CV = model_selection.KFold(K, shuffle=True)

    # Initialize variables
    # T = len(lambdas)
    opt_h = np.empty((K, 1))
    Error_test_ann = np.empty((K, 1))
    opt_lambda = np.empty((K, 1))
    w_rlr = np.empty((M+1, K))
    Error_test_rlr = np.empty((K, 1))
    Error_test_nofeatures = np.empty((K, 1))

    r_ann_rlr = np.empty((K, 1))
    r_rlr_nofeatures = np.empty((K, 1))
    r_nofeatures_ann = np.empty((K, 1))
    j = 0
    for train_index, test_index in CV.split(X, y):

        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        opt_val_err, opt_lambda[j], mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train,
                                                                                                          y_train,
                                                                                                          lambdas,
                                                                                                          k)
        Xoff  = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
        Xofftest = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)
        Xty = Xoff.T @ y_train
        XtX = Xoff.T @ Xoff

        # Compute mean squared error without using the input data at all
        Error_test_nofeatures[j] = np.square(y_test - y_test.mean()).sum(axis=0)*100 / y_test.shape[0]

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda[j] * np.eye(M+1)
        lambdaI[0, 0] = 0  # Do no regularize the bias term
        w_rlr[:, j] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_test_rlr[j] = np.square(y_test - Xofftest @ w_rlr[:, j]).sum(axis=0)*100 / y_test.shape[0]
        r_rlr_nofeatures[j] = np.mean(np.square(y_test - Xofftest @ w_rlr[:, j]) - np.square(y_test - y_test.mean()))

        # insert ANN
        Error_test_ann[j], min_net = ut.train_and_visualize_model(X_train, np.expand_dims(y_train, 1), attribute_names, h_range[0], 0.0001)
        opt_h[j] = h_range[0]
        print(y_test[1])
        print(torch.Tensor(X_test[1]))
        print(min_net(torch.Tensor(X_test)).detach().numpy())

        for h in h_range[1:]:
            error_rate, net = ut.train_and_visualize_model(X_train, np.expand_dims(y_train, 1), attribute_names, h, 0.0001)
            if error_rate < Error_test_ann[j]:
                Error_test_ann[j] = error_rate
                opt_h[j] = h
                min_net = net

        r_ann_rlr[j] = np.mean(np.square(y_test - Xofftest @ w_rlr[:, j]) - np.square(y_test - min_net(torch.Tensor(X_test)).detach().numpy()))

        r_nofeatures_ann[j] = np.mean(np.square(y_test - y_test.mean()) - np.square(y_test - min_net(torch.Tensor(X_test)).detach().numpy()))

        j += 1

    print('table:')
    for j in range(K):
        print('{:} {:} {:} {:} {:}'.format(opt_h[j], Error_test_ann[j], opt_lambda[j], Error_test_rlr[j], Error_test_nofeatures[j]))

    # Initialize parameters and run test appropriate for setup II
    alpha = 0.05
    rho = 1 / K
    p_rlr_nofeatures_setupII, CI_rlr_nofeatures_setupII = correlated_ttest(r_rlr_nofeatures, rho, alpha=alpha)
    p_ann_rlr_setupII, CI_ann_rlr_setupII = correlated_ttest(r_ann_rlr, rho, alpha=alpha)
    p_nofeatures_ann_setupII, CI_nofeatures_ann_setupII = correlated_ttest(r_nofeatures_ann, rho, alpha=alpha)

    print(p_rlr_nofeatures_setupII, CI_rlr_nofeatures_setupII)
    print(p_ann_rlr_setupII, CI_ann_rlr_setupII)
    print(p_nofeatures_ann_setupII, CI_nofeatures_ann_setupII)
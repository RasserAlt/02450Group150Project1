import utils as ut
import numpy as np


def main(file_name):
    # Load data
    # x0 is the first row of the data containing the discrete category variable Sex
    # y is the goal attribute we're interested in finding, here Rings
    # X are the remanding continuous variables
    x0, X, y, variable_names, class_names = ut.load_xls(file_name)

    # Summary Statistics
    # ut.summary_statistics(file_name)

    # Box plot for all continuous attributes
    ut.box_plot(X[:, :4], variable_names[:4])
    ut.box_plot(X[:, 4:], variable_names[4:])

    # Refactors the highest two height outliers by a factors of 10
    ut.refactor_outlier(X, 2, True, 10)
    ut.refactor_outlier(X, 2, True, 10)

    # Boxplot of refactored height
    ut.box_plot(X[:, :4], variable_names[:4])

    X = np.concatenate((x0, X), axis=1)

    # Standardized table
    X_std = X - X.mean(axis=0)
    X_std = X_std / X_std.std(axis=0)

    y_std = y - y.mean(axis=0)
    y_std = y_std / y_std.std(axis=0)

    # Histogram of continuous attributes
    Xy_std = np.concatenate((X_std, y_std), axis=1)
    ut.hist_plot(Xy_std, variable_names+['ring'])

    #categorizes the ring clas
    ring_class = [0, 6, 12, 18]


    # PCA Analysis
    ut.pca_analysis(X_std, y, ring_class, class_names + variable_names)


if __name__ == '__main__':
    main('data/abalone.xls')

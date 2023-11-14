import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_confidence_interval(a, b, c, d, confidence_level=0.95):
    odds_ratio = (b + c) / (a + d)
    log_odds_ratio = np.log(odds_ratio)

    se = np.sqrt(1 / (b + c) + 1 / (a + d))

    # Calculate critical value for the normal distribution
    z = norm.ppf(1 - (1 - confidence_level) / 2)

    # Calculate confidence interval for the log-odds ratio
    lower_bound = log_odds_ratio - z * se
    upper_bound = log_odds_ratio + z * se

    # Exponentiate to get the confidence interval for the odds ratio
    odds_ratio_ci = (np.exp(lower_bound), np.exp(upper_bound))

    return odds_ratio_ci

def mcnemar_models(model1, model2, y_test):

    n11 = 0
    n12 = 0
    n21 = 0
    n22 = 0

    for i in range(len(y_test)):
        if (model1[i] == y_test[i]) & (model2[i] == y_test[i]):
            n11 += 1
        if (model1[i] != y_test[i]) & (model2[i] != y_test[i]):
            n22 += 1
        if (model1[i] == y_test[i]) & (model2[i] != y_test[i]):
            n12 += 1
        if (model1[i] != y_test[i]) & (model2[i] == y_test[i]):
            n21 += 1

    contingency_table = [[n11, n12], [n21, n22]]
    # print(contingency_table)
    print('accuracy difference: {:.2f}'.format((n12 - n21) / len(y_test)))
    ci_lower, ci_upper = mcnemar_confidence_interval(n11, n12, n21, n22)
    print('CI: ({:.4f}, {:.4f})'.format(ci_lower, ci_upper))

    # Perform McNemar's test
    result = mcnemar(contingency_table, exact=True, correction=False)
    # Print the results
    print('p-value: {:.3f}'.format(result.pvalue))

def nested_crossvalidation(X,y):


    num_folds = 10
    # Create a KFold object
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    lr_param_grid = {
        'C': np.power(10., np.arange(-1, 1, 0.1)),
    }
    knn_param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    }

    # Create empty lists to store the results
    bl_errors = []
    lr_errors = []
    knn_errors = []

    lr_params = []
    knn_params = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Baseline model...
        baseline_model = DummyClassifier(strategy='most_frequent')
        baseline_model.fit(X_train, y_train)
        bl_pred = baseline_model.predict(X_test)

        # Logistic Regression...
        lr = LogisticRegression(random_state=42)
        lr_grid = GridSearchCV(lr, lr_param_grid, cv=5)
        lr_grid.fit(X_train, y_train)
        lr_best = lr_grid.best_estimator_
        lr_pred = lr_best.predict(X_test)


        # KNN...
        knn = KNeighborsClassifier()
        knn_grid = GridSearchCV(knn, knn_param_grid, cv=5)
        knn_grid.fit(X_train, y_train)
        knn_best = knn_grid.best_estimator_
        knn_pred = knn_best.predict(X_test)

        lr_params.append(lr_best.get_params()['C'])
        knn_params.append(knn_best.get_params()['n_neighbors'])

        # Calculate error rate
        lr_error = 1 - accuracy_score(y_test, lr_pred)
        knn_error = 1 - accuracy_score(y_test, knn_pred)
        bl_error = 1 - accuracy_score(y_test, bl_pred)

        y_test = y_test.to_numpy()

        # print("LR VS BL")
        # mcnemar_models(bl_pred, lr_pred, y_test)
        # print("KNN VS BL")
        # mcnemar_models(bl_pred, knn_pred, y_test)
        # print("LR VS KNN")
        # mcnemar_models(knn_pred, lr_pred, y_test)


        # Append the results to the lists
        lr_errors.append(lr_error)
        knn_errors.append(knn_error)
        bl_errors.append(bl_error)

    # Create a Pandas DataFrame with the results
    results = pd.DataFrame({'Baseline Error': bl_errors, 'Lambda': lr_params, 'Logistic Regression Error': lr_errors, 'K': knn_params, 'KNN Error': knn_errors})

    html = results.to_html()
    with open("results.html", "w") as f:
        f.write(html)
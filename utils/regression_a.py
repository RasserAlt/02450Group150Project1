from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt

def regression_a(X, y):
    lambda_values = np.logspace(-4, 4, 20)

    errors = []

    for lambda_value in lambda_values:
        model = Ridge(alpha=lambda_value)
        scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
        errors.append(-scores.mean())

    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, errors)
    plt.xlabel('Lambda')
    plt.ylabel('Generalization Error')
    plt.title('Generalization Error vs Lambda')
    plt.grid(True)
    plt.show()

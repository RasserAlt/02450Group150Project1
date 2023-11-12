from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import GridSearchCV
matplotlib.use('TkAgg')


def classification_baseline(X_train, X_test, y_train, y_test):
    # Create a DummyClassifier that predicts the majority class
    baseline_model = DummyClassifier(strategy='most_frequent')

    # Fit the baseline model on the training data
    baseline_model.fit(X_train, y_train)

    # Predict using the baseline model
    baseline_predictions = baseline_model.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, baseline_predictions)
    print("Accuracy:", accuracy)

    # Calculate and print F1 score
    f1 = f1_score(y_test, baseline_predictions)
    print("F1 Score:", f1)

    # Calculate and print recall
    recall = recall_score(y_test, baseline_predictions)
    print("Recall:", recall)

    # Calculate and print ROC AUC
    roc_auc = roc_auc_score(y_test, baseline_predictions)
    print("ROC AUC:", roc_auc)

def classification_regression(X_train, X_test, y_train, y_test):
    # Define a range of lambda values
    lambdas = np.power(10., np.arange(-2, 2, 0.1))

    # Create a logistic regression model
    model = LogisticRegression(penalty='l2', solver='liblinear')

    # Define the hyperparameter grid for grid search
    param_grid = {'C': 1.0 / lambdas}

    # Initialize GridSearchCV with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Check if the GridSearchCV has been fitted
    if grid_search.best_score_ is None:
        print("GridSearchCV did not find any suitable models.")
    else:
        # Get the best model and its metrics
        best_model = grid_search.best_estimator_

        # Check if the best model has a coef_ attribute
        if hasattr(best_model, 'coef_'):
            # Extract the coefficients of the logistic regression model
            coefficients = best_model.coef_[0]

            # Plot the relevance of the features
            plt.figure(figsize=(8, 6))
            plt.bar(range(len(coefficients)), coefficients)
            plt.xlabel('Feature Index')
            plt.ylabel('Coefficient')
            plt.title('Relevance of Features')
            plt.grid(True)
            plt.show()
        else:
            print("The best model does not have a coef_ attribute.")

    # Extract the accuracy scores for each lambda
    accuracy_scores = grid_search.cv_results_['mean_test_score']
    y_pred = best_model.predict(X_test)

    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)
    # Plot the accuracy of the model for each lambda
    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, accuracy_scores, marker='o')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Lambda')
    plt.grid(True)
    plt.show()

    # Get the best model and its metrics
    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    best_lambda = 1.0 / best_model.C
    best_recall = recall_score(y_test, y_pred)
    best_f1 = f1_score(y_test, y_pred)

    # Use PCA to reduce dimensions
    n_components = 2  # Number of components you want to keep
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Fit the best model on the PCA-transformed training data
    best_model.fit(X_train_pca, y_train)

    # Create a grid of points for visualization using PCA data
    x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
    y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Use the best model to predict on the mesh grid
    Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot decision regions with smaller markers
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', s=25, edgecolor='k', marker='o')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Decision Regions and Data Points')
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_pca)[:, 1])
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_pca)[:, 1])

    # Add the accuracy and other information within the same plot for the lambda with the best accuracy
    info_text = f'Accuracy: {best_accuracy:.4f}\nLambda: {best_lambda:.4f}\nRecall: {best_recall:.4f}\nF1 Score: {best_f1:.4f}\nROC AUC: {roc_auc:.4f}'
    # Move the info box to the bottom right corner of the mesh plot
    accuracy_box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
            horizontalalignment='right', bbox=accuracy_box_props)

    # Adjust plot limits to make sure the data points are more centered
    ax.set_xlim([xx.min(), xx.max()-5])
    ax.set_ylim([yy.min(), yy.max()-15])
    plt.show()

    # Plot the ROC curve
    fig_roc, ax_roc = plt.subplots(1, 1, figsize=(8, 6))
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

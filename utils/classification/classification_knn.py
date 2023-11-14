from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def classification_knn(X_train, X_test, y_train, y_test):
    # Define a range of K values to try
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    accuracy_scores, f1_scores, recall_scores, roc_auc_scores = evaluate_knn(X_train, X_test, y_train, y_test, k_values)
    plot_k_metrics(k_values, accuracy_scores, f1_scores, recall_scores, roc_auc_scores)
    k = 11
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict probabilities for the positive class
    y_prob = knn.predict_proba(X_test)[:, 1]

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Compute AUC (Area Under the Curve) for ROC
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    plt.show()

    # Perform PCA to reduce dimensions
    n_components = 2  # Number of components you want to keep
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train a KNN classifier on the reduced feature set
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca, y_train)

    # Create a grid of points for visualization
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predict class labels for each point in the mesh grid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    y_pred = knn.predict(X_test_pca)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Create a scatter plot to visualize the decision boundaries
    fig, ax = plt.subplots(figsize=(6, 6))

    # Add the contourf plot to the subplot
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Add the scatter plot to the same subplot with larger data points
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=25, edgecolor='k', marker='o')

    # Set the title
    ax.set_title('KNN Decision Boundaries (with PCA)')

    # Create the text
    info_text = f'F1 Score: {f1:.4f}\nRecall: {recall:.4f}\nAccuracy: {accuracy:.4f}\nK: {k}'

    # Add the text to the upper right corner of the subplot
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Adjust plot limits to make sure the data points are more centered
    ax.set_xlim([xx.min(), xx.max()-5])
    ax.set_ylim([yy.min(), yy.max()-15])

    # Display the plot
    plt.show()


def evaluate_knn(X_train, X_test, y_train, y_test, k_values):
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    roc_auc_scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        recall_scores.append(recall)
        roc_auc_scores.append(roc_auc)

    return accuracy_scores, f1_scores, recall_scores, roc_auc_scores

def plot_k_metrics(k_values, accuracy_scores, f1_scores, recall_scores, roc_auc_scores):
    plt.figure(figsize=(12, 6))

    # Accuracy vs. K
    plt.subplot(221)
    plt.plot(k_values, accuracy_scores, marker='o')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. K Value')

    # F1 Score vs. K
    plt.subplot(222)
    plt.plot(k_values, f1_scores, marker='o')
    plt.xlabel('K Value')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. K Value')

    # Recall vs. K
    plt.subplot(223)
    plt.plot(k_values, recall_scores, marker='o')
    plt.xlabel('K Value')
    plt.ylabel('Recall')
    plt.title('Recall vs. K Value')

    # ROC AUC vs. K
    plt.subplot(224)
    plt.plot(k_values, roc_auc_scores, marker='o')
    plt.xlabel('K Value')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC vs. K Value')
    plt.tight_layout()
    plt.show()





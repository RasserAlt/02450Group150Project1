from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_selection():
    # Load and preprocess your data
    data = pd.read_excel('data/abalone.xls')
    data['BinaryClass'] = (data['Rings'] < 7.5).astype(int)
    X = data.drop(['BinaryClass', 'Rings'], axis=1)
    y = data['BinaryClass']

    sex_feature = data['Sex']

    X['Is_M'] = (X['Sex'] == 'M').astype(int)
    X['Is_F'] = (X['Sex'] == 'F').astype(int)
    X['Is_I'] = (X['Sex'] == 'I').astype(int)
    X = X.drop('Sex', axis=1)
    pd.set_option('display.max_columns', None)
    X_original = X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split your dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Logistic Regression model (you can replace it with your model of choice)
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Create an RFE instance and specify the number of features to select
    num_features_to_select = 10  # Adjust based on your requirements
    rfe = RFE(model, n_features_to_select=num_features_to_select)

    # Fit the RFE model to your training data
    rfe.fit(X_train, y_train)

    # Retrieve the ranking of features based on their importance
    feature_ranking = rfe.ranking_

    # Get the selected features
    selected_features = rfe.support_
    all_feature_names = X_original.columns

    # Filter your training and testing data to include only the selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Train and evaluate your model using the reduced feature set
    model.fit(X_train_selected, y_train)
    coefficients = model.coef_[0]

    # Plot the relevance of the features
    plt.figure(figsize=(8, 6))
    plt.bar(all_feature_names, coefficients)
    plt.xlabel('Feature Name')
    plt.ylabel('Coefficient')
    plt.title('Relevance of Features')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    accuracy = model.score(X_test_selected, y_test)

    # Print the selected features and their ranking
    print("Selected Features:", selected_features)
    print("Feature Ranking:", feature_ranking)
    print("Accuracy with Selected Features:", accuracy)
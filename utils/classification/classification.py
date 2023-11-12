import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from feature_selection import feature_selection
from nested_crossvalidation import nested_crossvalidation
from utils.classification.baseline import classification_regression, classification_baseline
from utils.classification.classification_knn import classification_knn
matplotlib.use('TkAgg')

# Load and preprocess your data
data = pd.read_excel('../data/abalone.xls')
data['BinaryClass'] = (data['Rings'] < 7.5).astype(int)
X = data[['Height', 'Diameter', 'shucked_weight', 'Shell_weight']]
y = data['BinaryClass']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize the random oversampler
ros = RandomOverSampler(random_state=42)

# Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Resample the training data
X_train, y_train = ros.fit_resample(X_train, y_train)

train_binary_class_counts = y_train.value_counts()
plt.bar(train_binary_class_counts.index, train_binary_class_counts.values)
plt.xlabel('BinaryClass')
plt.ylabel('Count')
plt.title('Class Distribution in Original Training Set')
plt.xticks(train_binary_class_counts.index, ['Young Abalones', 'Mature Abalones'])  # Optional, if you want to label the x-axis
plt.show()

classification_baseline(X_train, X_test, y_train, y_test)
classification_knn(X_train, X_test, y_train, y_test)
classification_regression(X_train, X_test, y_train, y_test)

nested_crossvalidation(X, y)
feature_selection()

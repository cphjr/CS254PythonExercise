import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the digit dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate='adaptive', max_iter=300, batch_size=32, momentum=0.9, random_state=42)
mlp.fit(X_train, y_train)

# Predict and compute accuracy for MLP
y_pred_mlp = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Classifier Accuracy: {mlp_accuracy * 100:.2f}%")

# Initialize and train the K-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and compute accuracy for K-NN
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"K-NN Classifier Accuracy: {knn_accuracy * 100:.2f}%")

# Compare the results
print(f"MLP Classifier is {'better' if mlp_accuracy > knn_accuracy else 'worse'} than K-NN Classifier.")
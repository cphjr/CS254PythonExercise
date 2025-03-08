from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Display the first image from the dataset
plt.gray()
plt.matshow(digits.images[0])
plt.show()

# Split the dataset into training (80%) and test (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the K-NN algorithm
knn = KNeighborsClassifier(n_neighbors=5)  # You can change the number of neighbors to improve accuracy
knn.fit(X_train, y_train)

# Predict and calculate accuracy for K-NN
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K-NN Accuracy: {accuracy_knn * 100:.2f}%")

# Try to improve K-NN accuracy by changing parameters
best_accuracy = accuracy_knn
best_k = 5
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    if accuracy_knn > best_accuracy:
        best_accuracy = accuracy_knn
        best_k = k

print(f"Best K-NN Accuracy: {best_accuracy * 100:.2f}% with k={best_k}")
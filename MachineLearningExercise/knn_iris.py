from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training (80%) and test (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the K-NN algorithm
knn = KNeighborsClassifier(n_neighbors=5)  # You can change the number of neighbors to improve accuracy
knn.fit(X_train, y_train)

# Predict and calculate accuracy for K-NN
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"K-NN Accuracy: {accuracy_knn}")

# Train the decision tree algorithm
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and calculate accuracy for decision tree
y_pred_tree = clf.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree}")

# Compare the results
if accuracy_knn > accuracy_tree:
    print("K-NN has better accuracy.")
elif accuracy_knn < accuracy_tree:
    print("Decision Tree has better accuracy.")
else:
    print("Both K-NN and Decision Tree have the same accuracy.")

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

print(f"Best K-NN Accuracy: {best_accuracy} with k={best_k}")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training (80%) and test (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree algorithm
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Display the decision tree
tree_rules = export_text(clf, feature_names=iris['feature_names'])
print(tree_rules)

# Plot the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=iris['feature_names'], class_names=iris['target_names'], filled=True)
plt.show()

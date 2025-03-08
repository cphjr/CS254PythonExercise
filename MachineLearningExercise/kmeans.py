import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Compute RMSE for each cluster
def compute_rmse(X, labels, centroids):
    rmse = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        rmse += np.sqrt(mean_squared_error(cluster_points, np.tile(centroids[i], (len(cluster_points), 1))))
    return rmse / len(centroids)

rmse = compute_rmse(X_scaled, labels, centroids)
print(f"RMSE for each cluster: {rmse}")

# Compare clustering results with KNN classification
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)
knn_labels = knn.predict(X_scaled)

# Evaluate cluster quality
def cluster_quality(labels, true_labels):
    contingency_matrix = pd.crosstab(labels, true_labels)
    return contingency_matrix

contingency_matrix = cluster_quality(labels, y)
print("Contingency Matrix:\n", contingency_matrix)
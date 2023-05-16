import torch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate a synthetic dataset
X, y = make_blobs(n_samples=200, centers=3, random_state=42)

# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a K-means clustering model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
kmeans.fit(X_scaled)

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

# Display the intermediate and final results
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-means Clustering')
plt.show()

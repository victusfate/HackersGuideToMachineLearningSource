import torch
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Generate a moon-shaped dataset
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a DBSCAN clustering model
dbscan = DBSCAN(eps=0.3, min_samples=5)

# Fit the model to the data
dbscan.fit(X_scaled)

# Get the cluster labels for each data point
cluster_labels = dbscan.labels_

# Display the intermediate and final results
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()

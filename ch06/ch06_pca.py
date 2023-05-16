import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a PCA object
pca = PCA(n_components=2)

# Fit the PCA model to the data
pca.fit(X_scaled)

# Transform the data to the reduced dimensionality
X_pca = pca.transform(X_scaled)

# Display the intermediate and final results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Dimensionality Reduction')
plt.show()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define custom Dataset for PyTorch
class IrisDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create PyTorch DataLoader for training set
train_dataset = IrisDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Define KNN model using PyTorch
class KNN(nn.Module):
    def __init__(self, k):
        super(KNN, self).__init__()
        self.k = k

    def forward(self, x, train_data, train_targets):
        distances = ((x - train_data) ** 2).sum(dim=1).sqrt()
        _, indices = distances.topk(self.k, largest=False)
        knn_targets = train_targets[indices]
        predicted_class = knn_targets.mode().values.item()
        return predicted_class

# Define K value for KNN
k = 3

# Create KNN model
knn_model = KNN(k)

# Evaluate on test set
correct = 0
total = 0
for features, targets in zip(X_test, y_test):
    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)

    predicted = knn_model(features, train_dataset.features, train_dataset.targets)
    total += 1
    if predicted == targets:
        correct += 1

accuracy = correct / total
print(f"Accuracy on test set: {accuracy}")

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

# Define SVM model using PyTorch
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

# Create SVM model
svm_model = SVM()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(svm_model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    total_loss = 0
    for features, targets in train_loader:
        optimizer.zero_grad()
        outputs = svm_model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Print intermediate results
    average_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch+1}, Average Loss: {average_loss}")

# Evaluate on test set
with torch.no_grad():
    test_features = torch.tensor(X_test, dtype=torch.float32)
    test_targets = torch.tensor(y_test, dtype=torch.long)
    test_outputs = svm_model(test_features)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == test_targets).sum().item() / len(test_targets)
    print(f"Accuracy on test set: {accuracy}")

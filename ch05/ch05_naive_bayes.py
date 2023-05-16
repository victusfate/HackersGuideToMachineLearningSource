import torch
import torch.nn as nn

# Generate some random data
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Define naive Bayes model
class NaiveBayes(nn.Module):
    def __init__(self):
        super(NaiveBayes, self).__init__()
        self.priors = nn.Parameter(torch.randn(2))
        self.means = nn.Parameter(torch.randn(10, 2))
        self.vars = nn.Parameter(torch.randn(10, 2))

    def forward(self, x):
        probs = torch.exp(-((x.unsqueeze(1) - self.means) ** 2) / (2 * self.vars + 1e-8))
        probs = probs.prod(dim=2) * self.priors
        return probs / probs.sum(dim=1, keepdim=True)

model = NaiveBayes()

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    # Print epoch and loss
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Display predicted class probabilities for the first sample
sample = X[0]
predicted_probs = model(sample.unsqueeze(0))
print("Predicted Class Probabilities:", predicted_probs)

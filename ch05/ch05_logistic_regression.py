import torch
import torch.nn as nn

# Generate some random data
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Define logistic regression model
model = nn.Linear(10, 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), y.float())
    loss.backward()
    optimizer.step()

# Display the model's parameters
print("Model's Parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Display predicted probabilities for the first five samples
print("Predicted Probabilities for the First Five Samples:")
with torch.no_grad():
    predicted_probs = torch.sigmoid(model(X[:5]))
    print(predicted_probs)

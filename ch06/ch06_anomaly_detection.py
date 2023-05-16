from sklearn.ensemble import IsolationForest

# Create a sample dataset
X = [[1.0], [1.5], [2.0], [3.0], [3.5], [4.0], [8.0], [9.0], [10.0], [15.0]]

# Create an Isolation Forest model
model = IsolationForest(contamination=0.1)

# Fit the model to the data
model.fit(X)

# Predict the anomaly scores
anomaly_scores = model.decision_function(X)

# Print the anomaly scores
for score in anomaly_scores:
    print(f"Anomaly Score: {score}")

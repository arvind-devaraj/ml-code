import numpy as np

# Sample data (features)
# Assume each row represents a sample, and each column represents a feature
# In this example, we have 5 samples and 3 features
# Features might represent things like word frequencies, etc.
X = np.array([
    [0.2, 0.4, 0.5],
    [0.3, 0.5, 0.7],
    [0.1, 0.8, 0.6],
    [0.5, 0.2, 0.9],
    [0.6, 0.1, 0.4]
])

# Corresponding labels (0 for non-spam, 1 for spam)
# In this example, we have 5 samples
y = np.array([0, 1, 0, 1, 0])

# Add bias term to the features
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression hypothesis function
def predict_probability(theta, X):
    return sigmoid(np.dot(X, theta))

# Logistic regression cost function
def cost_function(theta, X, y):
    m = len(y)
    h = predict_probability(theta, X)
    return (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Logistic regression gradient descent
def gradient_descent(theta, X, y, alpha, epochs):
    m = len(y)
    costs = []

    for epoch in range(epochs):
        h = predict_probability(theta, X)
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
        cost = cost_function(theta, X, y)
        costs.append(cost)

    return theta, costs

# Initialize theta (weights)
theta = np.zeros(X.shape[1])

# Set hyperparameters
alpha = 0.1  # Adjusted learning rate
epochs = 3000

# Train the model
theta, costs = gradient_descent(theta, X, y, alpha, epochs)

# Predict function
def predict(theta, X):
    return np.round(predict_probability(theta, X))

# Predictions
predictions = predict(theta, X)
print("Predictions:", predictions)

# Evaluate accuracy
accuracy = np.mean(predictions == y) * 100
print("Accuracy:", accuracy)

import torch
import torch.nn as nn
import torch.optim as optim

# Define the inline Iris dataset
X = torch.tensor([[5.1, 3.5, 1.4, 0.2],
                  [4.9, 3.0, 1.4, 0.2],
                  [4.7, 3.2, 1.3, 0.2],
                  [4.6, 3.1, 1.5, 0.2],
                  [5.0, 3.6, 1.4, 0.2],
                  [7.0, 3.2, 4.7, 1.4],
                  [6.4, 3.2, 4.5, 1.5],
                  [6.9, 3.1, 4.9, 1.5],
                  [5.5, 2.3, 4.0, 1.3],
                  [6.5, 2.8, 4.6, 1.5],
                  [6.3, 3.3, 6.0, 2.5],
                  [5.8, 2.7, 5.1, 1.9],
                  [7.1, 3.0, 5.9, 2.1],
                  [6.3, 2.9, 5.6, 1.8],
                  [6.5, 3.0, 5.8, 2.2]])
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # Corresponding labels: 0 for Setosa, 1 for Versicolor, 2 for Virginica

# Define a simple neural network model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate the model
input_size = 4  # Number of features
hidden_size = 5  # Number of neurons in the hidden layer
num_classes = 3  # Number of classes (Iris species)
model = SimpleClassifier(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
num_epochs = 2500
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X.float())  # Convert tensor to float
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


#Inference
# Evaluate the model
def predict(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

# Sample input for prediction
sample_input = torch.tensor([[5.1, 3.5, 1.4, 0.2],   # Setosa
                             [6.9, 3.1, 4.9, 1.5],   # Versicolor
                             [6.3, 3.3, 6.0, 2.5]])  # Virginica

# Make predictions
predicted_labels = predict(model, sample_input.float())  # Convert tensor to float

# Map predicted labels to class names
class_names = ['Setosa', 'Versicolor', 'Virginica']
predicted_class_names = [class_names[label] for label in predicted_labels]

# Display predictions
for i in range(len(sample_input)):
    print(f"Predicted class for sample {i+1}: {predicted_class_names[i]}")

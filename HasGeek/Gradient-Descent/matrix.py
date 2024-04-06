import torch
import torch.nn as nn
import torch.optim as optim

# Define input and output data
inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
outputs = torch.tensor([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Initialize the model
input_dim = 2
output_dim = 2
model = LinearRegression(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs_pred = model(inputs)
    loss = criterion(outputs_pred, outputs)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print the learned matrix
print('Learned Matrix:')
print(model.linear.weight)

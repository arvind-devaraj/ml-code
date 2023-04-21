import torch
import torch.nn as nn
from torch import optim

from torch.autograd import Variable
X = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))

y = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))



# create the model
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.SGD(model.parameters(), lr=0.1)

# train the model
n_epochs = 400   # number of epochs to run
for epoch in range(n_epochs):
    # forward pass
    y_pred = model(X)
    # compute loss
    loss = loss_fn(y_pred, y)
    # backward pass
    loss.backward()
    # update parameters
    optimizer.step()
    # zero gradients
    optimizer.zero_grad()
    # print loss
    print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

x_test = Variable(torch.Tensor([[3.0], [4.0], [5.0]]))
print(model(x_test))

import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch
import torch.nn as nn
import torch.optim as optim


class NODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NODE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t, x):
        y = self.net(x)
        return y


input_dim = 2
hidden_dim = 150
output_dim = input_dim
lr = 0.002
epochs = 1000


def train(epoch):
    x0 = torch.tensor([[2.0, 2.0],
                       [2.0, -2.0]])
    x = torch.tensor([[4.0, 4.0],
                      [4.0, -4.0]])
    t = torch.linspace(0, 5, 2)
    for idx in range(1, 3):
        optimizer.zero_grad()
        output = odeint(node, x0[idx-1], t, method='euler')
        loss = torch.mean(torch.abs(output[1] - x[idx-1]))
        loss.backward()
        optimizer.step()
        if idx % 1 == 0:
            with torch.no_grad():
                print('Epoch:{} x{} Loss {:.6f}'.format(epoch, idx, loss.item()))
                #plt.scatter(output.numpy()[0][0], output.numpy()[0][1], c='b')
                #plt.scatter(x.numpy()[0][0], x.numpy()[0][1], c='g')
                #plt.scatter(x.numpy()[1][0], x.numpy()[1][1], c='g')
                #plt.scatter(output.numpy()[1][0], output.numpy()[1][1], c='r', marker='*')
    plt.show()


if __name__ == '__main__':
    node = NODE(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(node.parameters(), lr)
    for epoch in range(1, epochs + 1):
        train(epoch)


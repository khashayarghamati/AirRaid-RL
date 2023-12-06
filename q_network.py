import torch

# Define the Q-network
class QNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 24)
        self.fc2 = torch.nn.Linear(24, 24)
        self.fc3 = torch.nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

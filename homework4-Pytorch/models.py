"""Model classes"""
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """ConvNet without dropout"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,
                               8,
                               kernel_size=3,
                               padding=1,
                               padding_mode="zeros",
                               bias=True)
        self.conv2 = nn.Conv2d(8,
                               16,
                               kernel_size=3,
                               padding=1,
                               padding_mode="zeros",
                               bias=True)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input_data):
        input_data = F.relu(self.conv1(input_data))
        input_data = F.max_pool2d(input_data, 2)
        input_data = F.relu(self.conv2(input_data))
        input_data = F.max_pool2d(input_data, 2)
        input_data = input_data.view(-1, 784)
        input_data = F.relu(self.fc1(input_data))
        input_data = self.fc2(input_data)
        return F.log_softmax(input_data, dim=-1)


class ConvNetDropout(nn.Module):
    """ConvNet with dropout"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,
                               8,
                               kernel_size=3,
                               padding=1,
                               padding_mode="zeros",
                               bias=True)
        self.conv2 = nn.Conv2d(8,
                               16,
                               kernel_size=3,
                               padding=1,
                               padding_mode="zeros",
                               bias=True)
        self.dropout = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input_data):
        input_data = F.relu(self.conv1(input_data))
        input_data = self.dropout(input_data)
        input_data = F.max_pool2d(input_data, 2)
        input_data = F.relu(self.conv2(input_data))
        input_data = self.dropout(input_data)
        input_data = F.max_pool2d(input_data, 2)
        input_data = input_data.view(-1, 784)
        input_data = F.relu(self.fc1(input_data))
        input_data = self.fc2(input_data)
        return F.log_softmax(input_data, dim=-1)


class MLP(nn.Module):
    """MLP"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input_data):
        input_data = input_data.view(-1, 28 * 28)
        input_data = F.relu(self.fc1(input_data))
        input_data = self.fc2(input_data)
        return input_data

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, output_dim=10, input_dim=3, layer= -1):
        super(Net, self).__init__()
        self.layer = layer

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=1)
        self.bn1   = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=1)
        self.bn2   = nn.BatchNorm2d(192)

        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, stride=1)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, stride=1)
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        if self.layer == -8:
            z = x.clone()
        x = F.relu(x)

        x = self.conv3(x)
        if self.layer == -7:
            z = x.clone()
        x = F.relu(self.bn1(x))
        x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        if self.layer == -6:
            z = x.clone()
        x = F.relu(x)

        x = self.conv5(x)
        if self.layer == -5:
            z = x.clone()
        x = F.relu(x)

        x = self.conv6(x)
        if self.layer == -4:
            z = x.clone()
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, 2)

        x = self.conv7(x)
        if self.layer == -3:
            z = x.clone()
        x = F.relu(x)

        x = self.conv8(x)
        if self.layer == -2:
            z = x.clone()
        x = F.relu(x)

        x = self.conv9(x)
        if self.layer == -1:
            z = x.clone()
        x = F.relu(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x = torch.squeeze(x)

        # return x
        # return F.softmax(x, dim=1), z.view(z.shape[0], -1)

        return F.softmax(x, dim=1), z.view(z.shape[0], -1)

class LeNet(nn.Module):
    def __init__(self, output_dim, input_dim=1, layer=-1):
        super().__init__()

        self.layer = layer

        self.conv1 = nn.Conv2d(in_channels=input_dim,
                               out_channels=6,
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        # z = x.view(x.shape[0], -1)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        if self.layer == -5:
            z = (x.view(x.shape[0], -1)).clone()

        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.shape[0], -1)
        if self.layer == -4:
            z = x.clone()

        x = self.fc1(x)
        if self.layer == -3:
            z = x.clone()

        x = F.relu(x)

        x = self.fc2(x)
        if self.layer == -2:
            z = x.clone()

        x = F.relu(x)

        x = self.fc3(x)
        if self.layer == -1:
            z = x.clone()

        x = F.softmax(x)
        return x, z

class Net_fb(nn.Module):
    def __init__(self, output_dim=10, input_dim=3):
        super(Net_fb, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=1)
        self.bn1   = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=1)
        self.bn2   = nn.BatchNorm2d(192)

        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, stride=1)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, stride=1)
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.bn2(self.conv6(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x = torch.squeeze(x)

        # return x
        # return F.softmax(x, dim=1), z.view(z.shape[0], -1)

        return F.softmax(x, dim=1)

class LeNet_fb(nn.Module):
    def __init__(self, output_dim, input_dim=1):
        super(LeNet_fb, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_dim,
                               out_channels=6,
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        x = F.relu(x)

        x = self.fc3(x)

        x = F.softmax(x)
        return x
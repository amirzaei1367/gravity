import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

        l_1 = x.clone()
        x = F.relu(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x = torch.squeeze(x)

        # return x
        # return F.softmax(x, dim=1), z.view(z.shape[0], -1)

        return F.softmax(x, dim=1), l_1.view(l_1.shape[0], -1), z.view(z.shape[0], -1)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):  # (conv-bn-relu) x 3 times
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # in our case is none
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, depth, num_classes=10, layer=-1):
        super(ResNet, self).__init__()
        self.layer = layer
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)

        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)

        self.maxpool2 = nn.MaxPool2d(16)
        self.fc = nn.Linear(64 * block.expansion, 1024)
        self.fcf = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        if self.layer == -6:
            z = (x.view(x.shape[0], -1)).clone()

        x = self.layer2(x)
        if self.layer == -5:
            z = (x.view(x.shape[0], -1)).clone()

        x = self.layer3(x)
        if self.layer == -4:
            z = (x.view(x.shape[0], -1)).clone()

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 256 dimensional
        if self.layer == -3:
            z = x.clone()

        x = self.fc(x)  # 1024 dimensional
        if self.layer == -2:
            z = x.clone()

        x = self.fcf(x)  # num_classes dimensional
        if self.layer == -1:
            z = x.clone()

        l_1 = x.clone()
        x = F.softmax(x)
        return x, l_1, z

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

class ResNet_fb(nn.Module):

    def __init__(self, depth, num_classes=10, layer=-1):
        super(ResNet, self).__init__()
        self.layer = layer
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)

        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)

        self.maxpool2 = nn.MaxPool2d(16)
        self.fc = nn.Linear(64 * block.expansion, 1024)
        self.fcf = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)


        x = self.layer2(x)


        x = self.layer3(x)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 256 dimensional


        x = self.fc(x)  # 1024 dimensional


        x = self.fcf(x)  # num_classes dimensional


        l_1 = x.clone()
        x = F.softmax(x)
        return x

def resnet_fb(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet_fb(**kwargs)

# class LeNet(nn.Module):
#     def __init__(self, output_dim, input_dim=1, layer=-1):
#         super().__init__()
#
#         self.layer = layer
#
#         self.conv1 = nn.Conv2d(in_channels=input_dim,
#                                out_channels=6,
#                                kernel_size=5)
#
#         self.conv2 = nn.Conv2d(in_channels=6,
#                                out_channels=16,
#                                kernel_size=5)
#
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#
#         self.fc2 = nn.Linear(120, 84)
#
#         self.fc3 = nn.Linear(84, output_dim)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         # z = x.view(x.shape[0], -1)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2)
#
#         x = self.conv2(x)
#         if self.layer == -5:
#             z = (x.view(x.shape[0], -1)).clone()
#
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2)
#
#         x = x.view(x.shape[0], -1)
#         if self.layer == -4:
#             z = x.clone()
#
#         x = self.fc1(x)
#         if self.layer == -3:
#             z = x.clone()
#
#         x = F.relu(x)
#
#         x = self.fc2(x)
#         if self.layer == -2:
#             z = x.clone()
#
#         x = F.relu(x)
#
#         x = self.fc3(x)
#         if self.layer == -1:
#             z = x.clone()
#
#         l_1 = x.clone()
#
#         x = F.softmax(x)
#         return x, l_1, z
#
# class LeNet_fb(nn.Module):
#     def __init__(self, output_dim, input_dim=1):
#         super(LeNet_fb, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=input_dim,
#                                out_channels=6,
#                                kernel_size=5)
#
#         self.conv2 = nn.Conv2d(in_channels=6,
#                                out_channels=16,
#                                kernel_size=5)
#
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#
#         self.fc2 = nn.Linear(120, 84)
#
#         self.fc3 = nn.Linear(84, output_dim)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2)
#
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=2)
#
#         x = x.view(x.shape[0], -1)
#
#         x = self.fc1(x)
#         x = F.relu(x)
#
#         x = self.fc2(x)
#
#         x = F.relu(x)
#
#         x = self.fc3(x)
#
#         x = F.softmax(x)
#         return x

class LeNet(nn.Module):
    def __init__(self, output_dim, input_dim=1, layer=-1):
        super().__init__()

        self.layer = layer

        self.conv1 = nn.Conv2d(in_channels=input_dim,
                               out_channels=32,
                               kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 5 * 5, 512)

        self.fc2 = nn.Linear(512, 64)

        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        # z = x.view(x.shape[0], -1)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = self.bn2(x)
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

        l_1 = x.clone()

        x = F.softmax(x)
        return x, l_1, z

class LeNet_fb(nn.Module):
    def __init__(self, output_dim, input_dim=1):
        super(LeNet_fb, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_dim,
                               out_channels=32,
                               kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 5 * 5, 512)

        self.fc2 = nn.Linear(512, 64)

        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = self.bn2(x)
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
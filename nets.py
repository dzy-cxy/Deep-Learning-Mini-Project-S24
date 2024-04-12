import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetCifarBlock(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        stride = 1
        self.expand = False
        if input_nc != output_nc:
            assert input_nc * 2 == output_nc, 'output_nc must be input_nc * 2'
            stride = 2
            self.expand = True

        self.conv1 = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(output_nc)
        self.conv2 = nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_nc)

    def forward(self, x):
        xx = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(xx))
        if self.expand:
            x = F.interpolate(x, scale_factor=0.5, mode='nearest')  # subsampling
            zero = torch.zeros_like(x)
            x = torch.cat([x, zero], dim=1)  # option A in the original paper
        h = F.relu(y + x, inplace=True)
        return h


def make_resblock_group(cls, input_nc, output_nc, n):
    blocks = []
    blocks.append(cls(input_nc, output_nc))
    for _ in range(1, n):
        blocks.append(cls(output_nc, output_nc))
    return nn.Sequential(*blocks)


class ResNetCifar(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.block1 = make_resblock_group(ResNetCifarBlock, 16, 16, n)
        self.block2 = make_resblock_group(ResNetCifarBlock, 16, 32, n)
        self.block3 = make_resblock_group(ResNetCifarBlock, 32, 64, n)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # global average pooling
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

import torch
from torch import nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvBN2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBN2d, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, **kwargs),
            torch.nn.BatchNorm2d(out_channels)
        )

    def forward(self, input):
        return F.relu(self.model(input), inplace=True)


class PConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PConv2d, self).__init__()
        self.conv_1 = ConvBN2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv_2 = ConvBN2d(
            in_channels, out_channels, kernel_size=5, padding=2, stride=1)

    def forward(self, input):
        return F.relu(torch.add(self.conv_1(input), self.conv_2(input)), inplace=True)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_list = torch.nn.ModuleList().cuda()
        for i in range(10):
            self.conv_list.append(torch.nn.Sequential(
                ConvBN2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1),
                ConvBN2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                ConvBN2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1),
            ))

        self.conv_list.append(ConvBN2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1))

        self.conv = ConvBN2d(11*out_channels, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, input):
        cat = torch.cat([conv(input) for conv in self.conv_list], dim=1)
        return self.conv(cat)

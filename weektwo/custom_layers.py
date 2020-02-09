import torch
from torch import nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Module to flatten channelled image into single vector.

    Used when moving from convolutional layers to fully connected layers.
    """

    def forward(self, input) -> torch.FloatTensor:
        return input.view(input.size(0), -1)


class ConvBN2d(nn.Module):
    """Module that combines a convolution and batch normalisation. Output is
    also passed through the ReLU function.

    Dropout layer is also included.
    """

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        """ Class constructor.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        """
        super(ConvBN2d, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, **kwargs),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Dropout(p=0.1)
        )

    def forward(self, input) -> torch.FloatTensor:
        return F.relu(self.model(input), inplace=True)


class PConv2d(nn.Module):
    """Module which combines two parallel ConvBN2d blocks, each with a
    different kernel size.
    """

    def __init__(self, in_channels, out_channels) -> None:
        """ Class constructor.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        """
        super(PConv2d, self).__init__()
        self.conv_1 = ConvBN2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv_2 = ConvBN2d(
            in_channels, out_channels, kernel_size=5, padding=2, stride=1)

    def forward(self, input) -> torch.FloatTensor:
        return F.relu(
            torch.add(self.conv_1(input), self.conv_2(input)), inplace=True)


class ConvBlock(nn.Module):
    """ Module that imitates the Conv block used in ResNeXt-50."""

    def __init__(self, in_channels, out_channels, num_blocks=10) -> None:
        """ Class constructor.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        num_blocks : int
            Number of parallel blocks to include.
        """
        super(ConvBlock, self).__init__()
        self.conv_list = torch.nn.ModuleList().cuda()
        for i in range(num_blocks):
            self.conv_list.append(torch.nn.Sequential(
                ConvBN2d(
                    in_channels, in_channels,
                    kernel_size=1, padding=0, stride=1),
                ConvBN2d(
                    in_channels, out_channels,
                    kernel_size=3, padding=1, stride=1),
                ConvBN2d(
                    out_channels, out_channels,
                    kernel_size=1, padding=0, stride=1),
            ))

        self.conv_list.append(ConvBN2d(
            in_channels, out_channels, kernel_size=1, padding=0, stride=1))

        self.conv = ConvBN2d(
            (num_blocks + 1)*out_channels, out_channels,
            kernel_size=1, padding=0, stride=1)

    def forward(self, input) -> torch.FloatTensor:
        cat = torch.cat([conv(input) for conv in self.conv_list], dim=1)
        return self.conv(cat)

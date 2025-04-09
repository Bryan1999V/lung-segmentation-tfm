"""U-NET model used for input data with 2-dimensions as images."""

import torch
from torch import nn

from unet.utils import DoubleConv2d, DownConv2d, UpConv2d

DEFAULT_OUTPUT_CHANNELS_FIRST_BLOCK = 64
DEFAULT_OUTPUT_CHANNELS_SECOND_BLOCK = 2 * DEFAULT_OUTPUT_CHANNELS_FIRST_BLOCK
DEFAULT_OUTPUT_CHANNELS_THIRD_BLOCK = 2 * DEFAULT_OUTPUT_CHANNELS_SECOND_BLOCK
DEFAULT_OUTPUT_CHANNELS_FOURTH_BLOCK = 2 * DEFAULT_OUTPUT_CHANNELS_THIRD_BLOCK
DEFAULT_OUTPUT_CHANNELS_FIFTH_BLOCK = 2 * DEFAULT_OUTPUT_CHANNELS_FOURTH_BLOCK
DEFAULT_LAST_CONV_KERNEL_SIZE = 1
DEFAULT_LAST_CONV_STRIDE = 1


class UNet(nn.Module):
    """
    U-NET model for image segmentation.

    This model is composed by a series of down-sampling and up-sampling blocks.
    """

    def __init__(self, in_channels: int, n_classes: int) -> None:
        """Initialize the U-NET model with the given number of input channels and classes."""
        super().__init__()
        self._fisrt_double_conv = DoubleConv2d(in_channels, DEFAULT_OUTPUT_CHANNELS_FIRST_BLOCK)
        self._down_conv1 = DownConv2d(DEFAULT_OUTPUT_CHANNELS_FIRST_BLOCK, DEFAULT_OUTPUT_CHANNELS_SECOND_BLOCK)
        self._down_conv2 = DownConv2d(DEFAULT_OUTPUT_CHANNELS_SECOND_BLOCK, DEFAULT_OUTPUT_CHANNELS_THIRD_BLOCK)
        self._down_conv3 = DownConv2d(DEFAULT_OUTPUT_CHANNELS_THIRD_BLOCK, DEFAULT_OUTPUT_CHANNELS_FOURTH_BLOCK)
        self._down_conv4 = DownConv2d(DEFAULT_OUTPUT_CHANNELS_FOURTH_BLOCK, DEFAULT_OUTPUT_CHANNELS_FIFTH_BLOCK)
        self._up_conv1 = UpConv2d(DEFAULT_OUTPUT_CHANNELS_FIFTH_BLOCK, DEFAULT_OUTPUT_CHANNELS_FOURTH_BLOCK)
        self._up_conv2 = UpConv2d(DEFAULT_OUTPUT_CHANNELS_FOURTH_BLOCK, DEFAULT_OUTPUT_CHANNELS_THIRD_BLOCK)
        self._up_conv3 = UpConv2d(DEFAULT_OUTPUT_CHANNELS_THIRD_BLOCK, DEFAULT_OUTPUT_CHANNELS_SECOND_BLOCK)
        self._up_conv4 = UpConv2d(DEFAULT_OUTPUT_CHANNELS_SECOND_BLOCK, DEFAULT_OUTPUT_CHANNELS_FIRST_BLOCK)
        self._last_conv = nn.Conv2d(
            DEFAULT_OUTPUT_CHANNELS_FIRST_BLOCK,
            n_classes,
            kernel_size=DEFAULT_LAST_CONV_KERNEL_SIZE,
            stride=DEFAULT_LAST_CONV_STRIDE,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward process to calculate the U-NET output data based on the given input data.

        :param x: input data with 2 dimensions.
        :return: output data with 2 dimensions.
        """
        x1 = self._fisrt_double_conv(x)
        x2 = self._down_conv1(x1)
        x3 = self._down_conv2(x2)
        x4 = self._down_conv3(x3)
        x5 = self._down_conv4(x4)
        x = self._up_conv1(x5, x4)
        x = self._up_conv2(x, x3)
        x = self._up_conv3(x, x2)
        x = self._up_conv4(x, x1)
        return self._last_conv(x)

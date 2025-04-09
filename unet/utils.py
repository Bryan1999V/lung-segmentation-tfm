"""Different custom PyTorch modules used in the U-NET model."""

from __future__ import annotations

import torch
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, MaxPool2d, Module, ReLU, Sequential

DEFAULT_CONV_KERNEL_SIZE = 3
DEFAULT_CONV_PADDING = 1
DEFAULT_MAXPOOL_KERNEL_SIZE = 2
DEFAULT_MAXPOOL_STRIDE = 2
DEFAULT_UPSAMPLER_BILINEAR_INTERPOLATION = True
DEFAULT_UPSAMPLER_SCALE_FACTOR = 2
DEFAULT_UPSAMPLER_CONCATENATE_DIM = 1
DEFAULT_UP_SAMPLER_ALIGN_CORNERS = True
DEFAULT_CONV_TRANSPOSE_KERNEL_SIZE = 2
DEFAULT_CONV_TRANSPOSE_STRIDE = 2


class DoubleConv2d(Module):
    """
    A two-dimension custom convolutional block consisting of two consecutive layers `Conv2D -> BatchNorm2D -> ReLU`.

    This block is commonly used to extract and refine features in the U-Net model.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the convolutional block structure with the given number of channel for the input and output layers.

        The number of channels for the output of the first convolutional layer and the input of the second convolutional
        layer are the same. The kernel size for both convolutional layers is 3x3 and the padding is 1.

        :param channels_in: number of channels for the input convolutional block.
        :param channels_out: number of channels for the output convolutional block.
        """
        super().__init__()
        self._double_conv = Sequential(
            Conv2d(in_channels, out_channels, DEFAULT_CONV_KERNEL_SIZE, padding=DEFAULT_CONV_PADDING),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, DEFAULT_CONV_KERNEL_SIZE, padding=DEFAULT_CONV_PADDING),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation for the convolutional block.

        :param x: input data with 2 dimensions.
        :return: output data with 2 dimensions.
        """
        return self._double_conv(x)


class DownConv2d(Module):
    """
    A two-dimension custom down-sampling block composed by `MaxPool2D -> DoubleConv2D`.

    This block is commonly used to reduce the spatial dimensions of the input data while extracting features.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the down-sampling block structure with the given number of channel for the input and output layers.

        The max-pooling layer uses a kernel size of 2x2 and a stride of 2.

        :param in_channels: number of channels for the input convolutional block.
        :param out_channels: number of channels for the output convolutional block.
        """
        super().__init__()
        self._down_conv = Sequential(
            MaxPool2d(DEFAULT_MAXPOOL_KERNEL_SIZE, DEFAULT_MAXPOOL_STRIDE),
            DoubleConv2d(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward process to calculate the down-sampling output data based on the given input data.

        :param x: input data with 2 dimensions.
        :return: output data with 2 dimensions.
        """
        return self._down_conv(x)


class UpConv2d(Module):
    """
    A two-dimension custom up-sampling block composed by `UpSample -> DoubleConv2D`.

    This block is commonly used to increase the spatial dimensions of the input data while extracting features.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the up-sampling block structure with the given number of channel for the input and output layers.

        :param in_channels: number of channels for the input convolutional block.
        :param out_channels: number of channels for the output convolutional block.
        """
        super().__init__()
        self._up_sample = ConvTranspose2d(
            in_channels,
            in_channels // DEFAULT_UPSAMPLER_SCALE_FACTOR,
            kernel_size=DEFAULT_CONV_TRANSPOSE_KERNEL_SIZE,
            stride=DEFAULT_CONV_TRANSPOSE_STRIDE,
        )
        self._double_conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward process to calculate the up-sampling output data based on the given input data.

        :param x1: input data with 2 dimensions to be up-sampled.
        :param x2: input data with 2 dimensions to be concatenated with the up-sampled data.
        :return: output data with 2 dimensions.
        """
        x1 = self._up_sample(x1)
        return self._double_conv(torch.cat([x1, x2], dim=DEFAULT_UPSAMPLER_CONCATENATE_DIM))

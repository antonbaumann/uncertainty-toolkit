"""
Code taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""

import torch
import torch.nn as nn
from typing import List, Union

from .components import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """
    Implementation of the U-Net model.
    """

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            dropout_rate: Union[float, List[float]] = 0.0,
            bilinear: bool = False,
        ):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            dropout_rate (Union[float, List[float]]): The dropout rate. If a float is provided, the same dropout rate
                will be used for all layers. If a list of 8 floats is provided, the dropout rates will be used for each
                layer in the following order: `[down1, down2, down3, down4, up1, up2, up3, up4]`.
            bilinear (bool): Whether to use bilinear interpolation.
        """
        super(UNet, self).__init__()
        self.in_channels: int = in_channels
        """The number of input channels."""
        self.out_channels: int = out_channels
        """The number of output channels."""
        self.bilinear: bool = bilinear
        """Whether to use bilinear interpolation."""

        if isinstance(dropout_rate, float):
            dropout_rate = [dropout_rate] * 8
        
        if len(dropout_rate) != 8:
            raise ValueError('`dropout_rate` must be a float or a list of 8 floats.')

        self.inc = DoubleConv(
            in_channels=in_channels, 
            out_channels=64,
        )
        self.down1 = Down(
            in_channels=64, 
            out_channels=128,
            dropout_rate=dropout_rate[0],
        )
        self.down2 = Down(
            in_channels=128, 
            out_channels=256,
            dropout_rate=dropout_rate[1],
        )
        self.down3 = Down(
            in_channels=256, 
            out_channels=512,
            dropout_rate=dropout_rate[2],
        )
        factor = 2 if bilinear else 1
        self.down4 = Down(
            in_channels=512, 
            out_channels=1024 // factor,
            dropout_rate=dropout_rate[3],
        )
        self.up1 = Up(
            in_channels=1024, 
            out_channels=512 // factor, 
            dropout_rate=dropout_rate[4],
            bilinear=bilinear,
        )
        self.up2 = Up(
            in_channels=512, 
            out_channels=256 // factor, 
            dropout_rate=dropout_rate[5],
            bilinear=bilinear,
        )
        self.up3 = Up(
            in_channels=256, 
            out_channels=128 // factor, 
            dropout_rate=dropout_rate[6],
            bilinear=bilinear,
        )
        self.up4 = Up(
            in_channels=128, 
            out_channels=64, 
            dropout_rate=dropout_rate[7],
            bilinear=bilinear,
        )
        self.outc = OutConv(
            in_channels=64, 
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): The input tensor. Shape: `(B, C_in, H, W)`.

        Returns:
            `torch.Tensor`: The output tensor. Shape: `(B, C_out, H, W)`.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

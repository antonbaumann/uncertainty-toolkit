import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Double convolution module.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0, mid_channels=None):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            dropout_rate (float): The dropout rate.
            mid_channels (int): The number of channels in the middle layer. If `None`, `mid_channels` will be set to
                `out_channels`.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=mid_channels, 
                kernel_size=3, 
                padding=1, 
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                padding=1, 
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            dropout_rate (float): The dropout rate.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels=in_channels, 
                out_channels=out_channels,
                dropout_rate=dropout_rate,
            )
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the downscaling block.

        Args:
            x (torch.Tensor): The input tensor. Shape: `(B, C, H, W)`.
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block with double conv."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0, bilinear: bool = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, 
                mode='bilinear', 
                align_corners=True,
            )
            self.conv = DoubleConv(
                in_channels=in_channels, 
                out_channels=out_channels, 
                dropout_rate=dropout_rate, 
                mid_channels=in_channels // 2,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels, 
                out_channels=in_channels // 2, 
                kernel_size=2, 
                stride=2,
            )
            self.conv = DoubleConv(
                in_channels=in_channels, 
                out_channels=out_channels, 
                dropout_rate=dropout_rate,
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Forward pass of the upscaling block.

        Args:
            x1 (torch.Tensor): The input tensor.
            x2 (torch.Tensor): The tensor from the corresponding downscaling block.
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            input=x1, 
            pad=[diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution module."""
    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the output convolution module.
        """
        return self.conv(x)

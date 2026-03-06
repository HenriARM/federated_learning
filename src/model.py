"""
UNet for binary segmentation of histopathological images.

Architecture:
  - Encoder: 4 × (DoubleConv → MaxPool) downsampling blocks
  - Bottleneck: DoubleConv at the lowest resolution
  - Decoder: 4 × (ConvTranspose2d + skip-concat + DoubleConv) upsampling blocks
  - Output: 1×1 Conv → single-channel logit map (sigmoid applied at inference)

Default base_channels=32 gives a lightweight model (~1.9 M parameters) well
suited for a local machine.  Set base_channels=64 for the original UNet size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d followed by DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Transposed convolution upsampling + skip connection + DoubleConv (decoder step)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2's spatial dimensions (handles odd sizes)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """
    Standard UNet encoder-decoder for binary segmentation.

    Args:
        in_channels:    Number of input image channels (3 for RGB).
        base_channels:  Width multiplier.  Channels at each level are
                        [c, 2c, 4c, 8c, 16c] where c = base_channels.
        n_classes:      Number of output channels.  1 for binary segmentation
                        (BCEWithLogitsLoss compatible).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        n_classes: int = 1,
    ) -> None:
        super().__init__()
        c = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)  # bottleneck

        # Decoder
        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)

        # Output
        self.outc = nn.Conv2d(c, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)  # raw logits, shape (B, 1, H, W)

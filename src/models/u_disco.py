# Adapted from: https://github.com/milesial/Pytorch-UNet/tree/master/unet
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.types import ExpConfig, IBackendModel
from src.utils import adjust_dim


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU => Dropout) * 2"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: int = None,
            dropout_p: float = 0.1,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with max pooling, then double conv"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout_p: float = 0.1,
    ):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Up-scaling then double conv"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            linear: bool = True,
            dropout_p: float = 0.1,
    ):
        super().__init__()
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Input is CL
        diff_y = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UDisCo(IBackendModel):
    def __init__(
            self,
            in_channels: int,
            linear: bool = False,
            dropout_p: float = 0.1,
            use_control: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.linear = linear
        self.use_control = use_control

        self.inc = nn.Conv1d(in_channels, 64, kernel_size=25, padding='same')
        self.down1 = Down(64, 128, dropout_p=dropout_p)
        self.down2 = Down(128, 256, dropout_p=dropout_p)
        self.down3 = Down(256, 512, dropout_p=dropout_p)
        self.down4 = Down(512, 512, dropout_p=dropout_p)
        self.down5 = Down(512, 512, dropout_p=dropout_p)
        self.down6 = Down(512, 512, dropout_p=dropout_p)
        self.down7 = Down(512, 512, dropout_p=dropout_p)
        factor = 2 if linear else 1
        self.down8 = Down(512, 1024 // factor, dropout_p=dropout_p)

        self.up1 = Up(1024, 512, linear, dropout_p=dropout_p)
        self.up2 = Up(1024, 512, linear, dropout_p=dropout_p)
        self.up3 = Up(1024, 512, linear, dropout_p=dropout_p)
        self.up4 = Up(1024, 512, linear, dropout_p=dropout_p)
        self.up5 = Up(1024, 512 // factor, linear, dropout_p=dropout_p)
        self.up6 = Up(512, 256 // factor, linear, dropout_p=dropout_p)
        self.up7 = Up(256, 128 // factor, linear, dropout_p=dropout_p)
        self.up8 = Up(128, 64, linear, dropout_p=dropout_p)

        self.outc = nn.Conv1d(64, 1, kernel_size=25, padding='same')
        self.control_conv = nn.Conv1d(1 + 1, 1, kernel_size=25, padding='same')

    def forward(self, x: torch.Tensor, control: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_control and control is None:
            raise ValueError("Control data is required.")

        if not self.use_control and control is not None:
            raise ValueError("Control data is not required.")

        x = adjust_dim(x, self.in_channels)
        if self.use_control and control is not None:
            control = adjust_dim(control, 1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)

        x = self.up1(x9, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)

        logits = self.outc(x)
        if self.use_control:
            logits = self.control_conv(torch.cat([logits, control], dim=1))
        return logits.transpose(-1, -2).squeeze(-1)

    @classmethod
    def from_config(cls, config: ExpConfig) -> 'UDisCo':
        return cls(
            in_channels=config.model_dev.vocab_size + config.model_dev.n_epi,
            linear=config.model_dev.linear_upsample,
            dropout_p=config.model_dev.dropout_p,
            use_control=config.model_dev.use_control,
        )

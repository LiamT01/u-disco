import torch
import torch.nn.functional as F
from torch import nn

from src.types import ExpConfig, IBackendModel
from src.utils import adjust_dim


class BPNet(IBackendModel):
    def __init__(
            self,
            in_channels: int,
            use_control: bool = False,
    ):
        super(BPNet, self).__init__()
        self.in_channels = in_channels
        self.use_control = use_control

        # Body of BPNet
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=25, padding='same')
        self.conv_blocks = nn.ModuleList()
        for i in range(1, 10):
            dilation = 2 ** i
            conv = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=dilation)
            self.conv_blocks.append(conv)

        self.profile_layer = nn.ConvTranspose1d(64, 1, kernel_size=25, padding=12)
        self.control_conv = nn.Conv1d(2, 1, kernel_size=1) if use_control else None

    def forward(self, x, control=None):
        if self.use_control and control is None:
            raise ValueError("Control data is required.")

        if not self.use_control and control is not None:
            raise ValueError("Control data is not required.")

        x = adjust_dim(x, self.in_channels)
        if self.use_control and control is not None:
            control = adjust_dim(control, 1)

        # First conv layer (conv1)
        x = F.relu(self.conv1(x))

        # Residual dilated conv blocks
        for conv in self.conv_blocks:
            residual = x
            x = F.relu(conv(x))
            x = x + residual

        # Profile shape head (deconvolution)
        x = self.profile_layer(x)  # (batch_size, 1, 1000)

        if self.use_control:
            x = self.control_conv(torch.cat([x, control], dim=1))

        return x.transpose(-1, -2).squeeze(-1)

    @classmethod
    def from_config(
            cls,
            config: ExpConfig,
    ) -> 'BPNet':
        return cls(
            in_channels=config.model_dev.vocab_size + config.model_dev.n_epi,
            use_control=config.model_dev.use_control,
        )

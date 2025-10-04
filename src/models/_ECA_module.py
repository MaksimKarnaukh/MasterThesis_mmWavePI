import torch
import torch.nn as nn
import torch.nn.functional as F


class ECALayer1D(nn.Module):
    """
    1D Efficient Channel Attention (ECA) Layer adapted for 1D convolutional inputs.

    Source: https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    """
    def __init__(self, channel, k_size=3):
        super(ECALayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2))
        return x * y.expand_as(x)
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    """Thermal baseline CELL 5, parameter names and operations preserved."""

    def __init__(self, a, b, s=1):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(a, b, 3, s, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, b, 3, 1, 1, bias=False),
            nn.BatchNorm2d(b),
        )
        self.d = (
            nn.Sequential(nn.Conv2d(a, b, 1, s, bias=False), nn.BatchNorm2d(b))
            if a != b or s != 1
            else nn.Identity()
        )

    def forward(self, x):
        return torch.relu(self.c(x) + self.d(x))


class FrameNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            Block(32, 32),
            Block(32, 64, 2),
            Block(64, 128, 2),
            Block(128, 256, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, 40)

    def features(self, x):
        b, t, c, h, w = x.shape
        return self.f(x.reshape(b * t, c, h, w)).flatten(1).reshape(b, t, -1)

    def forward(self, x):
        return self.fc(self.features(x)).mean(1)


class ResBlock(nn.Module):
    """Thermal specialist CELL 5."""

    def __init__(self, a, b, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(a, b, 3, s, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, 1, 1, bias=False),
            nn.BatchNorm2d(b),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(a, b, 1, s, bias=False), nn.BatchNorm2d(b))
            if a != b or s != 1
            else nn.Identity()
        )

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x), inplace=True)


class ThermalGRUNet(nn.Module):
    def __init__(self, num_classes=40, feat_dim=256, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.frame_enc = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            ResBlock(32, 64),
            ResBlock(64, 64),
            ResBlock(64, 128, 2),
            ResBlock(128, 128),
            ResBlock(128, feat_dim, 2),
            ResBlock(feat_dim, feat_dim),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gru = nn.GRU(
            feat_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def features(self, x):
        b, t, c, h, w = x.shape
        return self.frame_enc(x.reshape(b * t, c, h, w)).flatten(1).reshape(b, t, -1)

    def classify(self, f):
        # Native fp32 GRU avoids cuDNN mixed-precision teardown failures on the Windows GPU.
        if f.is_cuda:
            with torch.backends.cudnn.flags(enabled=False), torch.autocast("cuda", enabled=False):
                g, _ = self.gru(f.float())
        else:
            g, _ = self.gru(f)
        w = F.softmax(self.attn(g), dim=1)
        return self.classifier((g * w).sum(dim=1))

    def forward(self, x):
        return self.classify(self.features(x))


class TemporalFrameNet(FrameNet):
    """Fine-tune the baseline with identity-initialized temporal residual attention."""

    def __init__(self):
        super().__init__()
        self.temporal = nn.Conv1d(256, 256, 3, padding=1, groups=256)
        self.attention = nn.Linear(256, 1)
        nn.init.zeros_(self.temporal.weight)
        nn.init.zeros_(self.temporal.bias)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def forward(self, x):
        features = self.features(x)
        features = features + self.temporal(features.transpose(1, 2)).transpose(1, 2)
        weights = self.attention(features).float().softmax(1)
        return self.fc((features * weights).sum(1))


class SEThermalGRUNet(ThermalGRUNet):
    """Fine-tune the specialist with an identity-initialized clip-wise channel gate."""

    def __init__(self):
        super().__init__()
        self.channel_gate = nn.Sequential(nn.Linear(256, 32), nn.SiLU(), nn.Linear(32, 256))
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)

    def forward(self, x):
        features = self.features(x)
        gate = 2 * self.channel_gate(features.mean(1)).sigmoid()
        return self.classify(features * gate[:, None])


def make_model(recipe):
    return {
        "r1_thermal_baseline": FrameNet,
        "r3_thermal_specialist": ThermalGRUNet,
        "i1_temporal_baseline": TemporalFrameNet,
        "i3_se_specialist": SEThermalGRUNet,
    }[recipe]()

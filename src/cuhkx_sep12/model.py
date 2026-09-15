from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import MODALITIES


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 8), 1),
            nn.SiLU(),
            nn.Conv2d(max(8, channels // 8), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class SpatialAttention(nn.Module):
    """The spatial half of CBAM; used only by the Sep13 SE variant."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * self.gate(pooled)


class SmallCNN(nn.Module):
    def __init__(self, dim, se=False, spatial_attention=False):
        super().__init__()
        layers = []
        previous = 3
        for channels in (24, 48, 96, 128):
            layers.extend(
                [
                    nn.Conv2d(previous, channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                ]
            )
            if se:
                layers.append(SE(channels))
            if spatial_attention:
                layers.append(SpatialAttention())
            previous = channels
        self.layers = nn.Sequential(
            *layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(previous, dim)
        )

    def forward(self, x):
        return self.layers(x)


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x, mask):
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        score = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        # Keep normalization in float32: 1e-8 underflows in fp16 for all-missing streams.
        weights = score.float().softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return (x * weights.unsqueeze(-1)).sum(1)


class VisualHAR(nn.Module):
    """Methods 2-8. Missing frames never contribute votes or probability mass."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.method = int(config["method"])
        default_modalities = MODALITIES[:2] if self.method in (3, 5) else MODALITIES
        self.modalities = tuple(config.get("modalities", default_modalities))
        if not self.modalities or any(item not in MODALITIES for item in self.modalities):
            raise ValueError("modalities must be a non-empty subset of IR, Depth_Color, Thermal")
        self.modality_indices = [MODALITIES.index(item) for item in self.modalities]
        dim, dropout = config["dim"], config["dropout"]
        count = len(self.modalities)
        encoders = []
        for _ in self.modalities:
            if self.method == 5:
                from torchvision.models import resnet18

                encoder = resnet18(weights=None)
                encoder.fc = nn.Linear(encoder.fc.in_features, dim)
            else:
                encoder = SmallCNN(
                    dim,
                    se=self.method == 8,
                    spatial_attention=bool(config.get("spatial_attention", False)),
                )
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        self.temporal = nn.ModuleList([TemporalAttention(dim) for _ in self.modalities])
        self.branches = nn.ModuleList([nn.Linear(dim, 40) for _ in self.modalities])
        self.head = nn.Sequential(
            nn.LayerNorm(dim * count), nn.Dropout(dropout), nn.Linear(dim * count, 40)
        )
        # A small content-and-availability gate prevents a weak or absent stream
        # from contributing as strongly as the consistently useful Depth stream.
        self.use_modality_gate = bool(config.get("modality_gate", False))
        if self.use_modality_gate:
            self.modality_gate = nn.Sequential(
                nn.Linear(dim + 1, max(16, dim // 2)), nn.SiLU(), nn.Linear(max(16, dim // 2), 1)
            )
        if self.method == 7:
            layer = nn.TransformerEncoderLayer(
                dim,
                nhead=4,
                dim_feedforward=dim * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=2, enable_nested_tensor=False
            )
            self.modality_embedding = nn.Parameter(torch.randn(1, count, 1, dim) * 0.02)
            self.position = nn.Parameter(torch.randn(1, 1, config["frames"], dim) * 0.02)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.transformer_head = nn.Linear(dim, 40)

    def forward(self, images, mask):
        # images B,M,T,C,H,W; mask B,M,T. Select only requested modalities.
        count = len(self.modalities)
        images = images[:, self.modality_indices]
        mask = mask[:, self.modality_indices].bool()
        b, _, t, c, h, w = images.shape
        sequences, pooled = [], []
        for m, encoder in enumerate(self.encoders):
            sequence = encoder(images[:, m].reshape(b * t, c, h, w)).reshape(b, t, -1)
            sequence = sequence * mask[:, m, :, None]
            sequences.append(sequence)
            if self.method == 6 and not self.config.get("attention_pooling", False):
                feature = sequence.sum(1) / mask[:, m].sum(1, keepdim=True).clamp_min(1)
            else:
                feature = self.temporal[m](sequence, mask[:, m])
            pooled.append(feature)
        present = mask.any(-1)
        branch = torch.stack([head(x) for head, x in zip(self.branches, pooled, strict=True)], 1)
        if self.use_modality_gate:
            gated = []
            for m, feature in enumerate(pooled):
                availability = present[:, m : m + 1].float()
                gate = torch.sigmoid(self.modality_gate(torch.cat([feature, availability], -1)))
                gated.append(feature * gate * availability)
            pooled = gated
        fused = self.head(torch.cat(pooled, -1))
        if self.method == 7:
            if self.config.get("transformer_layout") == "factorized":
                tokens = torch.stack(pooled, 1) + self.modality_embedding[:, :, 0]
                token_mask = ~present
            else:
                tokens = torch.stack(sequences, 1) + self.modality_embedding + self.position[:, :, :t]
                tokens = tokens.reshape(b, count * t, -1)
                token_mask = ~mask.reshape(b, -1)
            tokens = torch.cat([self.cls.expand(b, -1, -1), tokens], 1)
            padding = torch.cat(
                [torch.zeros(b, 1, device=mask.device, dtype=torch.bool), token_mask], 1
            )
            fused = self.transformer_head(
                self.transformer(tokens, src_key_padding_mask=padding)[:, 0]
            )
        soft = branch.float().softmax(-1) * present.unsqueeze(-1)
        if self.method == 2:
            # One hard vote from each available modality plus the concatenation head.
            weights = torch.as_tensor(
                self.config.get("branch_weights", [1.0] * count), device=images.device, dtype=soft.dtype
            ).view(1, count, 1)
            soft_sum = (soft * weights).sum(1) + fused.float().softmax(-1)
            hard = (F.one_hot(branch.argmax(-1), 40) * present.unsqueeze(-1) * weights).sum(1)
            hard = hard + F.one_hot(fused.argmax(-1), 40)
            # Fractional soft tie-break cannot overturn a one-vote lead.
            scores = hard + 0.25 * soft_sum / (present.sum(1, keepdim=True) + 1)
            probabilities = scores / scores.sum(-1, keepdim=True)
        elif self.method == 4:
            weights = torch.as_tensor(
                self.config.get("branch_weights", [1.0] * count), device=images.device, dtype=soft.dtype
            ).view(1, count, 1)
            scores = (soft * weights).sum(1)
            probabilities = scores / (present.unsqueeze(-1) * weights).sum(1).clamp_min(1)
            probabilities = torch.where(
                present.any(1, keepdim=True), probabilities, torch.full_like(probabilities, 1 / 40)
            )
        else:
            probabilities = fused.float().softmax(-1)
        return {
            "logits": fused,
            "branches": branch,
            "present": present,
            "probabilities": probabilities,
        }

    def loss(self, output, labels):
        if self.method == 4:
            main = F.nll_loss(output["probabilities"].clamp_min(1e-8).log(), labels)
        else:
            main = F.cross_entropy(output["logits"], labels)
        if self.method in (2, 4):
            b, m, c = output["branches"].shape
            losses = F.cross_entropy(
                output["branches"].reshape(b * m, c),
                labels[:, None].expand(-1, m).reshape(-1),
                reduction="none",
            )
            present = output["present"].reshape(-1)
            auxiliary = (losses * present).sum() / present.sum().clamp_min(1)
            main = main + 0.5 * auxiliary
        return main

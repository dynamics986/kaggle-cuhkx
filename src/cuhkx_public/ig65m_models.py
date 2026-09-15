"""Upstream-compatible R(2+1)D-34 constructors from moabitcoin/ig65m-pytorch.

This is architecture/pretraining support only. It is not the Kaggle author's
quantized, task-specific ``ensemble_packed.pt`` classifier.
"""

from __future__ import annotations

import torch.hub
import torch.nn as nn
from torchvision.models.video.resnet import BasicBlock, Conv2Plus1D, R2Plus1dStem, VideoResNet

model_urls = {
    "r2plus1d_34_8_ig65m": (
        "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/"
        "r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth"
    ),
    "r2plus1d_34_32_ig65m": (
        "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/"
        "r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth"
    ),
    "r2plus1d_34_8_kinetics": (
        "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/"
        "r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth"
    ),
    "r2plus1d_34_32_kinetics": (
        "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/"
        "r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth"
    ),
}


def r2plus1d_34_8_ig65m(num_classes, pretrained=False, progress=False):
    assert not pretrained or num_classes == 487, "pretrained on 487 classes"
    return r2plus1d_34(num_classes, pretrained, progress, "r2plus1d_34_8_ig65m")


def r2plus1d_34_32_ig65m(num_classes, pretrained=False, progress=False):
    assert not pretrained or num_classes == 359, "pretrained on 359 classes"
    return r2plus1d_34(num_classes, pretrained, progress, "r2plus1d_34_32_ig65m")


def r2plus1d_34_8_kinetics(num_classes, pretrained=False, progress=False):
    assert not pretrained or num_classes == 400, "pretrained on 400 classes"
    return r2plus1d_34(num_classes, pretrained, progress, "r2plus1d_34_8_kinetics")


def r2plus1d_34_32_kinetics(num_classes, pretrained=False, progress=False):
    assert not pretrained or num_classes == 400, "pretrained on 400 classes"
    return r2plus1d_34(num_classes, pretrained, progress, "r2plus1d_34_32_kinetics")


def r2plus1d_34(num_classes, pretrained=False, progress=False, arch=None):
    model = VideoResNet(
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem,
    )
    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    # Caffe2 / VMZ compatibility fixes preserved from the upstream source.
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eps = 1e-3
            module.momentum = 0.9
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

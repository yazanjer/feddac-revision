"""Model architectures.

* CNN_MNIST  - identical to the network used in the original submission (2 conv + 1 FC).
* CNN_CIFAR  - 2 conv (ReLU + max-pool) + 2 FC, as described in Section 5.1.
* ResNet18GN - CIFAR-style ResNet-18 with GroupNorm (BatchNorm is ill-suited to small,
               non-IID client batches and to per-sample-gradient DP-SGD).  Used for the new
               CIFAR-100 and Tiny-ImageNet experiments.
None of the models contains BatchNorm, so every model is directly compatible with DP-SGD.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_MNIST(nn.Module):
    def __init__(self, num_classes=10, in_ch=1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return self.out(x.flatten(1))


class CNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.out(x)


def _gn(c):
    return nn.GroupNorm(min(32, c // 4), c)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = _gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = _gn(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), _gn(cout))

    def forward(self, x):
        o = F.relu(self.n1(self.c1(x)))
        o = self.n2(self.c2(o))
        return F.relu(o + self.sc(x))


class ResNet18GN(nn.Module):
    def __init__(self, num_classes=100, in_ch=3, input_size=32):
        super().__init__()
        stem_stride = 2 if input_size >= 64 else 1  # Tiny-ImageNet: 64x64 -> 32x32 after stem
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 64, 3, stem_stride, 1, bias=False), _gn(64), nn.ReLU())
        cfg = [(64, 1), (128, 2), (256, 2), (512, 2)]
        layers, cin = [], 64
        for cout, s in cfg:
            layers += [BasicBlock(cin, cout, s), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.out(x)


def build_model(dataset: str, num_classes: int):
    if dataset in ("mnist", "synthetic"):
        return CNN_MNIST(num_classes)
    if dataset == "cifar10":
        return CNN_CIFAR(num_classes)
    if dataset == "cifar100":
        return ResNet18GN(num_classes, input_size=32)
    if dataset == "tinyimagenet":
        return ResNet18GN(num_classes, input_size=64)
    raise ValueError(dataset)


def output_layer(model: nn.Module) -> nn.Linear:
    """Final classification layer (used by the label-distribution inference attack)."""
    return model.out

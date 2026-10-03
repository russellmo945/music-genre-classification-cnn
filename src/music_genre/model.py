"""The six-block CNN topology selected in the original model-set-13 experiment."""

import torch
from torch import nn


class GenreCNN(nn.Module):
    def __init__(self, conv_layers=6, initial_channels=12, hidden=(512, 128, 64),
                 num_classes=10, input_size=64, dropout=.3):
        super().__init__()
        if conv_layers < 1 or input_size < 2 ** conv_layers or not hidden or initial_channels < 1:
            raise ValueError("Invalid CNN dimensions; each convolution block halves the spatial size")
        blocks = []
        channels = 3
        for i in range(conv_layers):
            out = initial_channels * 2 ** i
            blocks += [nn.Conv2d(channels, out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(dropout)]
            channels = out
        self.features = nn.Sequential(*blocks)
        width = channels * (input_size // (2 ** conv_layers)) ** 2
        layers = [nn.Flatten()]
        for out in hidden:
            layers += [nn.Linear(width, out), nn.ReLU()]
            width = out
        layers.append(nn.Linear(width, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

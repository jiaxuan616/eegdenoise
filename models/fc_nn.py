"""Fully-connected neural network for EEG denoising."""

import torch.nn as nn


class fcNN(nn.Module):
    def __init__(self, datanum):
        super(fcNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
        )

    def forward(self, x):
        return self.model(x)
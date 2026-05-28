"""Simple 1D CNN for EEG denoising."""

import torch.nn as nn


class simple_CNN(nn.Module):
    def __init__(self, datanum):
        super(simple_CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc = nn.Linear(64 * datanum, datanum)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.shape[-1] == 1:
            x = x.transpose(1, 2)
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
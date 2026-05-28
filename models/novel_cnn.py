"""Novel CNN architecture with deep channel expansion for EEG denoising."""

import torch.nn as nn


class Novel_CNN(nn.Module):
    def __init__(self, datanum=1024):
        super(Novel_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(2048, 2048, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc = nn.Linear(2048 * (datanum // 64), datanum)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.shape[-1] == 1:
            x = x.transpose(1, 2)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
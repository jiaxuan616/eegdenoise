"""LSTM-based model for EEG denoising."""

import torch.nn as nn


class RNN_lstm(nn.Module):
    def __init__(self, datanum):
        super(RNN_lstm, self).__init__()
        self.datanum = datanum
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
"""Complex CNN with multi-scale ResNet blocks for EEG denoising."""

import torch
import torch.nn as nn


class Res_BasicBlock(nn.Module):
    """Residual basic block with configurable kernel size."""

    def __init__(self, kernelsize, stride=1):
        super(Res_BasicBlock, self).__init__()
        padding = kernelsize // 2
        self.bblock = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=kernelsize, stride=stride, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=kernelsize, stride=1, padding=padding),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=kernelsize, stride=1, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.bblock(x) + x


class BasicBlockall(nn.Module):
    """Multi-scale ResNet block combining kernel sizes 3, 5, 7."""

    def __init__(self, stride=1):
        super(BasicBlockall, self).__init__()
        self.bblock3 = nn.Sequential(Res_BasicBlock(3), Res_BasicBlock(3))
        self.bblock5 = nn.Sequential(Res_BasicBlock(5), Res_BasicBlock(5))
        self.bblock7 = nn.Sequential(Res_BasicBlock(7), Res_BasicBlock(7))

    def forward(self, x):
        out3 = self.bblock3(x)
        out5 = self.bblock5(x)
        out7 = self.bblock7(x)
        return torch.cat([out3, out5, out7], dim=1)


class Complex_CNN(nn.Module):
    def __init__(self, datanum):
        super(Complex_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.basic_block_all = BasicBlockall()
        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.fc = nn.Linear(32 * datanum, datanum)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.shape[-1] == 1:
            x = x.transpose(1, 2)
        out = self.conv1(x)
        out = self.basic_block_all(out)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
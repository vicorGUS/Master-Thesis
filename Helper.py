import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearningCurvePlot:

    def __init__(self, title=None, metrics='Loss'):
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylabel(metrics)
        self.ax.set_xlabel('Epoch')

        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, label=None):
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def add_histogram(self, y, label=None):
        if label is not None:
            self.ax.hist(y, bins=20, label=label)
        else:
            self.ax.hist(y, bins=20)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='test.png'):
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


class CustomTensorDataset(Dataset):
    def __init__(self, tensors, train=True):
        self.tensors = tensors
        self.train = train

    def __getitem__(self, index):
        if self.train:
            x = self.tensors[0][index]
            y = self.tensors[1][index]
            return x, y
        else:
            x = self.tensors[index]
            return x

    def __len__(self):
        if self.train:
            return len(self.tensors[0])
        else:
            return len(self.tensors)


class Customloss(nn.Module):
    def __init__(self):
        super(Customloss, self).__init__()

    def forward(self, y_pred, y_true):
        # Extract predicted parameters
        amp_pred, width_pred, RM_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        # Extract true parameters
        amp_true, width_true, RM_true = y_true[:, 0], y_true[:, 1], y_true[:, 2]

        loss_amp = F.mse_loss(amp_pred, amp_true)

        loss_width = F.mse_loss(width_pred, width_true)

        loss_RM = F.mse_loss(RM_pred, RM_true)

        # Combine individual loss functions using weighted sum
        return (loss_amp + loss_width + 1/10 * loss_RM).mean()

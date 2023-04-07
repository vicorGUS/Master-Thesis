import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearningCurvePlot:

    def __init__(self, title=None, xlabel='Epoch'):
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylabel('Loss')
        if xlabel == 'Epoch':
            self.ax.set_xlabel('Epoch')
        else:
            self.ax.set_xlabel(f'{xlabel}')

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
        self.low_weight, self.high_weight = 1, 2

    def forward(self, y_pred, y_true, SNR):
        # Extract predicted Gaussian parameters
        amp_width_pred, RM_base_pred = y_pred[:, :2], y_pred[:, 2:4]

        # Extract true Gaussian parameters
        amp_width_true, RM_base_true = y_true[:, :2], y_true[:, 2:4]

        n = int(len(SNR) / 2)
        weights = torch.hstack((torch.linspace(self.low_weight, .1, n), torch.linspace(.1, self.high_weight, n)))

        thresholds = torch.linspace(3, 12, len(SNR))

        index = np.argmin(torch.abs(SNR[:, None] - thresholds), axis=1)

        weight = weights[index].reshape(-1, 1)

        # Define loss functions for pairs of parameters

        loss_amp_width = (F.mse_loss(amp_width_pred, amp_width_true, reduction='none') * weight).mean()

        loss_RM_base = F.mse_loss(RM_base_pred, RM_base_true)

        # Combine individual loss functions using weighted sum
        loss = (loss_amp_width + loss_RM_base).mean()
        return loss
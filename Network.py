import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi


class InceptionLayer(nn.Module):
    def __init__(self, in_channels, reduce_3x3, out_3x3, reduce_5x5, out_5x5, reduce_7x7, out_7x7, reduce_9x9, out_9x9, out_pool):
        super(InceptionLayer, self).__init__()
        # 3x3 conv branch
        self.branch3x3_reduce = nn.Conv1d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3 = nn.Conv1d(reduce_3x3, out_3x3, kernel_size=3, stride=1, padding=1)
        self.bn3x3 = nn.BatchNorm1d(out_3x3)
        # 5x5 conv branch
        self.branch5x5_reduce = nn.Conv1d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5 = nn.Conv1d(reduce_5x5, out_5x5, kernel_size=5, stride=1, padding=2)
        self.bn5x5 = nn.BatchNorm1d(out_5x5)
        # 7x7 conv branch
        self.branch7x7_reduce = nn.Conv1d(in_channels, reduce_7x7, kernel_size=1)
        self.branch7x7 = nn.Conv1d(reduce_7x7, out_7x7, kernel_size=7, stride=1, padding=3)
        self.bn7x7 = nn.BatchNorm1d(out_7x7)
        # 9x9 conv branch
        self.branch9x9_reduce = nn.Conv1d(in_channels, reduce_9x9, kernel_size=1)
        self.branch9x9 = nn.Conv1d(reduce_9x9, out_9x9, kernel_size=9, stride=1, padding=4)
        self.bn9x9 = nn.BatchNorm1d(out_9x9)
        # Max pooling branch
        self.branch_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = nn.Conv1d(in_channels, out_pool, kernel_size=1)
        self.bn_pool = nn.BatchNorm1d(out_pool)

    def forward(self, x):
        # 3x3 conv branch
        branch3x3 = self.branch3x3(self.branch3x3_reduce(x))
        branch3x3 = F.relu(self.bn3x3(branch3x3))
        # 5x5 conv branch
        branch5x5 = self.branch5x5(self.branch5x5_reduce(x))
        branch5x5 = F.relu(self.bn5x5(branch5x5))
        # 7x7 conv branch
        branch7x7 = self.branch7x7(self.branch7x7_reduce(x))
        branch7x7 = F.relu(self.bn7x7(branch7x7))
        # 9x9 conv branch
        branch9x9 = self.branch9x9(self.branch9x9_reduce(x))
        branch9x9 = F.relu(self.bn9x9(branch9x9))
        # Max pooling branch
        branch_pool = self.branch_pool_conv(self.branch_pool(x))
        branch_pool = F.relu(self.bn_pool(branch_pool))
        # Concatenate branches along the channel dimension
        output = torch.cat([branch3x3, branch5x5, branch7x7, branch9x9, branch_pool], dim=1)
        return output


class SubsampleLayer(nn.Module):
    def __init__(self, in_channels, reduce_3x3, out_3x3, reduce_5x5, out_5x5, reduce_7x7, out_7x7, reduce_9x9, out_9x9, out_pool):
        super(SubsampleLayer, self).__init__()
        # 3x3 conv branch
        self.branch3x3_reduce = nn.Conv1d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3 = nn.Conv1d(reduce_3x3, out_3x3, kernel_size=3, stride=2, padding=1)
        self.bn3x3 = nn.BatchNorm1d(out_3x3)
        # 5x5 conv branch
        self.branch5x5_reduce = nn.Conv1d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5 = nn.Conv1d(reduce_5x5, out_5x5, kernel_size=5, stride=2, padding=2)
        self.bn5x5 = nn.BatchNorm1d(out_5x5)
        # 7x7 conv branch
        self.branch7x7_reduce = nn.Conv1d(in_channels, reduce_7x7, kernel_size=1)
        self.branch7x7 = nn.Conv1d(reduce_7x7, out_7x7, kernel_size=7, stride=2, padding=3)
        self.bn7x7 = nn.BatchNorm1d(out_7x7)
        # 9x9 conv branch
        self.branch9x9_reduce = nn.Conv1d(in_channels, reduce_9x9, kernel_size=1)
        self.branch9x9 = nn.Conv1d(reduce_9x9, out_9x9, kernel_size=9, stride=2, padding=4)
        self.bn9x9 = nn.BatchNorm1d(out_9x9)
        # Max pooling branch
        self.branch_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.branch_pool_conv = nn.Conv1d(in_channels, out_pool, kernel_size=1)
        self.bn_pool = nn.BatchNorm1d(out_pool)

    def forward(self, x):
        # 3x3 conv branch
        branch3x3 = self.branch3x3(self.branch3x3_reduce(x))
        branch3x3 = F.relu(self.bn3x3(branch3x3))
        # 5x5 conv branch
        branch5x5 = self.branch5x5(self.branch5x5_reduce(x))
        branch5x5 = F.relu(self.bn5x5(branch5x5))
        # 7x7 conv branch
        branch7x7 = self.branch7x7(self.branch7x7_reduce(x))
        branch7x7 = F.relu(self.bn7x7(branch7x7))
        # 9x9 conv branch
        branch9x9 = self.branch9x9(self.branch9x9_reduce(x))
        branch9x9 = F.relu(self.bn9x9(branch9x9))
        # Max pooling branch
        branch_pool = self.branch_pool_conv(self.branch_pool(x))
        branch_pool = F.relu(self.bn_pool(branch_pool))
        # Concatenate branches along the channel dimension
        output = torch.cat([branch3x3, branch5x5, branch7x7, branch9x9, branch_pool], dim=1)
        return output


class ClassifyingCNN(nn.Module):
    def __init__(self):
        super(ClassifyingCNN, self).__init__()
        self.inception1 = InceptionLayer(in_channels=1, reduce_3x3=32, out_3x3=64, reduce_5x5=32, out_5x5=64,
                                         reduce_7x7=32, out_7x7=64, reduce_9x9=32, out_9x9=64, out_pool=64)
        self.subsample = SubsampleLayer(in_channels=320, reduce_3x3=32, out_3x3=64, reduce_5x5=32, out_5x5=64,
                                        reduce_7x7=32, out_7x7=64, reduce_9x9=32, out_9x9=64, out_pool=64)
        self.inception2 = InceptionLayer(in_channels=320, reduce_3x3=32, out_3x3=64, reduce_5x5=32, out_5x5=64,
                                         reduce_7x7=32, out_7x7=64, reduce_9x9=32, out_9x9=64, out_pool=64)
        self.fc1 = nn.Linear(320 * 100, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.inception1(x)
        x = self.subsample(x)
        x = self.inception2(x)
        x = F.dropout(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ParameterizingCNN(nn.Module):
    def __init__(self):
        super(ParameterizingCNN, self).__init__()
        self.inception1 = InceptionLayer(in_channels=1, reduce_3x3=32, out_3x3=64, reduce_5x5=32, out_5x5=64,
                                         reduce_7x7=32, out_7x7=64, reduce_9x9=32, out_9x9=64, out_pool=64)
        self.subsample = SubsampleLayer(in_channels=320, reduce_3x3=32, out_3x3=64, reduce_5x5=32, out_5x5=64,
                                        reduce_7x7=32, out_7x7=64, reduce_9x9=32, out_9x9=64, out_pool=64)
        self.inception2 = InceptionLayer(in_channels=320, reduce_3x3=32, out_3x3=64, reduce_5x5=32, out_5x5=64,
                                         reduce_7x7=32, out_7x7=64, reduce_9x9=32, out_9x9=64, out_pool=64)
        self.fc1 = nn.Linear(320 * 100, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.inception1(x)
        x = self.subsample(x)
        x = self.inception2(x)
        x = F.dropout(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
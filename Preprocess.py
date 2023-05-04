import numpy as np
import torch
import torchvision.transforms as transforms
from astropy.io import fits
from torch.utils.data import DataLoader
from Helper import CustomTensorDataset


class Preprocess:
    def __init__(self):
        self.P = None
        self.FD = np.linspace(-100, 100, 200)
        self.wl2_range = (299792458 / np.linspace(350e6, 1050e6, 700)[::-1]) ** 2


    def RMsynthesis(self, data_path):
        hdu = fits.open(data_path)[0]
        data = hdu.data
        #self.header = self.hdu.header
        Q = data[:, 0]
        U = data[:, 1]
        P = Q + 1j * U
        wl2_0 = np.mean(self.wl2_range)
        M = np.exp(-2j * self.FD[:, np.newaxis] * (self.wl2_range - wl2_0)).reshape(len(self.FD), len(self.wl2_range))
        return torch.tensor(1 / len(self.wl2_range) * np.einsum('ijk,li->jkl', P, M), dtype=torch.cfloat)


    def create_loaders(self, data, batch_size, train=True):
        # Extract inputs and labels from the data

        # Split the data into training and validation sets
        if train:
            mean = torch.mean(data[0], dim=(1, 2))
            std = torch.std(data[0], dim=(1, 2))

            normalize = transforms.Normalize(mean=mean, std=std)
            data[0] = normalize(data[0])

            dataset = CustomTensorDataset(tensors=[data[0], data[1]])

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            data_train, data_val = torch.utils.data.random_split(dataset, [train_size, val_size])

            X_train, y_train = data_train[:][0], data_train[:][1]
            X_val, y_val = data_val[:][0], data_val[:][1]

            # Create training and validation datasets
            train_data = CustomTensorDataset(tensors=[X_train, y_train])
            val_data = CustomTensorDataset(tensors=[X_val, y_val])

            # Create data loaders for training and validation datasets
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                      num_workers=(8 if torch.cuda.is_available() else 0))
            val_loader = DataLoader(dataset=val_data, batch_size=batch_size,
                                    num_workers=(8 if torch.cuda.is_available() else 0))

            return train_loader, val_loader
        else:
            mean = torch.mean(data, dim=(1, 2))
            std = torch.std(data, dim=(1, 2))

            normalize = transforms.Normalize(mean=mean, std=std)
            inputs = normalize(data)

            # Create test dataset
            test_data = CustomTensorDataset(tensors=inputs, train=False)

            # Create data loader for test dataset
            test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                     num_workers=(8 if torch.cuda.is_available() else 0))

            return test_loader

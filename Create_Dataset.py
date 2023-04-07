from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Helper import CustomTensorDataset
import numpy as np
import torch


def create_loaders(data, batch_size, train=True):

    if not train:
        inputs = [d for d in data]

        test_data = CustomTensorDataset(tensors=inputs, train=train)
        n_workers = (8 if torch.cuda.is_available()
                     else 0)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        return test_loader

    else:
        inputs = [d for d in data[0]]
        labels = [d for d in data[1]]
        X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=0.2, random_state=42)

        train_data = CustomTensorDataset(tensors=[X_train, y_train])
        val_data = CustomTensorDataset(tensors=[X_val, y_val])
        n_workers = (8 if torch.cuda.is_available()
                     else 0)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=n_workers)
        return train_loader, val_loader
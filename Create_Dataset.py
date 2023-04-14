import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Helper import CustomTensorDataset


def create_loaders(data, batch_size, train=True):
    # Extract inputs and labels from the data

    # Split the data into training and validation sets
    if train:
        inputs, labels = data
        X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=0.2, random_state=42)

        # Create training and validation datasets
        train_data = CustomTensorDataset(tensors=[X_train, y_train])
        val_data = CustomTensorDataset(tensors=[X_val, y_val])

        # Create data loaders for training and validation datasets
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=(8 if torch.cuda.is_available() else 0))
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=(8 if torch.cuda.is_available() else 0))

        return train_loader, val_loader
    else:
        inputs = data
        # Create test dataset
        test_data = CustomTensorDataset(tensors=inputs, train=False)

        # Create data loader for test dataset
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=(8 if torch.cuda.is_available() else 0))

        return test_loader

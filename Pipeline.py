import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from Preprocess import Preprocess
from Helper import LearningCurvePlot
from Train import training_loop
from Network import InceptionCNN


class Pipeline:
    def __init__(self, num_epochs, learning_rate, batch_size):
        self.types_pred = []
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.preprocessing = Preprocess()

    def train(self, loss_fn, model, train_loader, val_loader, num_epochs, classifying=False):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        best_model, train_losses, val_losses, train_accs, val_accs = training_loop(model, optimizer, loss_fn,
                                                                                   train_loader, val_loader,
                                                                                   classifying, num_epochs)
        return best_model, train_losses, val_losses, train_accs, val_accs

    def classifying(self):

        types = ['Gaussian', 'tophat']

        for j, name in enumerate(types):
            inputs = self.preprocessing.RMsynthesis(f'Data/QU_train_{name}.fits')
            outputs = torch.load('Data/outputs_train.npz')
            x = torch.abs(inputs)
            y = outputs[0].long()
            data = [x.view(-1, 1, x.shape[2]), y.reshape(-1)]

            # Load the model and loss function
            model = InceptionCNN(n_outputs=2).to(torch.float)
            model.load_state_dict(torch.load(f'Model_params/Classifying_model_params_{name}.pth', map_location=device))
            loss_fn = nn.CrossEntropyLoss()

            train_loader, val_loader = self.preprocessing.create_loaders(data, self.batch_size)

            # Train the model
            now = time.time()
            best_model, train_losses, val_losses, train_accs, val_accs = self.train(loss_fn, model, train_loader,
                                                                                    val_loader, int(self.num_epochs/2),
                                                                                    classifying=True)
            print('Classifying {} took {:.0f} minutes'.format(name, (time.time() - now) / 60))
            # Save the best model
            torch.save(best_model.state_dict(), f'Model_params/Classifying_model_params_{name}.pth')

            # Plot the loss curves
            Plot_loss = LearningCurvePlot(title='Classifying Loss Test')
            Plot_acc = LearningCurvePlot(title='Classifying Accuracy Test', metrics='Accuracy')
            Plot_acc.add_curve(train_accs, label='Training')
            Plot_acc.add_curve(val_accs, label='Validation')
            Plot_loss.add_curve(train_losses, label='Training')
            Plot_loss.add_curve(val_losses, label='Validation')
            Plot_loss.save(f'Results/Classifying_Loss_{name}.png')
            Plot_acc.save(f'Results/Classifying_Accuracy_{name}.png')

            # Create dataset for parmeterization
            test_loader = self.preprocessing.create_loaders(data[0], self.batch_size, train=False)
            with torch.no_grad():
                outputs = []
                for i, x in enumerate(test_loader, 1):
                    x = x.to(device)
                    output = best_model.forward(x.float())
                    outputs.append(output)
                outputs = torch.cat(outputs, dim=0)
                _, predicted = torch.max(outputs, 1)
            self.types_pred.append(predicted.long())

    def parameterizing(self):

        # Split the data into two classes
        types_dict = {'Gaussian': (self.types_pred[0] == 1), 'tophat': (self.types_pred[1] == 1)}

        for (j, (name, t)) in enumerate(types_dict.items()):
            inputs = self.preprocessing.RMsynthesis(f'Data/QU_train_{name}.fits')
            outputs = torch.load('Data/outputs_train.npz')
            x = torch.abs(inputs)
            data = [x.view(-1, 1, x.shape[2])[t], outputs[1].view(-1, 3)[t]]

            # Load the model parameters
            model = InceptionCNN(n_outputs=3).to(torch.float)
            model.load_state_dict(torch.load(f'Model_params/Parameterizing_model_params_{name}.pth', map_location=device))
            loss_fn = nn.MSELoss()

            # Create data loaders for each class
            train_loader, val_loader = self.preprocessing.create_loaders(data, self.batch_size)

            # Train the model
            now = time.time()
            best_model, train_losses, val_losses, _, _ = self.train(loss_fn, model, train_loader, val_loader,
                                                                    self.num_epochs)
            print('Parameterizing {} took {:.0f} minutes'.format(name, (time.time() - now) / 60))

            # Save the best model
            torch.save(best_model.state_dict(), f'Model_params/Parameterizing_model_params_{name}.pth')

            # Plot the loss curves
            Plot_loss = LearningCurvePlot(title=f'Parameterizing Loss Test {name}')
            Plot_loss.add_curve(train_losses, label='Training')
            Plot_loss.add_curve(val_losses, label='Validation')
            Plot_loss.save(f'Results/Parameterizing_Loss_{name}.png')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tester = Pipeline(num_epochs=2, learning_rate=1e-5, batch_size=16)

    tester.classifying()

    tester.parameterizing()
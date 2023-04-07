import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from Helper import LearningCurvePlot
from Train import training_loop
from Create_Dataset import create_loaders
from torch.utils.data import DataLoader
from Network import ClassifyingCNN, ParameterizingCNN


class Pipeline:

    def __init__(self, num_epochs, learning_rate, weight_decay, batch_size):
        self.data_type = None
        self.data_clean = []
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.data = torch.load('Data/data_train.npz')

    def train(self, loss_fn, model, train_loader, val_loader, classifying=False):
        now = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_model, train_losses, val_losses = training_loop(model, optimizer, loss_fn, train_loader, val_loader,
                                                             classifying, num_epochs=self.num_epochs)
        print('Running one setting takes {:.1f} minutes'.format((time.time() - now) / 60))
        return best_model, train_losses, val_losses


    def classifying(self, save_output=False):
        data = [self.data[0], self.data[2]]
        model = ClassifyingCNN().float()
        model.load_state_dict(torch.load('Classifying_model_params.pth'))
        loss_fn = nn.CrossEntropyLoss()

        Plot_loss = LearningCurvePlot(title='Classifying Loss Test')

        train_loader, val_loader = create_loaders(data, self.batch_size)
        best_model, train_losses, val_losses = self.train(loss_fn, model, train_loader, val_loader, classifying=True)
        torch.save(best_model.state_dict(), 'Classifying_model_params.pth')
        Plot_loss.add_curve(train_losses, label='Training')
        Plot_loss.add_curve(val_losses, label='Validation')
        Plot_loss.save('Classifying_Loss.png')

        if save_output:
            outputs = []
            test_loader = create_loaders(self.data[0], 32, train=False)
            with torch.no_grad():
                for i, x in enumerate(test_loader, 1):
                    output = best_model.forward(x.float())
                    outputs.extend(np.argmax(output.detach().numpy(), axis=1))
            self.data_type = torch.LongTensor(np.array(outputs))

    def parameterizing(self, save_output=False):
        Gaussian = (self.data_type == 1)
        tophat = (self.data_type == 2)

        types_dict = {'Gaussian': Gaussian, 'tophat': tophat}

        model = ParameterizingCNN().float()
        loss_fn = nn.MSELoss().float()

        for name, t in types_dict.items():
            data = [self.data[0][t], self.data[1][t]]
            train_loader, val_loader = create_loaders(data, self.batch_size)
            Plot_loss = LearningCurvePlot(title=f'Parameterizing Loss Test {name}')

            model.load_state_dict(torch.load(f'Parameterizing_model_params_{name}.pth'))
            best_model, train_losses, val_losses = self.train(loss_fn, model, train_loader, val_loader)

            torch.save(best_model.state_dict(), f'Parameterizing_model_params_{name}.pth')

            Plot_loss.add_curve(train_losses, label='Training')
            Plot_loss.add_curve(val_losses, label='Validation')
            Plot_loss.save(f'Parameterizing_Loss_{name}.png')

            if save_output:
                outputs = []
                test_loader = create_loaders(self.data[0][t], 32, train=False)
                with torch.no_grad():
                    for i, x in enumerate(test_loader, 1):
                        output = best_model.forward(x.float())
                        outputs.extend(output)
                self.data_clean.append(outputs)


if __name__ == "__main__":

    tester = Pipeline(num_epochs=10, learning_rate=1e-5, weight_decay=0, batch_size=32)

    tester.classifying(save_output=True)

    tester.parameterizing(save_output=True)
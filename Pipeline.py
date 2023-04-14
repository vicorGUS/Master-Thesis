import numpy as np
import torch
import torch.nn as nn
import time
from Helper import LearningCurvePlot
from Train import training_loop
from Create_Dataset import create_loaders
from Network import InceptionCNN


class Pipeline:
    def __init__(self, num_epochs, learning_rate, weight_decay, batch_size):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.data = None  # The data will be loaded later

    def train(self, loss_fn, model, train_loader, val_loader, num_epochs, classifying=False):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_model, train_losses, val_losses = training_loop(model, optimizer, loss_fn, train_loader, val_loader,
                                                             classifying, num_epochs)
        return best_model, train_losses, val_losses

    def classifying(self, save_output=False):
        # Load the data from disk
        self.data = torch.load('Data/data_train.npz')
        x = self.data[0].float()
        y = self.data[2].long()
        data = [x.view(-1, 1, 200), y.reshape(-1)]

        # Load the model and loss function
        model = InceptionCNN().float()
        model.load_state_dict(torch.load('Classifying_model_params.pth'))
        loss_fn = nn.CrossEntropyLoss()

        # Create data loaders
        train_loader, val_loader = create_loaders(data, self.batch_size)

        # Train the model
        now = time.time()
        best_model, train_losses, val_losses = self.train(loss_fn, model, train_loader, val_loader, 1, classifying=True)
        print('Classifying took {:.1f} minutes'.format((time.time() - now)/60))

        # Save the best model
        torch.save(best_model.state_dict(), 'Classifying_model_params.pth')

        # Plot the loss curves
        Plot_loss = LearningCurvePlot(title='Classifying Loss Test')
        Plot_loss.add_curve(train_losses, label='Training')
        Plot_loss.add_curve(val_losses, label='Validation')
        Plot_loss.save('Classifying_Loss.png')

        if save_output:
            # Generate predictions on the test set
            outputs = []
            test_loader = create_loaders(self.data[0].view(-1, 1, 200), self.batch_size, train=False)
            with torch.no_grad():
                for i, x in enumerate(test_loader, 1):
                    output = best_model.forward(x.float())
                    outputs.extend(np.argmax(output.detach().numpy(), axis=1))
            self.data[2] = torch.LongTensor(np.array(outputs))  # Update the labels with the predicted values

    def parameterizing(self):
        # Check if the classifying function has been called before
        if self.data is None:
            print('Error: Call the classifying function before the parameterizing function')
            return

        # Split the data into two classes
        Gaussian = (self.data[2] == 1)
        tophat = (self.data[2] == 2)

        types_dict = {'Gaussian': Gaussian, 'tophat': tophat}

        model = InceptionCNN().float()
        loss_fn = nn.MSELoss().float()

        for name, t in types_dict.items():
            # Create data loaders for each class
            data = [self.data[0].view(-1, 1, 200)[t], self.data[1].view(-1, 3)[t]]
            train_loader, val_loader = create_loaders(data, self.batch_size)

            # Load the model parameters
            model.load_state_dict(torch.load(f'Parameterizing_model_params_{name}.pth'))

            # Train the model
            now = time.time()
            best_model, train_losses, val_losses = self.train(loss_fn, model, train_loader, val_loader, 10)
            print('Parameterizing {} took {:.0f} minutes'.format(name, (time.time() - now)/60))

            # Save the best model
            torch.save(best_model.state_dict(), f'Parameterizing_model_params_{name}.pth')

            # Plot the loss curves
            Plot_loss = LearningCurvePlot(title=f'Parameterizing Loss Test {name}')
            Plot_loss.add_curve(train_losses, label='Training')
            Plot_loss.add_curve(val_losses, label='Validation')
            Plot_loss.save(f'Parameterizing_Loss_{name}.png')


if __name__ == "__main__":

    tester = Pipeline(num_epochs=10, learning_rate=1e-5, weight_decay=0, batch_size=64)

    tester.classifying(save_output=True)

    tester.parameterizing()
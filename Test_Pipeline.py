import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from Network import InceptionCNN
from Create_Dataset import create_loaders
from Helper import CustomTensorDataset


def bin_means(x, y, num_bins):
    """
    Calculates the mean of y values for each bin of x values.

    Args:
    - x: numpy array of shape (n,)
    - y: numpy array of shape (n,)
    - num_bins: int, number of bins to divide x into

    Returns:
    - means: numpy array of shape (num_bins,), where means[i] is the mean of y values
             for the i-th bin of x values
    """
    bin_boundaries = np.linspace(torch.min(x).item(), torch.max(x).item(), num_bins + 1, endpoint=True)
    bin_indices = np.digitize(x, bin_boundaries)
    means = np.array([torch.mean(y[bin_indices == i]) for i in range(1, num_bins + 1)])
    return means


class Test_Pipeline:
    def __init__(self):
        # Load the data
        self.data = torch.load('Data/data_test.npz')
        self.data_RM = torch.load('Data/data_RM.npz')

        # Reshape the data
        self.inputshapes = self.data[0].size()
        self.data[0] = self.data[0].reshape(-1, 1, 200)
        self.data[1] = self.data[1].reshape(-1, 3)
        self.data_RM[0] = self.data_RM[0].reshape(-1, 2000)
        self.data_RM[1] = self.data_RM[1].reshape(-1, 200)

        # Create a frequency domain range
        self.FD = torch.linspace(-100, 100, 200)

        # Initialize variables
        self.model = None
        self.num_samples = np.prod(self.data[0].shape[:2])
        self.types = None
        self.types_dict = None
        self.outputs = None
        self.labels = None
        self.diffs = None
        self.true_spectra = None
        self.RM_spectra = None

    def load_weights(self, test_type, signal_type=None):
        # Load the model and weights based on the test type and signal type
        self.model = InceptionCNN().float()
        if test_type == 'classifying':
            self.model.load_state_dict(torch.load('Classifying_model_params.pth'))
        elif test_type == 'parameterizing':
            if signal_type == 'Gaussian':
                self.model.load_state_dict(torch.load('Parameterizing_model_params_Gaussian.pth'))
            elif signal_type == 'tophat':
                self.model.load_state_dict(torch.load('Parameterizing_model_params_tophat.pth'))

    def test_classifier(self):
        # Load the classifier model
        self.load_weights(test_type='classifying')

        # Create a data loader for testing
        test_loader = create_loaders(self.data[0], 100, train=False)

        # Initialize the outputs tensor
        outputs = torch.empty(self.num_samples)
        outputs[:] = torch.nan

        # Iterate over the data loader and classify the signals
        with torch.no_grad():
            for i, x in enumerate(test_loader, 1):
                start = torch.where(torch.isnan(outputs))[0][0]
                output = self.model.forward(x.float())
                outputs[start: start + len(output)] = torch.argmax(output.detach(), dim=1)

        # Save the classified signal types
        self.types = outputs

    def test_parameter(self):
        # create boolean masks for Gaussian and tophat sources
        Gaussian = (self.types == 1)
        tophat = (self.types == 2)

        # create a dictionary mapping signal types to boolean masks
        self.types_dict = {'Gaussian': Gaussian, 'tophat': tophat}

        # initialize lists to store model outputs, true labels, and differences
        outputs = [torch.empty(sum(Gaussian), 3), torch.empty(sum(tophat), 3)]
        labels = [torch.empty(sum(Gaussian), 3), torch.empty(sum(tophat), 3)]
        diffs = [torch.empty(sum(Gaussian), 3), torch.empty(sum(tophat), 3)]

        # initialize tensors to store true spectra and RM synthesis spectra for each signal type
        self.true_spectra = [torch.empty(sum(Gaussian), 2000), torch.empty(sum(tophat), 2000)]
        self.RM_spectra = [torch.empty(sum(Gaussian), 200), torch.empty(sum(tophat), 200)]

        # initialize model outputs as NaNs
        outputs[0][:] = outputs[1][:] = torch.nan

        # loop over signal types and train corresponding models
        for (j, (name, t)) in enumerate(self.types_dict.items()):
            self.load_weights(test_type='parameterizing', signal_type=name)
            test_loader = create_loaders(self.data[0][t], 100, train=False)
            with torch.no_grad():
                for i, x in enumerate(test_loader, 1):
                    # get the first NaN index in outputs[j] to continue appending the model outputs
                    start = torch.where(torch.isnan(outputs[j]))[0][0]
                    output = self.model.forward(x.float())
                    outputs[j][start: start + len(output)] = output
            labels[j] = self.data[1][t].squeeze()
            self.true_spectra[j], self.RM_spectra[j] = self.data_RM[0][t], self.data_RM[1][t]
            diffs[j] = outputs[j] - labels[j]

        # store results as instance variables
        self.outputs = outputs
        self.labels = labels
        self.diffs = diffs

    def create_catalog(self):
        """
        Create a catalog based on the test results.
        The catalog will be saved in a csv file named 'Catalog.csv'.
        """
        amp_Gaussian, width_Gaussian, RM_Gaussian = self.outputs[0].T
        amp_tophat, width_tophat, RM_tophat = self.outputs[1].T

        x_Gaussian, y_Gaussian = np.where(self.types.reshape(self.inputshapes[:2]) == 1)
        x_tophat, y_tophat = np.where(self.types.reshape(self.inputshapes[:2]) == 2)

        dict_Gaussian = {'x [px]': x_Gaussian, 'y [px]': y_Gaussian, 'RM [rad m^(-2)]': RM_Gaussian,
                         '|F| [au]': amp_Gaussian, 'width [rad m^(-2)]': width_Gaussian,
                         'type': ['Gaussian'] * len(x_Gaussian)}

        dict_tophat = {'x [px]': x_tophat, 'y [px]': y_tophat, 'RM [rad m^(-2)]': RM_tophat,
                       '|F| [au]': amp_tophat, 'width [rad m^(-2)]': width_tophat, 'type': ['tophat'] * len(x_tophat)}

        df_Gaussian = pd.DataFrame(dict_Gaussian)
        df_tophat = pd.DataFrame(dict_tophat)

        df_combined = pd.concat([df_Gaussian, df_tophat], ignore_index=True)
        df_combined_sorted = df_combined.sort_values(['x [px]', 'y [px]']).reset_index(drop=True)

        df_combined_sorted.to_csv('Catalog.csv', index=False, sep=',', float_format='%.2f')

    def plot_results(self):
        """
        Plots the results of the test_parameter method by generating histograms of parameter errors,
        scatter plots of true vs predicted parameter values, and a plot of SNR vs parameter errors.
        """
        params = ['Amplitude', 'Width', 'Location']

        for (j, (name, t)) in enumerate(self.types_dict.items()):
            for i, param in enumerate(params):
                fig, ax = plt.subplots(ncols=3, figsize=(20, 10))
                x = np.linspace(torch.min(self.labels[j][:, i]).item(), torch.max(self.labels[j][:, i]).item(), 2)

                # Plot histogram of parameter errors
                ax[0].set_title(f'Histogram of {params[i]} error', fontsize=15)
                ax[0].hist(self.diffs[j][:, i], bins=30)
                ax[0].set_xlabel(f'$\Delta$ {params[i]}', fontsize=15)
                ax[0].set_ylabel('counts', fontsize=15)

                # Plot scatter plot of true vs predicted parameter values
                ax[1].set_title(f'True vs Predicted {params[i]}', fontsize=15)
                ax[1].plot(x, x, lw=2, c='r')
                ax[1].scatter(self.labels[j][:, i], self.outputs[j][:, i], marker='+', c='k', s=5)

                ax[1].set_xlabel(f'True {params[i]}', fontsize=15)
                ax[1].set_ylabel(f'Predicted {params[i]}', fontsize=15)
                ax[1].set_xlim((torch.min(self.data[1][t][:, i]), torch.max(self.outputs[j][:, i])))

                # Plot SNR vs parameter errors
                ax[2].set_title(f'SNR vs {params[i]} error', fontsize=15)
                ax[2].grid()

                ax[2].plot(np.arange(0.1, 16), bin_means(x=self.data[1][t][:, 0],
                                                         y=self.diffs[j][:, i] / self.labels[j][:, i], num_bins=16),
                           lw=2, c='r')

                ax[2].scatter(self.data[1][t][:, 0], self.diffs[j][:, i] / self.labels[j][:, i], marker='+', c='k', s=5,
                              alpha=.5)
                ax[2].set_xlabel('SNR', fontsize=15)
                ax[2].set_ylabel(f'$\Delta$ {params[i]} / {params[i]}', fontsize=15)

                plt.savefig(f'Experiment_results_{name}_{param}.png')

    def gaussian_source(self, amplitude, width, location):
        """
        Generates a Gaussian source with the given amplitude, width, and location.

        Args:
        - amplitude (torch.Tensor): tensor of shape (n_samples,) containing the amplitudes of the sources
        - width (torch.Tensor): tensor of shape (n_samples,) containing the widths of the sources
        - location (torch.Tensor): tensor of shape (n_samples,) containing the locations of the sources

        Returns:
        - gaussian (np.ndarray): array of shape (n_samples, 200) containing the generated Gaussian sources
        """

        return amplitude * np.exp(-1 / (2 * width ** 2) * (self.FD.reshape(1, -1) - location.reshape(-1, 1)) ** 2)

    def tophat_source(self, amplitde, width, location):
        """
            Generate a tophat source with the given amplitude, width, and location.

            Args:
                amplitude (torch.Tensor): A tensor of shape (num_sources,) representing the amplitudes of the sources.
                width (torch.Tensor): A tensor of shape (num_sources,) representing the widths of the sources.
                location (torch.Tensor): A tensor of shape (num_sources,) representing the locations of the sources.

            Returns:
                numpy.ndarray: A numpy array of shape (num_sources, 200) representing the tophat sources.
            """

        condition = (location - width / 2. < self.FD) & (self.FD < location + width / 2.)

        return np.where(condition, amplitde, 0)

    def plot_gaussians(self):
        # randomly select 2 rows and 3 columns from the samples classified as Gaussian
        randint = np.random.randint(0, sum(self.types == 1), (2, 3))

        # create a figure with two rows and three columns
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                # get the amplitude, sigma, and location parameters for the selected Gaussian source
                amp, sigma, loc = self.outputs[0].T[:, randint[i, j]]

                # generate the reconstructed Gaussian signal using the obtained parameters
                gaussian = self.gaussian_source(amp, sigma, loc)

                # get the true Gaussian signal from the data
                gaussian_true = self.true_spectra[0][randint[i, j]]

                # get the RM synthesis signal for the Gaussian source
                gaussian_RM = self.RM_spectra[0][randint[i, j]]

                # normalize the RM synthesis signal and true signal to have the same maximum amplitude
                gaussian_RM = gaussian_RM / torch.max(gaussian_RM) * torch.max(gaussian_true)

                # plot the reconstructed, true, and RM synthesis signals
                ax[i][j].plot(self.FD, gaussian.T, label='Reconstructed Signal')
                ax[i][j].plot(np.linspace(-100, 100, 2000), gaussian_true, label='True Signal', alpha=.5)
                ax[i][j].plot(self.FD, gaussian_RM, '--', label='RM synthesis signal')
                ax[i][j].set_xlabel(r'$\phi$ [rad m$^{-2}$]')
                ax[i][j].set_ylabel('Amplitude')

        plt.legend()
        plt.savefig('Gaussians.png')

    def plot_tophats(self):
        # randomly select 2 rows and 3 columns from the samples classified as tophat
        randint = np.random.randint(0, sum(self.types == 2), (2, 3))

        # create a figure with two rows and three columns
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                # get the amplitude, width, and location parameters for the selected tophat source
                amp, width, loc = self.outputs[1].T[:, randint[i, j]]

                # generate the reconstructed tophat signal using the obtained parameters
                tophat = self.tophat_source(amp, width, loc)

                # get the true tophat signal from the data
                tophat_true = self.true_spectra[1][randint[i, j]]

                # get the RM synthesis signal for the tophat source
                tophat_RM = self.RM_spectra[1][randint[i, j]]

                # normalize the RM synthesis signal and true signal to have the same maximum amplitude
                tophat_RM = tophat_RM / torch.max(tophat_RM) * torch.max(tophat_true)

                # plot the reconstructed, true, and RM synthesis signals
                ax[i][j].plot(self.FD, tophat.T, label='Reconstructed Signal')
                ax[i][j].plot(np.linspace(-100, 100, 2000), tophat_true, label='True Signal', alpha=.5)
                ax[i][j].plot(self.FD, tophat_RM, '--', label='RM synthesis signal')
                ax[i][j].set_xlabel(r'$\phi$ [rad m$^{-2}$]')
                ax[i][j].set_ylabel('Amplitude')
        plt.legend()

        plt.savefig('Tophats.png')


if __name__ == "__main__":
    tester = Test_Pipeline()

    tester.test_classifier()

    tester.test_parameter()

    tester.create_catalog()

    tester.plot_results()

    tester.plot_gaussians()

    tester.plot_tophats()

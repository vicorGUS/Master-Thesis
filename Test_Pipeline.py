import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from astropy.io import fits
from torch.utils.data import DataLoader
from Network import InceptionCNN
from Preprocess import Preprocess
from Helper import CustomTensorDataset
from sklearn import metrics


def bin_medians(x, y, num_bins):
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
    medians = np.array([torch.median(y[bin_indices == i]) for i in range(1, num_bins + 1)])
    return medians


class Test_Pipeline:
    def __init__(self):
        # Create a frequency domain range
        self.num_FD = 200
        self.FD = torch.linspace(-100, 100, self.num_FD)

        self.preprocessing = Preprocess()
        # Load the data
        self.inputs = [np.abs(self.preprocessing.RMsynthesis('Data/QU_test_Gaussian.fits')),
                       np.abs(self.preprocessing.RMsynthesis('Data/QU_test_tophat.fits'))]
        self.F = [torch.load('Data/data_true_Gaussian.npz').reshape(-1, 1, self.num_FD),
                  torch.load('Data/data_true_tophat.npz').reshape(-1, 1, self.num_FD)]
        self.labels = torch.load('Data/outputs_test.npz')

        # Reshape the data
        self.inputsize = self.inputs[0].size()
        self.num_samples = np.prod(self.inputsize[:2])
        self.types_true = self.labels[0].view(-1, self.num_samples)
        self.params_true = self.labels[1].view(-1, 3)
        self.SNR = self.labels[2].reshape(-1, self.num_samples)

        # Initialize variables
        self.model = None
        self.types_pred = []
        self.params_pred_types = None
        self.params_true_types = None
        self.types_dict = None
        self.outputs = None
        self.diffs_types = None
        self.true_spectra = None
        self.RM_spectra = None

    def load_weights(self, test_type, signal_type=None):
        # Load the model and weights based on the test type and signal type
        if test_type == 'classifying':
            self.model = InceptionCNN(n_outputs=2).float()
            if signal_type == 'Gaussian':
                self.model.load_state_dict(torch.load('Model_params/Classifying_model_params_Gaussian.pth',
                                                      map_location=torch.device('cpu')))
            elif signal_type == 'tophat':
                self.model.load_state_dict(
                    torch.load('Model_params/Classifying_model_params_tophat.pth', map_location=torch.device('cpu')))

        elif test_type == 'parameterizing':
            self.model = InceptionCNN(n_outputs=3).float()
            if signal_type == 'Gaussian':
                self.model.load_state_dict(torch.load('Model_params/Parameterizing_model_params_Gaussian.pth',
                                                      map_location=torch.device('cpu')))
            elif signal_type == 'tophat':
                self.model.load_state_dict(
                    torch.load('Model_params/Parameterizing_model_params_tophat.pth', map_location=torch.device('cpu')))

    def test_classifier(self):

        types = ['Gaussian', 'tophat']

        for j, name in enumerate(types):
            # Load the classifier model

            self.load_weights(test_type='classifying', signal_type=name)

            # Create a data loader for testing
            test_loader = self.preprocessing.create_loaders(self.inputs[j].view(-1, 1, self.num_FD), 10, train=False)

            # Initialize the outputs tensor
            outputs = torch.empty(self.num_samples.item())
            outputs[:] = torch.nan

            # Iterate over the data loader and classify the signals
            with torch.no_grad():
                for i, x in enumerate(test_loader, 1):
                    start = torch.where(torch.isnan(outputs))[0][0]
                    output = self.model.forward(x.float())
                    outputs[start: start + len(output)] = torch.argmax(output.detach(), dim=1)

            # Save the classified signal types
            self.types_pred.append(torch.LongTensor(np.array(outputs)))

    def test_parameter(self):
        # create boolean masks for Gaussian and tophat sources
        Gaussian = (self.types_pred[0] == 1)
        tophat = (self.types_pred[1] == 1)

        # create a dictionary mapping signal types to boolean masks
        self.types_dict = {'Gaussian': Gaussian, 'tophat': tophat}

        # initialize lists to store model outputs, true labels, and differences
        outputs = [torch.empty(sum(Gaussian), 3), torch.empty(sum(tophat), 3)]
        labels = [torch.empty(sum(Gaussian), 3), torch.empty(sum(tophat), 3)]
        diffs = [torch.empty(sum(Gaussian), 3), torch.empty(sum(tophat), 3)]

        # initialize tensors to store true spectra and RM synthesis spectra for each signal type
        self.true_spectra = [torch.empty(sum(Gaussian), self.num_FD), torch.empty(sum(tophat), self.num_FD)]
        self.RM_spectra = [torch.empty(sum(Gaussian), self.num_FD), torch.empty(sum(tophat), self.num_FD)]

        # initialize model outputs as NaNs
        outputs[0][:] = outputs[1][:] = torch.nan

        # loop over signal types and train corresponding models
        for (j, (name, t)) in enumerate(self.types_dict.items()):

            self.load_weights(test_type='parameterizing', signal_type=name)

            inputs = np.abs(self.inputs[j]).view(-1, 1, self.num_FD)[t]

            test_loader = self.preprocessing.create_loaders(inputs, 10, train=False)
            with torch.no_grad():
                for i, x in enumerate(test_loader, 1):
                    # get the first NaN index in outputs[j] to continue appending the model outputs
                    start = torch.where(torch.isnan(outputs[j]))[0][0]
                    output = self.model.forward(x.float())

                    outputs[j][start: start + len(output)] = output

            labels[j] = self.params_true[t]
            self.true_spectra[j], self.RM_spectra[j] = self.F[j][t], inputs
            diffs[j] = outputs[j] - labels[j]

        # store results as instance variables
        self.params_pred_types = outputs
        self.params_true_types = labels
        self.diffs_types = diffs

    def create_catalog(self):
        """
        Create a catalog based on the test results.
        The catalog will be saved in a csv file named 'Catalog.csv'.
        """
        amp_Gaussian, width_Gaussian, RM_Gaussian = self.params_pred_types[0].T
        amp_tophat, width_tophat, RM_tophat = self.params_pred_types[1].T

        x_Gaussian, y_Gaussian = np.where(self.types_pred.reshape(self.inputsize[:2]) == 1)
        x_tophat, y_tophat = np.where(self.types_pred.reshape(self.inputsize[:2]) == 2)

        dict_Gaussian = {'x [px]': x_Gaussian, 'y [px]': y_Gaussian, 'RM [rad m^(-2)]': RM_Gaussian,
                         '|F| [au]': amp_Gaussian, 'width [rad m^(-2)]': width_Gaussian,
                         'type': ['Gaussian'] * len(x_Gaussian)}

        dict_tophat = {'x [px]': x_tophat, 'y [px]': y_tophat, 'RM [rad m^(-2)]': RM_tophat,
                       '|F| [au]': amp_tophat, 'width [rad m^(-2)]': width_tophat, 'type': ['tophat'] * len(x_tophat)}

        df_Gaussian = pd.DataFrame(dict_Gaussian)
        df_tophat = pd.DataFrame(dict_tophat)

        df_combined = pd.concat([df_Gaussian, df_tophat], ignore_index=True)
        df_combined_sorted = df_combined.sort_values(['x [px]', 'y [px]']).reset_index(drop=True)

        df_combined_sorted.to_csv('Results/Catalog.csv', index=False, sep=',', float_format='%.2f')

    def create_Fmap(self):
        F_map = np.zeros((2, self.inputsize[0], self.inputsize[1]))

        amp_Gaussian, _, _ = self.params_pred_types[0].T
        amp_tophat, _, _ = self.params_pred_types[1].T

        x_Gaussian, y_Gaussian = np.where(self.types_pred[0].reshape(self.inputsize[:2]) == 1)
        x_tophat, y_tophat = np.where(self.types_pred[1].reshape(self.inputsize[:2]) == 1)

        F_map[0, y_Gaussian, x_Gaussian] = amp_Gaussian
        F_map[1, y_tophat, x_tophat] = amp_tophat

        im = plt.imshow(F_map[0], origin='lower')
        plt.title('Gaussian sources')
        plt.xlabel('x [px]')
        plt.ylabel('y [px]')

        cbar = plt.colorbar(im)
        cbar.set_label('|F| [au]')

        plt.savefig('Results/Fmap_Gaussian.png')
        plt.close()

        im = plt.imshow(F_map[1], origin='lower')
        plt.title('top-hat sources')
        plt.xlabel('x [px]')
        plt.ylabel('y [px]')

        cbar = plt.colorbar(im)
        cbar.set_label('|F| [au]')

        plt.savefig('Results/Fmap_tophat.png')
        plt.close()

    def plot_class(self):
        # Define the labels to use in the confusion matrix
        labels = [0, 1]

        for i, name in enumerate(self.types_dict):

            # Plot the confusion matrix for all data
            confusion_matrix = metrics.confusion_matrix(self.types_true[i], self.types_pred[i], normalize='true', labels=labels)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
            cm_display.plot()
            plt.savefig('Results/Confusion_Matrix_all.png')

            # Define the conditions for each subset of data
            conditions_dict = {
                'amp_0_5': ((self.params_true[:, 0] > 0) & (self.params_true[:, 0] < 5)),
                'amp_5_10': ((self.params_true[:, 0] > 5) & (self.params_true[:, 0] < 10)),
                'width_1_2': ((self.params_true[:, 1] > 1) & (self.params_true[:, 1] < 2.5)),
                'width_2_5': ((self.params_true[:, 1] > 2.5) & (self.params_true[:, 1] < 5)),
                'snr_5_10': ((self.SNR[i] > 5) & (self.SNR[i] < 10)),
                'snr_30_50': ((self.SNR[i] > 30) & (self.SNR[i] < 50))
            }

            # Plot the confusion matrix for each subset of data and save each plot separately
            for cond, condition in conditions_dict.items():
                # Compute the confusion matrix for the current subset of data
                cm = metrics.confusion_matrix(self.types_true[i][condition], self.types_pred[i][condition], normalize='true',
                                              labels=labels)

                # Plot the confusion matrix
                metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()

                # Save the plot with the corresponding name
                plt.savefig(f'Results/Confusion_Matrix_{cond}_{name}.png')
                plt.close()

    def plot_histograms(self):
        params = ['Amplitude', 'Width', 'Location']

        for (j, (name, t)) in enumerate(self.types_dict.items()):

            fig, ax = plt.subplots(ncols=3, figsize=(20, 10))

            fig.suptitle(f'Histograms of {name} parameter errors', fontsize=20)

            for i, param in enumerate(params):
                # Plot histogram of parameter errors
                ax[i].hist(self.diffs_types[j][:, i], bins=30)
                ax[i].set_xlabel(f'$\Delta$ {params[i]}', fontsize=15)
                ax[i].set_ylabel('counts', fontsize=15)

            plt.savefig(f'Results/Error_Histograms_{name}.png')
            plt.close()

    def plot_relative_error(self):
        params = ['Amplitude', 'Width', 'Location']

        for (j, (name, t)) in enumerate(self.types_dict.items()):

            fig, ax = plt.subplots(ncols=3, figsize=(20, 10))

            fig.suptitle(f'Relative {name} parameter errors', fontsize=20)

            for i, param in enumerate(params):
                # Plot SNR vs parameter errors
                ax[i].grid()

                SNRs = np.linspace(0, max(self.SNR[j][t]), 10)

                ax[i].set_xlim((5, max(self.SNR[j][t])))
                ax[i].set_ylim((-3, 3))

                ax[i].scatter(self.SNR[j][t],
                              self.diffs_types[j][:, i] / self.params_true_types[j][:, 0], marker='+', c='k',
                              s=5, alpha=.5)

                ax[i].plot(SNRs, bin_medians(x=self.SNR[j][t],
                                             y=self.diffs_types[j][:, i] / self.params_true_types[j][:, 0], 
                                             num_bins=len(SNRs)), '--', lw=2, c='r')

                ax[i].set_xlabel('SNR', fontsize=15)
                ax[i].set_ylabel(f'$\Delta$ {params[i]} / {params[i]}', fontsize=15)

            plt.savefig(f'Results/Error_Relative_{name}.png')
            plt.close()

    def plot_true_vs_pred(self):

        params = ['Amplitude', 'Width', 'Location']

        for (j, (name, t)) in enumerate(self.types_dict.items()):

            fig, ax = plt.subplots(ncols=3, figsize=(20, 10))

            fig.suptitle(f'True vs predicted {name} parameters', fontsize=20)

            for i, param in enumerate(params):
                x = np.linspace(torch.min(self.params_pred_types[j][:, i]), torch.max(self.params_pred_types[j][:, i]))

                # Plot scatter plot of true vs predicted parameter values

                ax[i].scatter(self.params_true_types[j][:, i], self.params_pred_types[j][:, i], marker='+', c='k', s=5)
                ax[i].plot(x, x, '--', lw=2, c='r')

                ax[i].set_xlabel(f'True {params[i]}', fontsize=15)
                ax[i].set_ylabel(f'Predicted {params[i]}', fontsize=15)
                ax[i].set_xlim((np.min(x), np.max(x)))

            plt.savefig(f'Results/True_vs_pred_{name}.png')
            plt.close()

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

        width /= 2.35

        return amplitude * np.exp(-1 / (2 * width ** 2) * (self.FD.reshape(1, -1) - location.reshape(-1, 1)) ** 2)

    def tophat_source(self, amplitude, width, location):
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

        return np.where(condition, amplitude, 0)

    def plot_gaussians(self):
        # randomly select 2 rows and 3 columns from the samples classified as Gaussian
        randint = np.random.randint(0, sum(self.types_pred[0] == 1), (2, 3))

        # create a figure with two rows and three columns
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                # get the amplitude, sigma, and location parameters for the selected Gaussian source
                amp, sigma, loc = self.params_pred_types[0].T[:, randint[i, j]]

                # generate the reconstructed Gaussian signal using the obtained parameters
                gaussian = self.gaussian_source(amp, sigma, loc)

                # get the true Gaussian signal from the data
                gaussian_true = self.true_spectra[0][randint[i, j]]

                # get the RM synthesis signal for the Gaussian source
                gaussian_RM = self.RM_spectra[0][randint[i, j]]

                # normalize the RM synthesis signal and true signal to have the same maximum amplitude
                # gaussian_RM = gaussian_RM / torch.max(gaussian_RM) * np.max(gaussian_true)

                # plot the reconstructed, true, and RM synthesis signals
                ax[i][j].plot(self.FD, gaussian.T, label='Reconstructed Signal')
                ax[i][j].plot(self.FD, gaussian_true[0], label='True Signal', alpha=.5)
                # ax[i][j].plot(self.FD, gaussian_RM[0], '--', label='RM synthesis signal')
                ax[i][j].set_xlabel(r'$\phi$ [rad m$^{-2}$]')
                ax[i][j].set_ylabel('|F| [au]')

        plt.legend()
        plt.savefig('Results/Gaussians.png')
        plt.close()

    def plot_tophats(self):
        # randomly select 2 rows and 3 columns from the samples classified as tophat
        randint = np.random.randint(0, sum(self.types_pred[1] == 1), (2, 3))

        # create a figure with two rows and three columns
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                # get the amplitude, width, and location parameters for the selected tophat source
                amp, width, loc = self.params_pred_types[1].T[:, randint[i, j]]

                # generate the reconstructed tophat signal using the obtained parameters
                tophat = self.tophat_source(amp, width, loc)

                # get the true tophat signal from the data
                tophat_true = self.true_spectra[1][randint[i, j]]

                # get the RM synthesis signal for the tophat source
                tophat_RM = self.RM_spectra[1][randint[i, j]]

                # normalize the RM synthesis signal and true signal to have the same maximum amplitude
                # tophat_RM = tophat_RM / torch.max(tophat_RM) * np.max(tophat_true)

                # plot the reconstructed, true, and RM synthesis signals
                ax[i][j].plot(self.FD, tophat.T, label='Reconstructed Signal')
                ax[i][j].plot(self.FD, tophat_true[0], label='True Signal', alpha=.5)
                # ax[i][j].plot(self.FD, tophat_RM[0], '--', label='RM synthesis signal')
                ax[i][j].set_xlabel(r'$\phi$ [rad m$^{-2}$]')
                ax[i][j].set_ylabel('|F| [au]')
        plt.legend()

        plt.savefig('Results/Tophats.png')
        plt.close()


if __name__ == "__main__":
    tester = Test_Pipeline()

    tester.test_classifier()

    tester.test_parameter()

    # tester.create_catalog()

    tester.create_Fmap()

    tester.plot_class()

    tester.plot_histograms()

    tester.plot_relative_error()

    tester.plot_true_vs_pred()

    tester.plot_gaussians()

    tester.plot_tophats()

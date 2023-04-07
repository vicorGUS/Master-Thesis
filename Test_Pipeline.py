import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Network import ClassifyingCNN, ParameterizingCNN
from Create_Dataset import create_loaders
from Helper import CustomTensorDataset
import torch.nn.functional as F


def bin_means(x, y, num_bins):
    bin_boundaries = np.linspace(torch.min(x).item(), torch.max(x).item(), num_bins + 1, endpoint=True)

    bin_indices = np.digitize(x, bin_boundaries)
    means = np.array([torch.mean(y[bin_indices == i]) for i in range(1, num_bins + 1)])

    return means


class Test_Pipeline:

    def __init__(self):
        self.data = torch.load('Data/data_test.npz')
        self.model = None
        self.num_samples = len(self.data[0])

    def load_weights(self, test_type, signal_type=None):
        if test_type == 'classifying':
            self.model = ClassifyingCNN().float()
            self.model.load_state_dict(torch.load('Classifying_model_params.pth'))
        elif test_type == 'parameterizing':
            self.model = ParameterizingCNN().float()
            if signal_type == 'Gaussian':
                self.model.load_state_dict(torch.load('Parameterizing_model_params_Gaussian.pth'))
            elif signal_type == 'tophat':
                self.model.load_state_dict(torch.load('Parameterizing_model_params_tophat.pth'))

    def test_classifier(self):
        self.load_weights(test_type='classifying')
        test_loader = create_loaders(self.data[0], 100, train=False)

        outputs = torch.empty(self.num_samples)
        outputs[:] = torch.nan

        with torch.no_grad():
            for i, x in enumerate(test_loader, 1):
                start = torch.where(torch.isnan(outputs))[0][0]
                output = self.model.forward(x.float())
                outputs[start: start + len(output)] = torch.argmax(output.detach(), dim=1)

        labels = self.data[2]
        diffs = outputs - labels

        return outputs, labels, diffs

    def test_parameter(self):
        types, _, _ = self.test_classifier()

        Gaussian = (types == 1)
        tophat = (types == 2)

        types_dict = {'Gaussian': Gaussian, 'tophat': tophat}

        outputs = [torch.empty(sum(Gaussian), 2), torch.empty(sum(tophat), 2)]
        labels = [torch.empty(sum(Gaussian), 2), torch.empty(sum(tophat), 2)]
        diffs = [torch.empty(sum(Gaussian), 2), torch.empty(sum(tophat), 2)]

        outputs[0][:] = outputs[1][:] = torch.nan
        labels[0][:] = labels[1][:] = torch.nan
        diffs[0][:] = diffs[1][:] = torch.nan

        for (j, (name, t)) in enumerate(types_dict.items()):
            self.load_weights(test_type='parameterizing', signal_type=name)
            test_loader = create_loaders(self.data[0][t], 100, train=False)
            with torch.no_grad():
                for i, x in enumerate(test_loader, 1):
                    start = torch.where(torch.isnan(outputs[j]))[0][0]
                    output = self.model.forward(x.float())
                    outputs[j][start: start + len(output)] = output
            labels[j] = self.data[1][t]
            diffs[j] = outputs[j] - labels[j]

        return outputs, labels, diffs, types_dict

    def plot_results(self):

        outputs, labels, diffs, types_dict = self.test_parameter()

        results_Gaussian = np.hstack([labels[0], outputs[0], diffs[0]])
        results_tophat = np.hstack([labels[1], outputs[1], diffs[1]])

        np.savetxt('results_Gaussian.txt', results_Gaussian, delimiter=',')
        np.savetxt('results_tophat.txt', results_tophat, delimiter=',')

        params = [r'Amplitude', r'$\sigma_{\phi}$']

        for (j, (name, t)) in enumerate(types_dict.items()):
            for i in range(len(params)):
                fig, ax = plt.subplots(ncols=3, figsize=(20, 10))
                x = np.linspace(torch.min(labels[j][:, i]).item(), torch.max(labels[j][:, i]).item(), 2)

                ax[0].set_title(f'Histogram of {params[i]} error', fontsize=15)
                ax[0].hist(diffs[j][:, i], bins=30)
                ax[0].set_xlabel(f'$\Delta$ {params[i]}', fontsize=15)
                ax[0].set_ylabel('counts', fontsize=15)

                ax[1].set_title(f'True vs Predicted {params[i]}', fontsize=15)
                ax[1].plot(x, x, lw=2, c='r')
                ax[1].scatter(labels[j][:, i], outputs[j][:, i], marker='+', c='k', s=5)

                ax[1].set_xlabel(f'True {params[i]}', fontsize=15)
                ax[1].set_ylabel(f'Predicted {params[i]}', fontsize=15)
                ax[1].set_xlim((torch.min(self.data[1][t][:, i]), torch.max(outputs[j][:, i])))

                ax[2].set_title(f'SNR vs {params[i]} error', fontsize=15)
                ax[2].grid()

                ax[2].plot(np.arange(5, 21), bin_means(x=self.data[1][t][:, 0] / 0.65,
                                                       y=diffs[j][:, i] / labels[j][:, i], num_bins=16), lw=2, c='r')

                ax[2].scatter(self.data[1][t][:, 0] / 0.65, diffs[j][:, i] / labels[j][:, i], marker='+', c='k', s=5, alpha=.5)
                ax[2].set_xlabel('SNR', fontsize=15)
                ax[2].set_ylabel(f'$\Delta$ {params[i]} / {params[i]}', fontsize=15)

                plt.savefig(f'Experiment_results_{name}_{i}.png')


if __name__ == "__main__":

    tester = Test_Pipeline()

    tester.plot_results()

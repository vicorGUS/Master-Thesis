import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class CreateData:

    def __init__(self, l, SNR, RM, width, SNR_I, FD, FD_RM, frequency):
        # Number of spectra and frequency domain parameters
        self.l = l
        self.num_frequency = frequency[2]
        self.wl2_range = (299792458 / np.linspace(frequency[0], frequency[1], num=frequency[2])[::-1]) ** 2
        self.FWHM = 2 * np.sqrt(3) / (np.ptp(self.wl2_range))

        # Set FD domain parameters
        self.FD = np.linspace(FD[0], FD[1], num=FD[2])
        self.FD_RM = np.linspace(FD_RM[0], FD_RM[1], num=FD_RM[2])
        self.num_FD = FD[2]
        self.num_FD_RM = FD_RM[2]

        # Define parameters for real signal
        self.types = np.random.randint(0, 3, l)
        self.SNR = np.random.uniform(SNR[0], SNR[1], size=(l, 1))
        self.RM = np.random.uniform(RM[0], RM[1], size=(l, 1)) * self.FWHM
        self.width = np.random.uniform(width[0], width[1], size=(l, 1)) * self.FWHM
        self.theta0 = np.random.uniform(-np.pi / 2, np.pi / 2, size=(l, 1))

        # Define parameters for instrumental polarisation
        self.SNR_I = np.random.uniform(SNR_I[0], SNR_I[1], size=(l, 1))
        self.RM_I = np.zeros((l, 1))
        self.width_I = 1 * np.ones((l, 1))
        self.theta0_I = np.zeros((l, 1))

    def generate_noise(self):
        ### Generates Ricean noise in phi ###
        q_noise = np.random.normal(0, 1, size=(self.l, self.num_FD))
        u_noise = np.random.normal(0, 1, size=(self.l, self.num_FD))
        rice_noise = q_noise + 1j * u_noise
        return rice_noise

    def add_leakage(self):
        amp = self.SNR_I * self.sigma
        width = self.width_I
        location = self.RM_I
        phase = self.theta0_I

        condition = (location - width / 2. < self.FD) & (self.FD < location + width / 2.)
        leakage = np.where(condition, amp, 0) * np.exp(2j * phase)
        return leakage

    def add_signal(self):
        types = self.types

        Gaussian = (types == 1)
        tophat = (types == 2)

        amp = self.SNR * self.sigma
        width = self.width
        width[tophat] *= np.sqrt(2 * np.pi)
        location = self.RM
        phase = self.theta0

        signal = 1j * np.zeros((self.l, self.num_FD))

        signal[Gaussian] = amp[Gaussian] * np.exp(-1 / (2 * width[Gaussian] ** 2) * (self.FD.reshape(1, -1) -
                                                                                     location[Gaussian].reshape(-1,
                                                                                                                1)) ** 2) * np.exp(
            2j * phase[Gaussian])

        condition = (location[tophat] - width[tophat] / 2. < self.FD) & (
                self.FD < location[tophat] + width[tophat] / 2.)

        signal[tophat] = np.where(condition, amp[tophat], 0) * np.exp(2j * phase[tophat])
        return signal

    def RMsynthesis(self, P):
        ### Performs Rotation Measure synthesis on complex P ###
        wl2_0 = np.mean(self.wl2_range)
        fd_function = 1j * np.zeros((self.l, self.num_FD_RM))
        for j, p in enumerate(self.FD_RM):
            M = np.exp(-2j * p * (self.wl2_range - wl2_0))
            fd_function[:, j] = np.dot(P, M)
        return 1 / self.num_frequency * fd_function

    def F_and_P(self):
        ### Generates signal with noise, adds instrumental polarization and returns true F and RM-synthesis F ###
        rice_noise = self.generate_noise()
        self.sigma = np.std(np.abs(rice_noise), axis=1).reshape(-1, 1)
        self.offset = np.mean(np.abs(rice_noise), axis=1).reshape(-1, 1)

        F = rice_noise + self.add_leakage() + self.add_signal()
        P = 1j * np.zeros((self.l, self.num_frequency))
        for j, w in enumerate(self.wl2_range):
            M = np.exp(2j * self.FD * w)
            P[:, j] = np.dot(F, M)
        Ftilde = self.RMsynthesis(P)
        return F, Ftilde

    def save_data(self, test=False):
        ### Saves input and output to network, and true signal if test=True ###
        data_true, data_RM = self.F_and_P()

        data_true, data_RM = torch.tensor(data_true), torch.tensor(data_RM)

        amplitudes = torch.tensor(self.SNR * self.sigma)
        width = torch.tensor(self.width)

        dataset = [torch.abs(data_RM).unsqueeze(dim=1),
                   torch.cat((amplitudes, width), dim=1),
                   torch.LongTensor(self.types)]

        if test:
            torch.save(dataset, 'Data/data_test.npz')
        else:
            torch.save(dataset, 'Data/data_train.npz')


CreateData(l=10000, SNR=[5, 20], RM=[-10, 10], width=[.2, 1.], SNR_I=[5, 12], FD=[-100, 100, 2000],
           FD_RM=[-100, 100, 200], frequency=[350e6, 1050e6, 700]).save_data(test=True)

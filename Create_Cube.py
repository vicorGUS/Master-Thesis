import numpy as np
import torch
import random

class CreateCube:

    def __init__(self, l, amp, RM, width, FD, FD_RM, frequency):
        self.l = l
        self.num_frequency = frequency[2]
        self.wl2_range = (299792458 / np.linspace(frequency[0], frequency[1], num=frequency[2])[::-1]) ** 2
        self.FWHM = 2 * np.sqrt(3) / (np.ptp(self.wl2_range))

        self.FD = np.linspace(FD[0], FD[1], num=FD[2])
        self.FD_RM = np.linspace(FD_RM[0], FD_RM[1], num=FD_RM[2])
        self.num_FD = FD[2]
        self.num_FD_RM = FD_RM[2]

        self.types = np.reshape(random.choices([0, 1, 2], [1., 1., 1.], k=l**2), (l,l))
        self.amp = np.random.uniform(amp[0], amp[1], size=(l, l))
        self.RM = np.random.uniform(RM[0], RM[1], size=(l, l)) * self.FWHM
        self.width = np.random.uniform(width[0], width[1], size=(l, l)) * self.FWHM
        self.theta0 = np.random.uniform(-np.pi / 2, np.pi / 2, size=(l, l))

        self.amp_I = np.random.uniform(1, 3, size=(l, l)) * self.amp
        self.RM_I = np.zeros((l, l))
        self.width_I = np.ones((l, l))
        self.theta0_I = np.zeros((l, l))

    def generate_noise(self):
        q_noise = np.random.normal(0, 1, size=(self.l, self.l, self.num_frequency))
        u_noise = np.random.normal(0, 1, size=(self.l, self.l, self.num_frequency))
        rice_noise = q_noise + 1j * u_noise
        return rice_noise

    def add_leakage(self):
        return self.amp_I[..., np.newaxis] * np.exp(
            -1 / (2 * self.width_I[..., np.newaxis] ** 2) * (
                        self.FD.reshape(1, 1, -1) - self.RM_I[..., np.newaxis]) ** 2
        ) * np.exp(2j * self.theta0_I[..., np.newaxis])

    def add_signal(self):
        nosignal = (self.types == 0)
        Gaussian = (self.types == 1)
        tophat = (self.types == 2)

        self.amp[nosignal] = 0.
        self.width[nosignal] = 0.

        amp = self.amp[..., np.newaxis]
        self.width[tophat] *= np.sqrt(2 * np.pi)
        width = self.width[..., np.newaxis]
        location = self.RM[..., np.newaxis]
        phase = self.theta0[..., np.newaxis]

        signal = np.zeros((self.l, self.l, self.num_FD), dtype=complex)

        for row, col in zip(*np.where(Gaussian)):
            signal[row, col] = amp[row, col] * np.exp(
                -1 / (2 * width[row, col] ** 2) * (self.FD - location[row, col][..., np.newaxis]) ** 2
            ) * np.exp(2j * phase[row, col])

        """signal[Gaussian] = amp[Gaussian] * np.exp(
            -1 / (2 * width[Gaussian] ** 2) * (self.FD - location[Gaussian][..., np.newaxis]) ** 2
        ) * np.exp(2j * phase[Gaussian])"""

        for row, col in zip(*np.where(tophat)):
            condition = (location[row, col] - width[row, col] / 2. < self.FD) & (
                    self.FD < location[row, col] + width[row, col] / 2.)
            signal[row, col] = np.where(condition, amp[row, col], 0) * np.exp(2j * phase[row, col])

        return signal

    def RMsynthesis(self, P):
        wl2_0 = np.mean(self.wl2_range)
        M = np.exp(-2j * self.FD_RM[:, np.newaxis] * (self.wl2_range - wl2_0)).reshape(self.num_FD_RM, self.num_frequency, 1)
        return 1 / self.num_frequency * np.einsum('ijk,lkm->ijl', P, M)

    def F_and_P(self):
        F = self.add_signal() + self.add_leakage()
        M = np.exp(2j * self.FD[:, np.newaxis] * self.wl2_range).reshape(self.num_FD, self.num_frequency, 1)
        P = np.einsum('ijk,klm->ijl', F, M)
        P += self.generate_noise() * 50

        Ftilde = self.RMsynthesis(P)
        return F, Ftilde

    def save_data(self, test=False):
        data_true, data_RM = self.F_and_P()

        data_true, data_RM = torch.tensor(data_true), torch.tensor(data_RM)

        amplitude = torch.tensor(self.amp)
        width = torch.tensor(self.width)
        location = torch.tensor(self.RM)

        dataset = [torch.abs(data_RM).unsqueeze(dim=2),
                   torch.stack((amplitude, width, location), dim=2),
                   torch.LongTensor(self.types)]

        dataset_RM = [torch.abs(data_true), torch.abs(data_RM)]

        if test:
            torch.save(dataset_RM, 'Data/data_RM.npz')
            torch.save(dataset, 'Data/data_test.npz')
        else:
            torch.save(dataset, 'Data/data_train.npz')


CreateCube(l=100, amp=[0.1, 15], RM=[-10, 10], width=[.1, 1.], FD=[-100, 100, 2000],
           FD_RM=[-100, 100, 200], frequency=[350e6, 1050e6, 700]).save_data(test=True)

import numpy as np
import torch
import matplotlib.pyplot as plt
from astropy.io import fits


class CreateCube:

    def __init__(self, l, amp, RM, width, FD, frequency):
        self.P = None
        self.F = None
        self.amp_I = None
        self.SNR = np.zeros((2, l, l))
        self.l = l
        self.num_frequency = frequency[2]
        self.wl2_range = (299792458 / np.linspace(frequency[0], frequency[1], num=frequency[2])[::-1]) ** 2
        self.FWHM = 2 * np.sqrt(3) / (np.ptp(self.wl2_range))

        self.FD = np.linspace(FD[0], FD[1], num=FD[2])
        self.num_FD = FD[2]

        types = np.random.choice([0, 1], size=(l, l), p=[.5, .5])
        self.types = np.stack([types, types])
        self.amp = np.random.uniform(amp[0], amp[1], size=(l, l))
        self.RM = np.random.uniform(RM[0], RM[1], size=(l, l)) * self.FWHM
        self.width = np.random.uniform(width[0], width[1], size=(l, l)) * self.FWHM
        self.theta0 = np.random.uniform(-np.pi / 2, np.pi / 2, size=(l, l))

        self.RM_I = np.zeros((l, l))
        self.width_I = np.ones((l, l))
        self.theta0_I = np.zeros((l, l))

    def generate_noise(self):
        q_noise = np.random.normal(0, np.sqrt(self.num_frequency), size=(2, self.l, self.l, self.num_frequency))
        u_noise = np.random.normal(0, np.sqrt(self.num_frequency), size=(2, self.l, self.l, self.num_frequency))
        rice_noise = q_noise + 1j * u_noise
        return rice_noise

    def add_leakage(self):
        self.amp_I = np.random.uniform(1, 3, size=(self.l, self.l)) * self.amp
        return self.amp_I[..., np.newaxis] * np.exp(
            -1 / (2 * self.width_I[..., np.newaxis] ** 2) * (
                    self.FD.reshape(1, 1, -1) - self.RM_I[..., np.newaxis]) ** 2
        ) * np.exp(2j * self.theta0_I[..., np.newaxis])

    def add_Gaussian(self):
        self.amp[self.types[0] == 0] = 0
        self.width[self.types[0] == 0] = 0
        self.RM[self.types[0] == 0] = np.random.uniform(-50, 50)

        # Get boolean masks for signals
        source = self.types[0] == 1

        # Reshape amp and width arrays
        amp = self.amp[..., np.newaxis]
        width = np.copy(self.width[..., np.newaxis])

        # Adjust the width for tophat signals
        width /= 2.35

        # Reshape location and phase arrays
        location = self.RM[..., np.newaxis]
        phase = self.theta0[..., np.newaxis]

        # Add Gaussian signals to the signal array
        self.F[0][source] = (amp[source] * np.exp(-0.5 * ((self.FD - location[source]) / width[source]) ** 2) *
                             np.exp(2j * phase[source]))

        self.SNR[0] = np.trapz(np.abs(self.F[0]), self.FD)

        no_signal = (self.SNR[0] < 5)

        self.types[0][no_signal] = 0

    def add_tophat(self):
        self.amp[self.types[1] == 0] = 0
        self.width[self.types[1] == 0] = 0
        self.RM[self.types[1] == 0] = np.random.uniform(-50, 50)

        source = self.types[1] == 1

        # Reshape amp and width arrays
        amp = self.amp[..., np.newaxis]
        width = self.width[..., np.newaxis]

        # Reshape location and phase arrays
        location = self.RM[..., np.newaxis]
        phase = self.theta0[..., np.newaxis]

        # Add tophat signals to the signal array
        condition = np.logical_and(location[source] - width[source] / 2. < self.FD,
                                   self.FD < location[source] + width[source] / 2.)

        self.F[1][source] = np.where(condition, amp[source] * np.exp(2j * phase[source]), 0)

        self.SNR[1] = np.trapz(np.abs(self.F[1]), self.FD)

        no_signal = (self.SNR[1] < 5)

        self.types[1][no_signal] = 0

    def F_and_P(self):
        self.F = np.zeros((2, self.l, self.l, self.num_FD), dtype=complex)
        self.add_Gaussian()
        self.add_tophat()
        self.F += self.add_leakage()
        M = np.exp(2j * self.FD[:, np.newaxis] * self.wl2_range).reshape(self.num_FD, self.num_frequency)
        self.P = np.einsum('ijkl,lm->ijkm', self.F, M)
        """for i in range(20):
            plt.plot(self.FD, np.real(self.F[1][i, 0]), label='real')
            plt.plot(self.FD, np.imag(self.F[1][i, 0]), label='imag')
            plt.plot(self.FD, np.abs(self.F[1][i, 0]), c='k', label='abs')
            plt.xlabel('$\phi$ [rad m$^{-2}$]')
            plt.ylabel('Amplitude [au]')
            plt.legend()
            #plt.savefig('QUF.png', transparent=True)
            plt.show()"""
        """plt.plot(self.wl2_range, np.real(self.P[1][0, 0]), label='Q')
        plt.plot(self.wl2_range, np.imag(self.P[1][0, 0]), label='U')
        plt.plot(self.wl2_range, np.abs(self.P[1][0, 0]), c='k', label='|P|')
        plt.xlabel('$\lambda^2$ [m$^2$]')
        plt.ylabel('Amplitude [au]')
        plt.legend()
        #plt.savefig('W.png', transparent=True)
        plt.show()"""
        self.P += self.generate_noise()
        """plt.plot(self.wl2_range, np.real(self.P[1][0, 0]), label='Q')
        plt.plot(self.wl2_range, np.imag(self.P[1][0, 0]), label='U')
        plt.plot(self.wl2_range, np.abs(self.P[1][0, 0]), c='k', label='|P|')
        plt.xlabel('$\lambda^2$ [m$^2$]')
        plt.ylabel('Amplitude [au]')
        plt.legend()
        #plt.savefig('QUP_noise.png', transparent=True)
        plt.show()"""

    def write_fits(self, P):
        hdu = fits.PrimaryHDU()

        hdu.data = np.array([np.real(P), np.imag(P)], dtype='float32').transpose(3, 0, 1, 2)

        hdu.header['CRPIX1'] = f'{int(self.l - 1) / 2.} / Pixel coordinate of reference point'
        hdu.header['CRPIX2'] = f'{int(self.l - 1) / 2.} / Pixel coordinate of reference point'
        hdu.header['CRPIX3'] = '1.0 / Pixel coordinate of reference point'
        hdu.header['CRPIX4'] = '1.0 / Pixel coordinate of reference point'
        hdu.header['CDELT1'] = '1.0 / Coordinate increment at reference point'
        hdu.header['CDELT2'] = '1.0 / Coordinate increment at reference point'
        hdu.header['CDELT3'] = '1.0 / Coordinate increment at reference point'
        hdu.header['CDELT4'] = f'{(1050e6 - 350e6) / self.num_frequency} / [Hz] Coordinate increment at reference point'
        hdu.header['CRVAL1'] = f'{int(self.l - 1) / 2.}  / Coordinate value at reference point'
        hdu.header['CRVAL2'] = f'{int(self.l - 1) / 2.}  / Coordinate increment at reference point'
        hdu.header['CRVAL3'] = '2.0 / Coordinate increment at reference point'
        hdu.header['CRVAL4'] = '350000000 / [Hz] Coordinate increment at reference point'
        return hdu

    def save_data(self, test=False):
        self.F_and_P()

        hdu_Gaussian = self.write_fits(self.P[0])
        hdu_tophat = self.write_fits(self.P[1])

        amplitude = torch.tensor(self.amp)
        width = torch.tensor(self.width)
        location = torch.tensor(self.RM)

        outputs = [torch.LongTensor(self.types),
                   torch.stack((amplitude, width, location), dim=2),
                   torch.tensor(self.SNR)]
        if test:
            hdu_Gaussian.writeto('Data/QU_test_Gaussian.fits', overwrite=True)
            hdu_tophat.writeto('Data/QU_test_tophat.fits', overwrite=True)
            torch.save(np.abs(self.F[0]), 'Data/data_true_Gaussian.npz')
            torch.save(np.abs(self.F[1]), 'Data/data_true_tophat.npz')
            torch.save(outputs, 'Data/outputs_test.npz')
        else:
            hdu_Gaussian.writeto('Data/QU_train_Gaussian.fits', overwrite=True)
            hdu_tophat.writeto('Data/QU_train_tophat.fits', overwrite=True)
            torch.save(outputs, 'Data/outputs_train.npz')


CreateCube(l=20, amp=[0, 20], RM=[-10, 10], width=[.2, 2.], FD=[-100, 100, 200],
           frequency=[350e6, 1050e6, 700]).save_data(test=True)
import numpy as np
from PIL import Image


class _MeanSquareField:

    def __init__(self, cfa):
        self.cfa = cfa
        self.cfa_shifted = np.empty(cfa().shape)  # allocate space, doing this outside the loop reduces computation time
        self.msf = np.zeros((cfa.n_points, 1))
        self.cos_phase = None
        self.sin_phase = None
        self.phases = None
        self.n_phaseshifts = None

    def __call__(self):
        return self.msf

    def __iter__(self):
        self._iter = -1
        return self

    def __next__(self):
        try:
            self._iter += 1
            return self.calculate(self._iter)
        except IndexError:
            raise StopIteration

    def linearly_generate_phases(self, max_phaseshifts, randomness=0):

        # number of antennas
        n_antennas = self.cfa.n_antennas

        # determine number of phase steps
        n_steps = int(np.floor(max_phaseshifts ** (1 / (n_antennas - 1))))

        # determine number of points
        n_shifts = int(n_steps ** (n_antennas - 1))

        # determine the unique phases
        unique_phases = np.linspace(0, n_steps - 1, n_steps) * 2 * np.pi / n_steps

        # allocate memory for the phases
        phases = np.zeros((n_shifts, n_antennas))

        # determine the phases per antenna
        for idx_antenna in range(n_antennas - 1):
            rep1 = int(n_steps ** (n_antennas - 2 - idx_antenna))
            rep2 = int(n_steps ** idx_antenna)
            phases[:, idx_antenna + 1] = np.tile(unique_phases, (rep1, rep2)).T.reshape(-1)

        # add a 'randomness' using a gaussian distribution
        if randomness is not 0:
            phases[:, 1:n_antennas] += unique_phases[1] * \
                                       np.random.normal(0, randomness * 0.2, (n_shifts, n_antennas - 1))

        # set attributes
        self.phases = phases
        self.cos_phase = np.cos(phases)
        self.sin_phase = np.sin(phases)
        self.n_phaseshifts = n_shifts

        return self

    def calculate(self, phase_index):

        # calculate msf for given phase_index
        self._shift_cfa(phase_index)
        self.msf = self._mean_square(self.cfa_shifted)

        # return object
        return self.msf, self.phases[phase_index]

    def _shift_cfa(self, phase_index):

        # todo: find a different solution to slicing, since that will return a copy of the ndarray (i think), this
        #  increases computation time

        # cfa : ndarray, shape [n_points, n_antenna, (x,y,z), (real,imag) ]

        # reshape cos_phase and sin_phase such that numpy knows which dimension must be multiplied element-wise
        cos_phase = self.cos_phase[phase_index].reshape(1, -1, 1)
        sin_phase = self.sin_phase[phase_index].reshape(1, -1, 1)

        # real and imag part of cfa
        cfa_real = self.cfa()[:, :, :, self.cfa.REAL]
        cfa_imag = self.cfa()[:, :, :, self.cfa.IMAG]

        # shift cfa
        self.cfa_shifted[:, :, :, self.cfa.REAL] = cfa_real * cos_phase - cfa_imag * sin_phase
        self.cfa_shifted[:, :, :, self.cfa.IMAG] = cfa_imag * cos_phase + cfa_real * sin_phase

    @staticmethod
    def _mean_square(cfa_shifted):

        # cfa : ndarray, shape [n_points, n_antenna, (x,y,z), (real,imag) ]

        return 0.5 * np.sum(np.sum(cfa_shifted, axis=1) ** 2, axis=(1, 2))

    def export_as_png(self, dst, width, height, efield2_max):

        # sanity check efield2_max, if provided
        if efield2_max is not None:
            if efield2_max < np.max(self.msf):
                raise Exception('ERROR: given efield2_max (%d) is lower than the actual max (%d)' %
                                (efield2_max, np.max(self.msf)))
        else:
            efield2_max = np.max(self.msf)

        # create image and save it
        img = Image.fromarray(np.uint8(self.msf.reshape(width, height).T * 255 / efield2_max))
        # file = dst.joinpath('efield %04d.png' % self._iter)
        img.save(dst)

        # return filename
        return dst

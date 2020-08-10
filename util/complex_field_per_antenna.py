import numpy as np
import pandas as pd
import scipy.interpolate


class _ComplexFieldPerAntenna:
    # the constants below are indices that relate to self.cfa
    REAL = 0
    IMAG = 1
    X = 0
    Y = 1
    Z = 2

    def __init__(self, src, projection):

        # set attribute: projection, must be either 2d or 3d
        if projection is '2d':
            self.projection = '2d'
        elif projection is '3d':
            self.projection = '3d'
        else:
            raise Exception('ERROR: projection must be a string containing either 2d or 3d')

        # assume that all csv files in src folder contain 'complex field per antenna' (cfa)
        files = list(src.glob('*.csv'))

        # number of antenna's
        self.n_antennas = len(files)

        # loop through files
        for idx, file in enumerate(files):

            # read file into a pandas DataFrame
            data = pd.read_csv(file, delimiter=';')

            # set attributes that only depend on the first file read
            if idx is 0:
                # set x, y, z points
                self.x = data['#x[mm]'].values
                self.y = data['y[mm]'].values
                self.z = data['z[mm]'].values

                # obtain number of x,y,z points
                self.nx = len(np.unique(self.x))
                self.ny = len(np.unique(self.y))
                self.nz = len(np.unique(self.z))

                # image width, height, depth
                self.width, self.height, self.depth = self.nx, self.ny, self.nz

                # number of data points
                self.n_points = len(data['#x[mm]'])

                # allocate memory for real and imag part of efield per antenna (cfa)
                self.cfa = np.zeros((self.n_points, self.n_antennas, 3, 2))

            # convert the data from a pandas.DataFrame to a numpy ndarray, because pandas is slower and this will be
            # needed for many calculations. A numpy ndarray is also faster than lists, dictionaries and functions.
            self.cfa[:, idx, self.X, self.REAL] = data['xRe[V/m]'].values
            self.cfa[:, idx, self.Y, self.REAL] = data['yRe[V/m]'].values
            self.cfa[:, idx, self.Z, self.REAL] = data['zRe[V/m]'].values
            self.cfa[:, idx, self.X, self.IMAG] = data['xIm[V/m]'].values
            self.cfa[:, idx, self.Y, self.IMAG] = data['yIm[V/m]'].values
            self.cfa[:, idx, self.Z, self.IMAG] = data['zIm[V/m]'].values

    def __call__(self):
        return self.cfa

    def _generate_xyz(self, nx, ny, nz):

        # generate linearly distributed xyz with min/max equal to the already existing xyz of this object
        x = np.linspace(self.x[0], self.x[-1], nx)
        y = np.linspace(self.y[0], self.y[-1], ny)
        z = np.linspace(self.z[0], self.z[-1], nz)

        # create meshgrid
        xx, yy, zz = np.meshgrid(x, y, z)

        # flatten it
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        zz = zz.reshape(-1)

        return xx, yy, zz

    def get_mm_per_px(self):
        # todo: this function does not check if the mm_per_px in height/width direction is the same
        if self.nx is 1:
            return (self.z[-1] - self.z[0]) / self.width
        elif self.ny is 1:
            return (self.x[-1] - self.x[0]) / self.width
        elif self.nz is 1:
            return (self.y[-1] - self.y[0]) / self.width
        else:
            return (self.x[-1] - self.x[0]) / self.width

    def get_plane(self):
        # todo: this function does not check if the mm_per_px in height/width direction is the same
        if self.nx is 1:
            return 'yz'
        elif self.ny is 1:
            return 'xz'
        elif self.nz is 1:
            return 'xy'
        else:
            'none'

    def downsample(self, width, height, depth):

        # set width, height, depth
        self.width, self.height, self.depth = width, height, depth

        # set nx, ny, nz according to 2d/3d projection
        if self.projection is '2d':
            # set nx, ny, nz according to plane orientation
            if self.nx is 1:
                nx, ny, nz = 1, height, width
            elif self.ny is 1:
                nx, ny, nz = width, 1, height
            elif self.nz is 1:
                nx, ny, nz = width, height, 1
            else:
                raise Exception('ERROR: projection is set to 2d, but 3d data is loaded')
        elif self.projection is '3d':
            nx, ny, nz = width, height, depth
        else:
            raise Exception('self.projection is not equal to 2d or 3d')

        # determine points for 2d/3d interpolation
        #   the interpolation function does not work properly if a dimension has only has 1 point, hence the separate
        #   declarations
        x_new, y_new, z_new = self._generate_xyz(nx, ny, nz)
        if nx is 1:
            size = (self.ny, self.nz)
            points_old = np.array([self.y, self.z]).T
            points_new = np.array([y_new, z_new]).T
        elif ny is 1:
            size = (self.nx, self.nz)
            points_old = (np.unique(self.x), np.unique(self.z))
            points_new = np.array([x_new, z_new]).T
        elif nz is 1:
            size = (self.nx, self.ny)
            points_old = np.array([self.x, self.y]).T
            points_new = np.array([x_new, y_new]).T
        else:
            size = (self.nx, self.ny, self.nz)
            points_old = np.array([self.x, self.y, self.z]).T
            points_new = np.array([x_new, y_new, z_new]).T

        # update class attributes
        self.x = x_new
        self.y = y_new
        self.z = z_new
        self.nx = len(np.unique(x_new))
        self.ny = len(np.unique(y_new))
        self.nz = len(np.unique(z_new))

        # allocate space for the new cfa
        cfa_new = np.zeros((points_new.shape[0], self.n_antennas, 3, 2))

        # loop through antennas
        for idx in range(self.n_antennas):

            # loop through x,y,z dimension
            for dimension in (self.X, self.Y, self.Z):

                # loop through real and imaginary part
                for part in (self.REAL, self.IMAG):
                    # define interpolate function
                    interpol = scipy.interpolate.RegularGridInterpolator(
                        points_old, self.cfa[:, idx, dimension, part].reshape(size))

                    # apply interpolation
                    cfa_new[:, idx, dimension, part] = interpol(points_new).reshape(-1)

        # set new cfa
        self.cfa = cfa_new

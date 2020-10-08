import json
from pathlib import Path
from time import time
from typing import List
from zipfile import ZipFile

import numpy as np
from numpy import pi

import settings as settings
from .print import Print

MAX_PROJECTS = 3200
MAX_SAMPLES_PER_PROJECT = 3200


def cst_to_dataset(partition_id: int):
    """
    Converts the data generated in CST to a zipped PyTorch dataset.
    """

    # start main timer
    timer = time()

    # create print object which logs the print messages to a log.txt file
    print_ = Print('log.txt', partition_id).log

    # create msf and sar dataset
    for dataset in ['msf', 'sar']:

        # create dataset
        zipfile = ZipFile('dataset_%s.zip' % dataset, 'w')

        # counters to keep track of input/output images
        cnt_in = 0  # counter of input imgs
        cnt_out = 0  # counter of output imgs

        # path to each project that is to be added to the dataset
        paths_project = list(Path(settings.Paths.src).iterdir())

        # get the ids of the valid projects
        ids_valid = []
        for idx_project, path_project in enumerate(paths_project):
            # remember idx if results exist
            if path_project.joinpath('e-field 11.csv').exists():
                ids_valid.append(idx_project)

        # limit the number of projects to MAX_PROJECTS
        if len(ids_valid) > MAX_PROJECTS:
            ids_valid = ids_valid[0:(MAX_PROJECTS - 1)]
        n_projects = len(ids_valid)

        # get the valid project paths
        paths_valid_project = []
        for idx in ids_valid:
            paths_valid_project.append(paths_project[idx])

        # initialize csv file object
        csv = CSV(n_projects)

        # loop through each project
        for idx_project, path_project in enumerate(paths_valid_project):

            # log
            print_('importing project (%i/%i)...' %
                  (idx_project + 1, n_projects))
            print_('\t%s ' % str(path_project))

            # add input images to dataset
            print_('\tadding input images...')
            for img in ['conductivity', 'density', 'model', 'permittivity']:
                src = path_project.joinpath('maps', img + '.png')
                dst = 'input/%s_%04i.png' % (img, cnt_in)
                zipfile.write(src, dst)
            print_('\t\t...done')

            # get the path to each output image
            paths_output = sorted(list(
                path_project.joinpath(dataset).glob('*.png')
            ))

            # read the antenna configuration file
            pth_cnf = path_project.joinpath('%s/configuration.json' % dataset)
            with open(pth_cnf, 'r') as file:
                cnf = json.load(file)

            # add each output image to the dataset
            print_('\tadding output images...')
            n_outputs = len(list(paths_output))
            pct = 0
            pct_step = 10
            timer2 = time()
            timer3_ = 0.
            timer4_ = 0.
            timer5_ = 0.
            for idx, src in enumerate(paths_output):
                if idx % (n_outputs / (100 / pct_step)) == 0:
                    print_('\t\t%i%% (%.2f sec)' % (pct, time() - timer2))
                    # print('\t\t\tCnf: %.2f sec' % timer3_)
                    # print('\t\t\tCSV: %.2f sec' % timer4_)
                    # print('\t\t\tWrit: %.2f sec' % timer5_)
                    timer2 = time()
                    timer3_ = 0.
                    timer4_ = 0.
                    timer5_ = 0.
                    pct += pct_step

                dst = 'output/%s_%07i.png' % (dataset, cnt_out)

                # timer3 = time()
                cnf, cnf_idx = get_cnf(cnf, src)
                # timer3_ += time() - timer3
                #
                # timer4 = time()
                csv.append(cnf_idx, dataset, cnt_in, cnt_out)
                # timer4_ += time() - timer4
                #
                # timer5 = time()
                zipfile.write(src, dst)
                # timer5_ += time() - timer5

                # update counter for output images
                cnt_out += 1

            print_('\t\t100%')
            print_('\t\t...done')

            # update counter for input images
            cnt_in += 1

        # add csv to dataset
        print_('adding dataset.csv...')
        zipfile.writestr('dataset.csv', csv())
        print_('\t...done')


def get_cnf(cnf, src) -> [List[dict], dict]:
    # obtain the configuration of the given source
    src_name = Path(src).stem
    for idx in range(len(cnf)):
        if src_name == Path(cnf[idx]['filename']).stem:
            cnf_ = cnf[idx].copy()
            del cnf[idx]
            return cnf, cnf_
    # configuration with given source is not found, return empty dict
    raise Exception('ERROR: configuration with given filename not found')


class CSV:
    n_antennas = 12

    def __init__(self, n_projects):
        self.csv = bytearray()
        # initialize header and data array
        self.header = self.init_header()
        self.header_str = ';'.join(self.header) + '\n'
        self.data = np.empty(
            (n_projects * MAX_SAMPLES_PER_PROJECT, self.header.shape[0]),
            dtype='S30'
        )

    def init_header(self):
        header = [
            'idx',
            'input_permittivity',
            'input_conductivity',
            'input_density',
        ]
        for idx in range(self.n_antennas):
            header.append('amplitude_%02i' % idx)
            header.append('phase_%02i' % idx)
        header.append('output')
        return np.array(header).reshape(-1)

    def append(
            self,
            conf: dict,
            dataset: str,
            cnt_in: int,
            cnt_out: int
    ) -> None:
        # add index
        self.data[cnt_out, 0] = '%07i' % cnt_out

        # add each input image
        for idx, img in enumerate(['permittivity', 'conductivity', 'density']):
            self.data[cnt_out, idx + 1] = 'input/%s_%04i.png' % (img, cnt_in)

        # add amplitudes and normalized phases
        for idx in range(self.n_antennas):
            # add amplitude
            self.data[cnt_out, 2 * idx + 4] = \
                '%.17f' % conf['amplitudes'][idx]
            # add normalized phase
            self.data[cnt_out, 2 * idx + 5] = \
                '%.17f' % (conf['phases'][idx] / 2 / pi)

        # add output img
        self.data[cnt_out, -1] = 'output/%s_%07i.png' % (dataset, cnt_out)

    def to_str(self):
        return self.header_str + \
               '\n'.join(';'.join(row) for row in self.data.astype(str))

    def __call__(self, *args, **kwargs):
        return self.to_str()

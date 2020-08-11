import shutil
import zipfile
from pathlib import Path
from typing import Union

from ._complex_field_per_antenna import _ComplexFieldPerAntenna
from ._drawing_interchange_format import _DrawingInterchangeFormat
from ._mean_square_field import _MeanSquareField

default_dst = Path('./data')


def cst_to_dataset(src: Union[str, Path],
                   dst: Union[str, Path],
                   width: int,
                   height: int,
                   efield2_max: float,
                   max_samples: Union[int, float],
                   phase_randomness: float,
                   projection: str = '2d',
                   zipped: bool = True):
    """
    Converts the data generated in CST to a (zipped) PyTorch dataset.
    Projection can either be '2d' or '3d'
    the src must be a directory
    """

    # if the given dirs are a string, convert it to path
    if type(src) is str:
        src = Path(src)
    if type(dst) is str:
        dst = Path(dst)

    # make dst if it does not exist
    if not dst.exists():
        dst.mkdir()
        print('... created dst ' + str(dst))

    # csv file
    file_dataset = dst.joinpath('dataset.csv')

    # loop through source subdirs
    #   the source dir should contain multiple subdirs that are created during parametric sweep
    print('\nLooping through subdirs ...')
    ctr_output = -1
    ctr_input = -1
    n_dirs = len(list(src.iterdir()))
    for src_subdir in src.iterdir():
        ctr_output += 1
        ctr_input += 1

        # print name of subdir
        print('\n"' + str(src_subdir) + '"')

        # load complex field per antenna (cfa) into memory
        cfa = _ComplexFieldPerAntenna(src_subdir, projection)

        depth = None
        cfa.downsample(width, height, depth)

        # initialize input_bottleneck.csv
        if ctr_input is 0:
            print('\ninitializing dataset.csv ...')
            with open(file_dataset, 'w') as file:
                file.write('output_img;input_img')
                for ii in range(1, cfa.n_antennas):
                    file.write(';amplitude_%d;phase_%d' % (ii + 1, ii + 1))
                file.write('\n')

        # define mean square field object
        msf_obj = _MeanSquareField(cfa)

        # generate phases that are linearly distributed + some Gaussian randomness
        msf_obj.linearly_generate_phases(max_samples // n_dirs, phase_randomness)

        # output dir, create it if it doesn't exist
        dst_output_dir = dst.joinpath('output')
        if not dst_output_dir.exists():
            dst_output_dir.mkdir()
            print('... created dst_output_dir ' + str(dst_output_dir))

        # input dir, create it if it doesn't exist
        dst_input_dir = dst.joinpath('input')
        if not dst_input_dir.exists():
            dst_input_dir.mkdir()
            print('... created dst_input_dir ' + str(dst_input_dir))

        # loop over msf_obj, this will calculate the msf for each phaseshift
        print('  generating imgs from e-field ...')
        for msf, phase in msf_obj:
            # convert to file
            dst_img = dst_output_dir.joinpath('%07d.png' % ctr_output)
            msf_obj.export_as_png(dst_img, cfa.width, cfa.height, efield2_max)
            print('    created "' + str(dst_img) + '"')

            # append ids to pairing.csv
            amplitude = 1.0
            with open(file_dataset, 'a') as file:
                file.write('"output/%07d.png";"input/%07d.png";%.4f;%.4f\n' %
                           (ctr_output, ctr_input, amplitude, phase[1:]))

            ctr_output += 1

            # find model (.dxf file) and convert it to png
        print('  converting dxf model to png ...')
        for src in src_subdir.glob('*.dxf'):
            dst_img = dst_input_dir.joinpath('%07d.png' % ctr_input)
            dxf = _DrawingInterchangeFormat()
            dxf.read(src)
            dxf.export_as_png(dst_img, width, height, cfa.get_mm_per_px(), cfa.get_plane())
            print('    converted ' + str(src) + ' to ' + str(dst_img))

    # zip folder and delete files
    if zipped:
        print('creating zip folder...')
        zip_file = zipfile.ZipFile(str(dst) + '.zip', 'w')
        with zip_file:
            for file in dst.rglob('*'):
                zip_file.write(file)
        print('  ...Done')
        print('Removing files...')
        shutil.rmtree(dst)
        print('  ...Done')

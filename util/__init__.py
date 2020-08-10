import shutil
import imageio
from pathlib import Path
from numbers import Number
import zipfile
from .complex_field_per_antenna import _ComplexFieldPerAntenna
from .mean_square_field import _MeanSquareField
from .drawing_interchange_format import _DrawingInterchangeFormat


def cst_to_zip_dataset(src_dir, dst_dir, width, height, efield2_max, max_phaseshifts, phase_randomness,
                       projection='2d'):
    cst_to_dataset(src_dir, dst_dir, width, height, efield2_max, max_phaseshifts, phase_randomness, projection='2d')
    zip_folder_and_del(dst_dir)


def zip_folder(src):
    zip_file = zipfile.ZipFile(str(src)+'.zip', 'w')
    with zip_file:
        for file in src.rglob('*'):
            zip_file.write(file)


def zip_folder_and_del(src):
    print('creating zip folder...')
    zip_file = zipfile.ZipFile(str(src)+'.zip', 'w')
    with zip_file:
        for file in src.rglob('*'):
            zip_file.write(file)
    print('  ...Done')
    print('Removing files...')
    shutil.rmtree(src)
    print('  ...Done')


def cst_to_dataset(src_dir, dst_dir, width, height, efield2_max, max_phaseshifts, phase_randomness, projection='2d'):
    """
    tvt : training-validation-testing data , for neural network
    """

    # if the given dirs are a string, convert it to path
    if type(src_dir) is str:
        src_dir = Path(src_dir)
    if type(dst_dir) is str:
        dst_dir = Path(dst_dir)

    # sanity check type of arguments
    if not isinstance(src_dir, type(Path())):
        raise Exception('ERROR: src_dir must be either <str> or <pathlib.Path>')
    if not isinstance(dst_dir, type(Path())):
        raise Exception('ERROR: dst_dir must be either <str> or <pathlib.Path>')
    if not isinstance(efield2_max, Number):
        raise Exception('ERROR: efield2_max must be a number')
    if projection is not '2d' and projection is not '3d':
        raise Exception('ERROR: projection must be a <str> equal to 2d or 3d')

    # make dst_dir if it does not exist
    if not dst_dir.exists():
        dst_dir.mkdir()
        print('... created dst_dir ' + str(dst_dir))

    # csv file
    file_dataset = dst_dir.joinpath('dataset.csv')

    # loop through source subdirs
    #   the source dir should contain multiple subdirs that are created during parametric sweep
    print('\nLooping through subdirs ...')
    ctr_output = -1
    ctr_input = -1
    for src_subdir in src_dir.iterdir():
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
                    file.write(';amplitude_%d;phase_%d' % (ii+1, ii+1))
                file.write('\n')

        # define mean square field object
        msf_obj = _MeanSquareField(cfa)

        # generate phases that are linearly distributed + some Gaussian randomness
        msf_obj.linearly_generate_phases(max_phaseshifts, phase_randomness)

        # output dir, create it if it doesn't exist
        dst_output_dir = dst_dir.joinpath('output')
        if not dst_output_dir.exists():
            dst_output_dir.mkdir()
            print('... created dst_output_dir ' + str(dst_output_dir))

        # input dir, create it if it doesn't exist
        dst_input_dir = dst_dir.joinpath('input')
        if not dst_input_dir.exists():
            dst_input_dir.mkdir()
            print('... created dst_input_dir ' + str(dst_input_dir))

        # loop over msf_obj, this will calculate the msf for each phaseshift
        print('  generating imgs from e-field ...')
        for msf, phase in msf_obj:

            # convert to file
            dst = dst_output_dir.joinpath('%07d.png' % ctr_output)
            msf_obj.export_as_png(dst, cfa.width, cfa.height, efield2_max)
            print('    created "' + str(dst) + '"')

            # append ids to pairing.csv
            amplitude = 1.0
            with open(file_dataset, 'a') as file:
                file.write('"output/%07d.png";"input/%07d.png";%.4f;%.4f\n' %
                           (ctr_output, ctr_input, amplitude, phase[1:]))

            ctr_output += 1

            # find model (.dxf file) and convert it to png
        print('  converting dxf model to png ...')
        for src in src_subdir.glob('*.dxf'):
            dst = dst_input_dir.joinpath('%07d.png' % ctr_input)
            dxf = _DrawingInterchangeFormat()
            dxf.read(src)
            dxf.export_as_png(dst, width, height, cfa.get_mm_per_px(), cfa.get_plane())
            print('    converted ' + str(src) + ' to ' + str(dst))


def cst_to_imgs(src_dir, dst_dir, width, height, efield2_max, max_phaseshifts, phase_randomness, projection='2d'):
    """
    WARNING: DEPRECIATED , use cst_to_dataset()
    """

    # if the given dirs are a string, convert it to path
    if type(src_dir) is str:
        src_dir = Path(src_dir)
    if type(dst_dir) is str:
        dst_dir = Path(dst_dir)

    # sanity check type of arguments
    if not isinstance(src_dir, type(Path())):
        raise Exception('ERROR: src_dir must be either <str> or <pathlib.Path>')
    if not isinstance(dst_dir, type(Path())):
        raise Exception('ERROR: dst_dir must be either <str> or <pathlib.Path>')
    if not isinstance(efield2_max, Number):
        raise Exception('ERROR: efield2_max must be a number')
    if projection is not '2d' and projection is not '3d':
        raise Exception('ERROR: projection must be a <str> equal to 2d or 3d')

    # make dst_dir if it does not exist
    if not dst_dir.exists():
        dst_dir.mkdir()
        print('... created dst_dir ' + str(dst_dir))

    # loop through source subdirs
    #   the source dir should contain multiple subdirs that are created during parametric sweep
    print('\nLooping through subdirs ...')
    for src_subdir in src_dir.iterdir():

        # print name of subdir
        print('\n"' + str(src_subdir) + '"')

        # create destination subdir with same name, create it if needed
        dst_subdir = dst_dir.joinpath(src_subdir.stem)
        if not dst_subdir.exists():
            print('  creating subdir ... ')
            dst_subdir.mkdir()
            print('    "' + str(dst_subdir) + '"')

        # create subdir to place the efield images in, if needed
        dst_efield_dir = dst_subdir.joinpath('E-field')
        if not dst_efield_dir.exists():
            print('  creating efield dir ...')
            dst_efield_dir.mkdir()
            print('    "' + str(dst_efield_dir) + '"')

        # load complex field per antenna (cfa) into memory
        cfa = _ComplexFieldPerAntenna(src_subdir, projection)

        depth = None
        cfa.downsample(width, height, depth)

        # define mean square field object
        msf_obj = _MeanSquareField(cfa)

        # generate phases that are linearly distributed + some Gaussian randomness
        msf_obj.linearly_generate_phases(max_phaseshifts, phase_randomness)

        # loop over msf_obj, this will calculate the msf for each phaseshift, then export it as png
        print('  generating imgs from e-field ...')
        for idx, msf in enumerate(msf_obj):
            dst = dst_efield_dir.joinpath('efield_%04d.png' % idx)
            file = msf.export_as_png(dst, cfa.width, cfa.height, efield2_max)
            print('    created "' + str(file) + '"')

        # create gif from e-field images
        print('  creating GIF from E-field PNGs ...')
        imgs = []
        for src in sorted(dst_efield_dir.glob('*.png')):
            imgs.append(imageio.imread(src))
        dst = dst_subdir.joinpath('efield.gif')
        imageio.mimsave(dst, imgs)
        print('    created "' + str(dst) + '"')

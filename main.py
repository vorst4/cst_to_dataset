from pathlib import Path
import numpy as np
import util

# settings
n_max = 50e3
src = 'C:/Users/Dennis/Documents/cst projects/simple v6/Export'
width = 32
height = width
mm_per_px = 1 / 8
efield2_max = 4e3
max_phaseshifts = np.floor(n_max / 3)
phase_randomness = 0
dst = Path('data')

# convert cst data to zipped dataset
util.cst_to_zip_dataset(src, dst, width, height, efield2_max, max_phaseshifts, phase_randomness)

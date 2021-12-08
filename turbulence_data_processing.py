import os
from giverny import turbulence_isotropic_cube as turb_processing

# user-defined parameters for processing data.
# size of the model cube that data will be retrieved for.
cube_num = 8192
# number of dimensions that model data exists in.  default is 3 (i.e. X, Y, and Z dimensions).
cube_dimensions = 3
# turbulence dataset name, e.g. "isotropic8192" or "isotropic1024fine".
cube_title = 'isotropic8192'
# folder name to write the hdf5 output files to.
output_folder_name = 'turbulence_hdf5_output'

# user specified box rather for which data values will be retrieved for each point inside the box.
# the user should specify the 1-based index range. this code will convert to 0-based index ranges for python.
x_range = [1, 512]
y_range = [1, 512]
z_range = [1, 512]

# converts the 1-based axes ranges above to 0-based axes ranges.
x_range, y_range, z_range = turb_processing.convert_to_0_based_ranges(x_range, y_range, z_range)

# variable of interest, currently set to velocity.
var = 'vel'
# time point. the user should specify the 1-based timepoint. this code will convert to a 0-based timepoint for python.
timepoint = 1

# converts the 1-based timepoint above to a 0-based timepoint.
timepoint = turb_processing.convert_to_0_based_value(timepoint)

# process the data.
# -----
# create the output folder directory if it does not already exist.
dir_path = os.path.dirname(os.path.realpath('__file__')) + '/'
output_path = dir_path + output_folder_name + '/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

# parse the database files, generate the output_data matrix, and write the matrix to an hdf5 file.
output_data = turb_processing.process_data(cube_num, cube_dimensions, cube_title, output_path, x_range, y_range, z_range, var, timepoint)
import os
from giverny import turbulence_isotropic_cube as turb_cube

def turbulence_cube_processing(cube_num, cube_dimensions, cube_title, dir_path, output_folder_name, \
                               x_range, y_range, z_range, var, timepoint):
    # converts the 1-based axes ranges above to 0-based axes ranges.
    x_range, y_range, z_range = turb_cube.convert_to_0_based_ranges(x_range, y_range, z_range)

    # converts the 1-based timepoint above to a 0-based timepoint.
    timepoint = turb_cube.convert_to_0_based_value(timepoint)

    # process the data.
    # -----
    # create the output folder directory if it does not already exist.
    output_path = dir_path + output_folder_name + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # parse the database files, generate the output_data matrix, and write the matrix to an hdf5 file.
    output_data = turb_cube.process_data(cube_num, cube_dimensions, cube_title, output_path, x_range, y_range, z_range, var, timepoint)
    
    return output_data
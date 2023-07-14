import os
import sys
import math
import time
import numpy as np
import xarray as xr
from giverny.turbulence_gizmos.constants import *
from giverny.turbulence_gizmos.basic_gizmos import *

def getCutout_process_data(cube, axes_ranges, var, timepoint,
                           axes_ranges_original, strides, var_original, timepoint_original,
                           time_step = 1, filter_width = 1):
    # calculate how much time it takes to run the code.
    start_time = time.perf_counter()

    # data constants.
    # -----
    c = get_constants()
    dataset_title = cube.dataset_title

    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)
    
    # placeholder spatial interpolation value which is not used for getCutout.
    sint = ''
    # initialize cube constants.
    cube.init_constants(var, var_original, timepoint, sint, num_values_per_datapoint, c)

    # used for determining the indices in the output array for each x, y, z datapoint.
    axes_min = axes_ranges[:, 0]

    # used for creating the 3-D output array using numpy.
    axes_lengths = axes_ranges[:, 1] - axes_ranges[:, 0] + 1

    # total number of datapoints, used for checking if the user requested too much data.
    num_datapoints = np.prod(axes_lengths)
    # total size of data, in GBs, requested by the user's box.
    requested_data_size = (num_datapoints * c['bytes_per_datapoint'] * num_values_per_datapoint) / float(1024**3)
    # maximum number of datapoints that can be read in. currently set to 3 GBs worth of datapoints.
    max_datapoints = int((c['max_data_size'] * (1024**3)) / (c['bytes_per_datapoint'] * float(num_values_per_datapoint)))
    # approximate max size of a cube representing the maximum data points. this number is rounded down.
    approx_max_cube = int(max_datapoints**(1/3))

    if requested_data_size > c['max_data_size']:
        raise ValueError(f'Please specify a box with fewer than {max_datapoints} data points. This represents an approximate cube size ' + \
                         f'of ({approx_max_cube} x {approx_max_cube} x {approx_max_cube}).')

    # begin processing of data.
    # -----
    print('Note: For larger boxes, e.g. 512-cubed and up, processing will take approximately 1 minute or more...\n' + '-' * 5)
    sys.stdout.flush()
    
    # -----
    # get a map of the database files where all the data points are in.
    print('\nStep 1: Determining which database files the user-specified box is found in...\n' + '-' * 25)
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 1.
    start_time_step1 = time.perf_counter()
    
    user_single_db_boxes = cube.identify_single_database_file_sub_boxes(axes_ranges)

    num_db_files = sum(len(value) for value in user_single_db_boxes.values())
    num_db_disks = len(user_single_db_boxes)
    print(f'number of database files that the user-specified box is found in:\n{num_db_files}\n')
    print(f'number of hard disks that the database files are distributed on:\n{num_db_disks}\n')
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 1.
    end_time_step1 = time.perf_counter()

    print('Successfully completed.\n' + '-' * 5)
    sys.stdout.flush()

    # -----
    # read the data.
    print('\nStep 2: Reading the data from all of the database files and storing the values into a matrix...\n' + '-' * 25)
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 3.
    start_time_step2 = time.perf_counter()
    
    # pre-fill the output data 3-d array that will be filled with the data that is read in. initially the datatype is set to "f" (float)
    # so that the array is filled with the missing placeholder values (-999.9). 
    output_data = np.full((axes_lengths[2], axes_lengths[1], axes_lengths[0], num_values_per_datapoint),
                           fill_value = c['missing_value_placeholder'], dtype = 'f')
    
    # determines if the database files will be read sequentially with base python, or in parallel with dask.
    if num_db_disks == 1:
        # sequential processing.
        print('Database files are being read sequentially...')
        sys.stdout.flush()
        
        result_output_data = cube.read_database_files_sequentially(user_single_db_boxes)
    else:
        # parallel processing.
        result_output_data = cube.read_database_files_in_parallel_with_dask(user_single_db_boxes)
    
    # iterate over the results to fill output_data.
    for result in result_output_data:
        output_data[result[1][2] - axes_min[2] : result[2][2] - axes_min[2] + 1,
                    result[1][1] - axes_min[1] : result[2][1] - axes_min[1] + 1,
                    result[1][0] - axes_min[0] : result[2][0] - axes_min[0] + 1] = result[0]
    
    # determines how many copies of data need to be me made along each axis when the number of datapoints the user specified
    # exceeds the cube resolution (cube.N). note: no copies of the data values should be made, hence data_value_multiplier equals 1.
    axes_multipliers = np.ceil(axes_lengths / cube.N).astype(int)
    data_value_multiplier = 1
    
    # duplicates the data along the z-, y-, and x-axes of output_data if the the user asked for more datapoints than the cube resolution along any axis.
    if np.any(axes_multipliers > 1):
        output_data = np.tile(output_data, (axes_multipliers[2], axes_multipliers[1], axes_multipliers[0], data_value_multiplier))
        # truncate any extra datapoints from the duplicate data outside of the original range of the datapoints specified by the user.
        output_data = np.copy(output_data[0 : axes_lengths[2], 0 : axes_lengths[1], 0 : axes_lengths[0], :])
    
    # create axis coordinate ranges, shifted to 0-based indices, to store in the xarray metadata.
    z_coords = np.around(np.arange(axes_ranges_original[2][0] - 1, axes_ranges_original[2][1], strides[2], dtype = np.float32) * cube.dx, decimals = c['decimals'])
    y_coords = np.around(np.arange(axes_ranges_original[1][0] - 1, axes_ranges_original[1][1], strides[1], dtype = np.float32) * cube.dx, decimals = c['decimals'])
    x_coords = np.around(np.arange(axes_ranges_original[0][0] - 1, axes_ranges_original[0][1], strides[0], dtype = np.float32) * cube.dx, decimals = c['decimals'])
    
    # set the dataset name to be used in the hdf5 file.
    h5_var_name = get_output_variable_name(var_original)
    h5_attribute_type = get_attribute_type(var_original)
    h5_dataset_name = h5_var_name + '_' + str(timepoint_original).zfill(4)
    
    # apply the strides to output_data.
    output_data = xr.DataArray(data = output_data[::strides[2], ::strides[1], ::strides[0], :],
                               dims = ['zcoor', 'ycoor', 'xcoor', 'values'])
    
    output_data = xr.Dataset(data_vars = {h5_dataset_name:output_data},
                             coords = {'zcoor':z_coords, 'ycoor':y_coords, 'xcoor':x_coords}, 
                             attrs = {'dataset':dataset_title, 't_start':timepoint_original, 't_end':timepoint_original, 't_step':time_step,
                                      'x_start':axes_ranges_original[0][0], 'y_start':axes_ranges_original[1][0], 'z_start':axes_ranges_original[2][0], 
                                      'x_end':axes_ranges_original[0][1], 'y_end':axes_ranges_original[1][1], 'z_end':axes_ranges_original[2][1],
                                      'x_step':strides[0], 'y_step':strides[1], 'z_step':strides[2],
                                      'filterWidth':filter_width})
    
    # checks to make sure that data was read in for all points.
    if c['missing_value_placeholder'] in output_data:
        raise Exception(f'output_data was not filled correctly')
    
    # calculate how much time it takes to run step 3.
    end_time_step2 = time.perf_counter()
    
    print('\nSuccessfully completed.\n' + '-' * 5)
    sys.stdout.flush()
    
    # -----
    # write the output file.
    print('\nStep 3: Writing the output matrix to a hdf5 file...\n' + '-' * 25)
    sys.stdout.flush()
    
    # calculate how much time it takes to run step 4.
    start_time_step3 = time.perf_counter()
    
    # write output_data to a hdf5 file.
    # the output filename specifies the title of the cube, and the x-, y-, and z-ranges so that the file is unique. 1 is added to all of the 
    # ranges, and the timepoint, because python uses 0-based indices, and the output is desired to be 1-based indices.
    output_filename = f'{dataset_title}_{var_original}_' + \
                      f't{timepoint_original}_' + \
                      f'z{axes_ranges_original[2][0]}-{axes_ranges_original[2][1]}_' + \
                      f'y{axes_ranges_original[1][0]}-{axes_ranges_original[1][1]}_' + \
                      f'x{axes_ranges_original[0][0]}-{axes_ranges_original[0][1]}'
    
    # writes the output hdf5 file.
#     cube.write_output_matrix_to_hdf5(output_data, output_filename)

    # writes the output xmf file.
#     cube.write_xmf(axes_lengths, h5_var_name, h5_attribute_type, h5_dataset_name, output_filename)
    
    # calculate how much time it takes to run step 4.
    end_time_step3 = time.perf_counter()
    
    print('\nSuccessfully completed.\n' + '-' * 5)
    sys.stdout.flush()

    end_time = time.perf_counter()
    
    # see how long the program took to run.
    print(f'\nstep 1 time elapsed = {end_time_step1 - start_time_step1:0.3f} seconds ({(end_time_step1 - start_time_step1) / 60:0.3f} minutes)')
    print(f'step 2 time elapsed = {end_time_step2 - start_time_step2:0.3f} seconds ({(end_time_step2 - start_time_step2) / 60:0.3f} minutes)')
    print(f'step 3 time elapsed = {end_time_step3 - start_time_step3:0.3f} seconds ({(end_time_step3 - start_time_step3) / 60:0.3f} minutes)')
    print(f'total time elapsed = {end_time - start_time:0.3f} seconds ({(end_time - start_time) / 60:0.3f} minutes)')
    sys.stdout.flush()
    
    print('\nData processing pipeline has completed successfully.\n' + '-' * 5)
    sys.stdout.flush()
    
    return output_data
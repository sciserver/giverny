import os
import sys
import math
import time
import numpy as np
import xarray as xr
from giverny.turbulence_gizmos.constants import *
from giverny.turbulence_gizmos.basic_gizmos import *

def getCutout_process_data(cube, axes_ranges, var, timepoint,
                           axes_ranges_original, strides, var_original, var_dimension_offsets, timepoint_original,
                           time_step = 1, filter_width = 1,
                           verbose = False):
    # calculate how much time it takes to run the code.
    start_time = time.perf_counter()

    # data constants.
    # -----
    c = get_constants()
    dataset_title = cube.dataset_title

    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)
    
    # define the query type.
    query_type = 'getcutout'
    
    # placeholder spatial interpolation, temporal interpolation, and option values which are not used for getCutout.
    sint = 'none'
    sint_specified = 'none'
    tint = 'none'
    option = ['none', 'none']
    # initialize cube constants.
    cube.init_constants(query_type, var, var_original, var_dimension_offsets, timepoint, timepoint_original, sint, sint_specified, tint, option, num_values_per_datapoint, c)

    # used for determining the indices in the output array for each x, y, z datapoint.
    axes_min = axes_ranges[:, 0]

    # used for creating the 3-D output array using numpy.
    axes_lengths = axes_ranges[:, 1] - axes_ranges[:, 0] + 1

    # begin processing of data.
    # -----
    # get a map of the database files where all the data points are in.
    if verbose:
        print('\nstep 1: determining which database files the user-specified box is found in...\n' + '-' * 25)
        sys.stdout.flush()
    
    # calculate how much time it takes to run step 1.
    start_time_step1 = time.perf_counter()
    
    user_single_db_boxes = cube.identify_single_database_file_sub_boxes(axes_ranges)

    num_db_files = sum(len(value) for value in user_single_db_boxes.values())
    num_db_disks = len(user_single_db_boxes)
    if verbose:
        print(f'number of database files that the user-specified box is found in:\n{num_db_files}\n')
        print(f'number of hard disks that the database files are distributed on:\n{num_db_disks}\n')
        sys.stdout.flush()
    
    # calculate how much time it takes to run step 1.
    end_time_step1 = time.perf_counter()

    if verbose:
        print('successfully completed.\n' + '-' * 5)
        sys.stdout.flush()

    # -----
    # read the data.
    if verbose:
        print('\nstep 2: reading the data from all of the database files and storing the values into a matrix...\n' + '-' * 25)
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
        if verbose:
            print('database files are being read sequentially...')
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
    
    # calculate how much time it takes to run step 3.
    end_time_step2 = time.perf_counter()
    
    if verbose:
        print('\nsuccessfully completed.\n' + '-' * 5)
        sys.stdout.flush()

    end_time = time.perf_counter()
    
    # see how long the program took to run.
    if verbose:
        print(f'\nstep 1 time elapsed = {end_time_step1 - start_time_step1:0.3f} seconds ({(end_time_step1 - start_time_step1) / 60:0.3f} minutes)')
        print(f'step 2 time elapsed = {end_time_step2 - start_time_step2:0.3f} seconds ({(end_time_step2 - start_time_step2) / 60:0.3f} minutes)')
        # print(f'step 3 time elapsed = {end_time_step3 - start_time_step3:0.3f} seconds ({(end_time_step3 - start_time_step3) / 60:0.3f} minutes)')
        print(f'total time elapsed = {end_time - start_time:0.3f} seconds ({(end_time - start_time) / 60:0.3f} minutes)')
        sys.stdout.flush()
    
    if verbose:
        print('\nquery completed successfully.\n' + '-' * 5)
        sys.stdout.flush()
    
    return output_data
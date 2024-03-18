import os
import sys
import math
import time
import numpy as np
import xarray as xr
from giverny.turbulence_gizmos.constants import *
from giverny.turbulence_gizmos.basic_gizmos import *

def getData_process_data(cube, points,
                         var, timepoint, tint, sint,
                         var_original, var_dimension_offsets, timepoint_original, sint_specified, option,
                         verbose = False):
    # data constants.
    # -----
    c = get_constants()

    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)
    
    # define the query type.
    query_type = 'getdata'
    
    # initialize cube constants.
    cube.init_constants(query_type, var, var_original, var_dimension_offsets, timepoint, timepoint_original, sint, sint_specified, tint, option, num_values_per_datapoint, c)

    # begin processing of data.
    # -----
    # mapping the points to database files and sorting them into native and visitor bucket maps.
    if verbose:
        print('\nStep 1: Sorting the points to native and visitor bucket maps...\n' + '-' * 25)
        sys.stdout.flush()
    
    # calculate how much time it takes to run step 1.
    start_time_step1 = time.perf_counter()
    
    # get the maps of points that require native and visitor buckets for interpolation.
    db_native_map, db_visitor_map = cube.identify_database_file_points(points)
    
    if verbose:
        print(f'len db_native_map = \n{len(db_native_map)}')
        print('-')
        print(f'len db_visitor_map = \n{len(db_visitor_map)}')
    
    # calculate how much time it takes to run step 1.
    end_time_step1 = time.perf_counter()
    
    if verbose:
        print('\nSuccessfully completed.\n' + '-' * 5)
        sys.stdout.flush()
    
    # read and interpolate the data.
    if verbose:
        print('\nStep 2: Interpolating and differentiating...\n' + '-' * 25)
        sys.stdout.flush()
    
    # calculate how much time it takes to run step 2.
    start_time_step2 = time.perf_counter()
    
    # determines if the database files will be read sequentially with base python, or in parallel with dask.
    if len(db_native_map) == 1:
        # sequential processing.
        result_output_data = cube.read_database_files_sequentially_variable(db_native_map, db_visitor_map)
    else:
        # parallel processing.
        result_output_data = cube.read_database_files_in_parallel_with_dask_variable(db_native_map, db_visitor_map)
    
    # iterate over the results to fill output_data.
    output_data = []
    original_points_indices = []
    for original_point_index, result in result_output_data:
        output_data.append(result)
        original_points_indices.append(original_point_index)
    
    # re-sort output_data to match the original ordering of points.
    output_data_indices, output_data = zip(*sorted(zip(original_points_indices, output_data), key = lambda x: x[0]))
    
    # remove the points from output_data to reduce data transmitted over the network.
    output_data = np.array([output_vals[1] for output_vals in output_data])
    
    # calculate how much time it takes to run step 2.
    end_time_step2 = time.perf_counter()
    
    if verbose:
        print('\nSuccessfully completed.\n' + '-' * 5)
        sys.stdout.flush()
    
    # -----
    # see how long the program took to run.
    if verbose:
        print(f'\nstep 1 time elapsed = {end_time_step1 - start_time_step1:0.3f} seconds ({(end_time_step1 - start_time_step1) / 60:0.3f} minutes)')
        print(f'step 2 time elapsed = {end_time_step2 - start_time_step2:0.3f} seconds ({(end_time_step2 - start_time_step2) / 60:0.3f} minutes)')
    
    return output_data
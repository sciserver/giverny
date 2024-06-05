########################################################################
#
#  Copyright 2024 Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact: turbulence@pha.jhu.edu
# Website: http://turbulence.pha.jhu.edu/
#
########################################################################

import sys
import math
import time
import numpy as np
from giverny.turbulence_gizmos.basic_gizmos import get_num_values_per_datapoint

def getCutout_process_data(cube, axes_ranges, var, timepoint,
                           axes_ranges_original, strides, var_original, var_dimension_offsets, timepoint_original, c,
                           verbose = False):
    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)
    
    # define the query type.
    query_type = 'getcutout'
    
    # placeholder spatial interpolation, temporal interpolation, and option values which are not used for getCutout.
    sint = 'none'
    sint_specified = 'none'
    tint = 'none'
    option = [-999.9, -999.9]
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
    
    user_single_db_boxes = cube.map_chunks_getcutout(axes_ranges)

    num_db_files = sum(len(value) for value in user_single_db_boxes)
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
    
    # calculate how much time it takes to run step 2.
    start_time_step2 = time.perf_counter()
    
    # pre-fill the output data 3-d array that will be filled with the data that is read in. initially the datatype is set to "f" (float)
    # so that the array is filled with the missing placeholder values (-999.9). 
    output_data = np.full((axes_lengths[2], axes_lengths[1], axes_lengths[0], num_values_per_datapoint),
                           fill_value = c['missing_value_placeholder'], dtype = 'f')
    
    # sequential and parallel processing.
    result_output_data = cube.read_database_files_getcutout(user_single_db_boxes)
    
    # iterate over the results to fill output_data.
    for result in result_output_data:
        output_data[result[1][2] - axes_min[2] : result[2][2] - axes_min[2] + 1,
                    result[1][1] - axes_min[1] : result[2][1] - axes_min[1] + 1,
                    result[1][0] - axes_min[0] : result[2][0] - axes_min[0] + 1] = result[0]
    
    # calculate how much time it takes to run step 2.
    end_time_step2 = time.perf_counter()
    
    if verbose:
        print('\nsuccessfully completed.\n' + '-' * 5)
        sys.stdout.flush()
    
    # see how long the program took to run.
    if verbose:
        print(f'\nstep 1 time elapsed = {end_time_step1 - start_time_step1:0.3f} seconds ({(end_time_step1 - start_time_step1) / 60:0.3f} minutes)')
        print(f'step 2 time elapsed = {end_time_step2 - start_time_step2:0.3f} seconds ({(end_time_step2 - start_time_step2) / 60:0.3f} minutes)')
        sys.stdout.flush()
    
    if verbose:
        print('\nquery completed successfully.\n' + '-' * 5)
        sys.stdout.flush()
    
    return output_data
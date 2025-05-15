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

import os
import sys
import math
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def get_parquet_filenames_map(dataset_title, times, time_offset, dt):
    """
    map the user-specified times to the parquet filenames they are stored in.
    """
    # define the hour sequence (starting from 15:00).
    hour_sequence = np.array([(15 + i) % 24 for i in range(24)], dtype = np.int64)
    
    # stored timepoint before and after the user-specified timepoint.
    before_times = np.floor(times / dt) * dt
    after_times = np.ceil(times / dt) * dt
    
    # hour indices for each time in before_times and after_times. 3600 is the number of seconds per hour.
    before_indices = ((before_times - time_offset) / 3600).astype(np.int64)
    after_indices = ((after_times - time_offset) / 3600).astype(np.int64)
    
    # hour number for each time in before_times and after_times. multiplied by 100 to match parquet filename formatting.
    before_hours = hour_sequence[before_indices] * 100
    after_hours = hour_sequence[after_indices] * 100
    
    # formulate the parquet filenames from the hour information above. filenames for before_hours and after_hours are formulated in case
    # a time needs to be interpolated from the stored timepoints in adjacent files. users cannot specify timepoints at the beginning or ending
    # time boundaries to align with getData queries which allow pchip interpolation - so there is no issue with the time axis being non-periodic.
    if dataset_title == 'diurnal_windfarm':
        format_filename = np.vectorize(lambda hour: f"turbineOutput{hour:04d}{(hour + 100):04d}")
    elif dataset_title == 'nbl_windfarm':
        # only 1 hour of data is stored for the 'nbl_windfarm' dataset.
        format_filename = np.vectorize(lambda hour: f"turbineOutput")
    before_filenames = format_filename(before_hours)
    after_filenames = format_filename(after_hours)
    
    parquet_filenames_map = defaultdict(list)
    for before_filename, after_filename, time, before_time, after_time in \
        zip(before_filenames, after_filenames, times, before_times, after_times):
        parquet_filenames_map[before_filename + ',' + after_filename].append((time, before_time, after_time))
        
    for key in parquet_filenames_map:
        parquet_filenames_map[key] = np.array(parquet_filenames_map[key])
    
    return parquet_filenames_map

def t_linear(df, times, before_times, after_times, before_values, after_values, time_offset, time_align, turbine_number, blade_number):
    """
    linear time interpolations.
    """
    # create a mask for user-specified timepoints that are not exact stored timepoints. user-specified timepoints that exactly
    # match stored timepoints will be returned without interpolation.
    non_exact_mask = ~np.isclose(after_times - before_times, 0.0, rtol = 10**-9, atol = 0.0)

    # calculate weight for each of the timepoints in between stored timepoints.
    weight = np.zeros_like(times, dtype = np.float64)
    if np.any(non_exact_mask):
        weight[non_exact_mask] = (times[non_exact_mask] - before_times[non_exact_mask]) / \
                                 (after_times[non_exact_mask] - before_times[non_exact_mask])

    # reshape weight to allow broadcasting with before_values and after_values. i.e. change shape from (..., ) to (..., 1).
    weight = weight[:, np.newaxis]

    interpolated_values = pd.DataFrame(((1 - weight) * before_values[:, 3:]) + (weight * after_values[:, 3:]))
    interpolated_values.columns = [col for col in df.columns if col not in ['time', 'turbine', 'blade']]
    interpolated_values.insert(0, 'time', times - (time_offset + time_align))
    interpolated_values.insert(1, 'turbine', turbine_number)
    interpolated_values.insert(2, 'blade', blade_number)
    
    return interpolated_values

def read_parquet_files_getbladedata(args):
    """
    read from the parquet files.
    """
    filename_key, times_data, turbine_number, blade_numbers, blade_points, var, folderpath, time_offset, time_align, dt = args
    
    before_filename, after_filename = filename_key.split(',')
    times = times_data[:, 0]
    before_times = times_data[:, 1]
    after_times = times_data[:, 2]
    
    # get a list of columns that we want to retrieve from the parquet files.
    blade_points_columns = [f"{var}_{point}" for point in blade_points]
    get_columns = ['time', 'turbine', 'blade'] + blade_points_columns
    
    blade_data = []
    if before_filename == after_filename:
        """
        handles cases where the preceding time and succeeding time are in the same parquet file.
        """
        # read the parquet file with column filtering.
        file_path = folderpath + f'TURBINE_{turbine_number}{os.sep}{var}{os.sep}{before_filename}.parquet'
        
        for blade_number in blade_numbers:
            # use pyarrow to filter both columns and rows at once.
            filters = [('blade', '=', blade_number)]
            df1_filtered = pq.read_table(
                file_path, 
                columns = get_columns, 
                filters = filters
            ).to_pandas()
            
            df1_initial_time = df1_filtered['time'].iloc[0]
            # calculate the table indices for before_times and after_times.
            before_indices = np.floor(((before_times - df1_initial_time) / dt) + dt).astype(int)
            after_indices = np.floor(((after_times - df1_initial_time) / dt) + dt).astype(int)
            
            # retrieve the values for the blade_points.
            before_values = df1_filtered.iloc[before_indices].to_numpy()
            after_values = df1_filtered.iloc[after_indices].to_numpy()
            
            blade_data.append(t_linear(df1_filtered, times, before_times, after_times, before_values, after_values, time_offset, time_align, turbine_number, blade_number))
    else:
        """
        handles cases where the preceding time and succeeding time are in different parquet files.
        """
        # read the parquet files with column filtering.
        file_path1 = folderpath + f'TURBINE_{turbine_number}{os.sep}{var}{os.sep}{before_filename}.parquet'
        file_path2 = folderpath + f'TURBINE_{turbine_number}{os.sep}{var}{os.sep}{after_filename}.parquet'
        
        for blade_number in blade_numbers:
            # Use pyarrow to filter both columns and rows at once.
            filters = [('blade', '=', blade_number)]
            df1_filtered = pq.read_table(
                file_path1, 
                columns = get_columns, 
                filters = filters
            ).to_pandas()
            
            df2_filtered = pq.read_table(
                file_path2, 
                columns = get_columns, 
                filters = filters
            ).to_pandas()
            
            df1_initial_time = df1_filtered['time'].iloc[0]
            df2_initial_time = df2_filtered['time'].iloc[0]
            # calculate the table indices for before_times and after_times.
            before_indices = np.floor(((before_times - df1_initial_time) / dt) + dt).astype(int)
            after_indices = np.floor(((after_times - df2_initial_time) / dt) + dt).astype(int)
            
            # retrieve the values for the blade_points.
            before_values = df1_filtered.iloc[before_indices].to_numpy()
            after_values = df2_filtered.iloc[after_indices].to_numpy()
            
            blade_data.append(t_linear(df1_filtered, times, before_times, after_times, before_values, after_values, time_offset, time_align, turbine_number, blade_number))
    
    return pd.concat(blade_data, ignore_index = True)

def getBladeData_process_data(dataset_title, times, turbine_number, blade_numbers, blade_points, var, folderpath,
                              time_offset, time_align, dt,
                              verbose = False):
    # define the query type.
    query_type = 'getbladedata'
    
    # begin processing of data.
    # -----
    # mapping the times to parquet filenames.
    if verbose:
        print('\nstep 1: mapping the times to parquet filenames...\n' + '-' * 25)
        sys.stdout.flush()
    
    # calculate how much time it takes to run step 1.
    start_time_step1 = time.perf_counter()
    
    # map the user-specified times to the parquet filenames they are stored in. times in between two timepoints are linearly interpolated.
    parquet_filenames_map = get_parquet_filenames_map(dataset_title, times, time_offset, dt)
    
    # calculate how much time it takes to run step 1.
    end_time_step1 = time.perf_counter()
    
    if verbose:
        print('\nsuccessfully completed.\n' + '-' * 5)
        sys.stdout.flush()
        
    # read and interpolate the data.
    if verbose:
        print('\nstep 2: reading and interpolating...\n' + '-' * 25)
        sys.stdout.flush()
    
    # calculate how much time it takes to run step 2.
    start_time_step2 = time.perf_counter()
    
    # prepare arguments for parallel processing.
    args_list = [
        (key, value, turbine_number, blade_numbers, blade_points, var, folderpath, time_offset, time_align, dt) 
        for key, value in parquet_filenames_map.items()
    ]
    
    # read the parquet files in parallel using ProcessPoolExecutor. there is 1 parquet file per hour of the day.
    max_workers = 24
    num_workers = min(max_workers, len(parquet_filenames_map))
    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        results = list(executor.map(read_parquet_files_getbladedata, args_list))
    
    # concatenate the results and reset the indices of the concatenated pandas dataframe.
    result = pd.concat(results, ignore_index = True)
    
    # handles the 'xPos' variable offset.
    if var == 'xPos':
        if dataset_title == 'diurnal_windfarm':
            xPos_offset = 5953.5
        elif dataset_title == 'nbl_windfarm':
            xPos_offset = 10584
            
        result.loc[:, result.columns.str.contains('xPos')] += xPos_offset
        
    # calculate how much time it takes to run step 2.
    end_time_step2 = time.perf_counter()
    
    if verbose:
        print('\nsuccessfully completed.\n' + '-' * 5)
        sys.stdout.flush()
        
    # -----
    # see how long the program took to run.
    if verbose:
        print(f'\nstep 1 time elapsed = {end_time_step1 - start_time_step1:0.3f} seconds ({(end_time_step1 - start_time_step1) / 60:0.3f} minutes)')
        print(f'step 2 time elapsed = {end_time_step2 - start_time_step2:0.3f} seconds ({(end_time_step2 - start_time_step2) / 60:0.3f} minutes)')
    
    return result

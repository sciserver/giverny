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
import json
import math
import time
import logging
import requests
import tracemalloc
import numpy as np
import pandas as pd
import xarray as xr
from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_gizmos.basic_gizmos import *

def getCutout(cube, var, xyzt_axes_ranges_original, xyzt_strides,
              trace_memory = False, verbose = True):
    """
    retrieve a cutout of the isotropic cube.
    """
    if verbose:
        print('\n' + '-' * 5 + '\ngetCutout is processing...')
        sys.stdout.flush()
        
    # calculate how much time it takes to run the code.
    start_time = time.perf_counter()
    
    # set cube attributes.
    metadata = cube.metadata
    dataset_title = cube.dataset_title
    auth_token = cube.auth_token
    
    # define the query type.
    query_type = 'getcutout'
    
    # data constants.
    c = metadata['constants']
    
    # only filter_width value of 1 is currently allowed.
    filter_width = 1
    
    # retrieve the list of datasets processed by the giverny code.
    giverny_datasets = get_giverny_datasets()
    
    # xyz original axes ranges.
    axes_ranges_original = xyzt_axes_ranges_original[:3]
    # time original range.
    timepoint_range_original = xyzt_axes_ranges_original[3]
    # xyz original axes strides.
    strides = xyzt_strides[:3]
    # time original stride.
    timepoint_stride = xyzt_strides[3]
    
    # housekeeping procedures.
    # -----
    var_offsets, timepoint_range = \
        getCutout_housekeeping_procedures(query_type, metadata, dataset_title, axes_ranges_original, xyzt_strides, var, timepoint_range_original)
    
    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_cardinality(metadata, var)
    # number of original datapoints along each axis specified by the user. used for checking that the user did not request
    # too much data and that result is filled correctly.
    axes_lengths_original = axes_ranges_original[:, 1] - axes_ranges_original[:, 0] + 1
    # number of original times queried by the user.
    num_times = ((timepoint_range_original[1] - timepoint_range_original[0]) // timepoint_stride) + 1
    # total number of datapoints, used for checking if the user requested too much data..
    num_datapoints = np.prod(axes_lengths_original) * num_times
    # total size of data, in GBs, requested by the user's box.
    requested_data_size = (num_datapoints * c['bytes_per_datapoint'] * num_values_per_datapoint) / float(1024**2)
    # maximum number of datapoints that can be read in. currently set to 16 GBs worth of datapoints.
    max_cutout_size = c['max_local_cutout_size']
    max_datapoints = int((max_cutout_size * (1024**2)) / (c['bytes_per_datapoint'] * float(num_values_per_datapoint)))

    # check the authorization token for larger queries.
    if auth_token == c['pyJHTDB_testing_token'] and num_datapoints > 4096:
        turb_email = c['turbulence_email_address']
        raise Exception(f'too many points requested for the testing authorization token: {num_datapoints} > 4096\n\n' + \
                        f'an authorization token can be requested by email from {turb_email}\n' + \
                        f' include your name, email address, institutional affiliation and department, together with a short description of your intended use of the database')
    
    if requested_data_size > max_cutout_size:
        raise ValueError(f'max local cutout size, {max_cutout_size} MB, exceeded. please specify a box with fewer than (xe - xs) * (ye - ys) * (ze - zs) = {max_datapoints + 1:,} ' + \
                         f'data points, regardless of strides.')
        
    if num_datapoints > 128**3:
        logging.warning(f'givernylocal will typically work for up to ~200^3 cube cutouts (~100 MB) depending on system load ' + \
                        'and internet connection speed, otherwise HTTP errors may result. for larger cutouts, use the giverny library getCutout function on SciServer')
    
    # placeholder values for getData settings.
    spatial_method = 'none'
    temporal_method = 'none'
    option = [-999.9, -999.9]
    # initialize cube constants. this is done so that all of the constants are known for pre-processing of the data. use the last timepoint
    # as a placeholder to keep consistent with how giverny on SciServer works, i.e. the last queried timepoint is the timepoint stored as the instance variable.
    cube.init_constants(query_type, var, var_offsets, timepoint_range[1], timepoint_range_original[1],
                        spatial_method, temporal_method, option, num_values_per_datapoint, c)
    
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)
    
    # request url.
    url = f'https://web.idies.jhu.edu/turbulence-svc/cutout/api/local?token={auth_token}' \
          f'&function={var}&dataset={dataset_title}' \
          f'&xs={axes_ranges_original[0, 0]}&xe={axes_ranges_original[0, 1]}' \
          f'&ys={axes_ranges_original[1, 0]}&ye={axes_ranges_original[1, 1]}' \
          f'&zs={axes_ranges_original[2, 0]}&ze={axes_ranges_original[2, 1]}' \
          f'&ts={timepoint_range_original[0]}&te={timepoint_range_original[1]}' \
          f'&stridet={timepoint_stride}&stridex={strides[0]}&stridey={strides[1]}&stridez={strides[2]}' \
          f'&filter_width={filter_width}'
    
    try:
        # send http get request.
        response = requests.get(url, timeout = 1000)
        # catch server side errors, e.g. server side timeout.
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        try:
            result = response.json()
            if 'description' in result:
                # join description list with newlines.
                description = result['description']
                description = '\n'.join(description) if isinstance(description, list) else description
                raise Exception(f"HTTP Error {response.status_code}:\n{description}")
            else:
                raise Exception(f"HTTP Error {response.status_code}.")
        except ValueError:
            # response isn't JSON.
            raise Exception(f"HTTP Error {response.status_code}.")
            
    # load the xarray dataset returned by giverny.
    json_data = json.loads(response.content)
    
    # result DataArray map.
    result_map = {}
    for dataset_name in json_data['data_vars']:
        # create a small placeholder array for error checking. a full pre-filled array is created in lJHTDB.getbigCutout (pyJHTDB datasets) and
        # getCutout_process_data (giverny datasets). initially the datatype is set to "f" (float) so that the array is filled with the
        # missing placeholder value (-999.9).
        result = np.array([c['missing_value_placeholder']], dtype = 'f')
        
        # load the data for the variable-time into a numpy array.
        result = np.array(json_data['data_vars'][dataset_name]['data'], dtype = 'f')
        
        # checks to make sure that data was received for all points.
        strided_lengths = (axes_lengths_original + strides - 1) // strides
        if c['missing_value_placeholder'] in result or result.shape != (strided_lengths[2], strided_lengths[1], strided_lengths[0], num_values_per_datapoint):
            raise Exception(f'result was not filled correctly for the "{dataset_name}" dataset')
            
        # create the xarray DataArray.
        result = xr.DataArray(data = result,
                              dims = json_data['data_vars'][dataset_name]['dims'])
        
        # aggregate the DataArrays into a dictionary.
        result_map[dataset_name] = result
    
    # parse the json data into the coords map. store the values as np.float64 for accuracy since the json data does not contain
    # the original data type information (mostly np.float32, but sometimes np.float64).
    coords_map = {k: np.array(v['data'], dtype = np.float64) for k, v in json_data['coords'].items()}
    
    # create the xarray Dataset.
    result = xr.Dataset(data_vars = result_map,
                        coords = coords_map, 
                        attrs = {'dataset':dataset_title, 't_start':timepoint_range_original[0], 't_end':timepoint_range_original[1], 't_step':timepoint_stride,
                                 'x_start':axes_ranges_original[0][0], 'y_start':axes_ranges_original[1][0], 'z_start':axes_ranges_original[2][0], 
                                 'x_end':axes_ranges_original[0][1], 'y_end':axes_ranges_original[1][1], 'z_end':axes_ranges_original[2][1],
                                 'x_step':strides[0], 'y_step':strides[1], 'z_step':strides[2],
                                 'filterWidth':filter_width})
    
    # -----
    end_time = time.perf_counter()
    
    if verbose:
        print(f'\ntotal time elapsed = {end_time - start_time:0.3f} seconds ({(end_time - start_time) / 60:0.3f} minutes)')
        sys.stdout.flush()

        print('\nquery completed successfully.\n' + '-' * 5)
        sys.stdout.flush()
    
    # closing the tracemalloc library.
    if trace_memory:
        # memory used during processing as calculated by tracemalloc.
        tracemem_end = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_end = tracemalloc.get_tracemalloc_memory() / (1024**3)
        # stopping the tracemalloc library.
        tracemalloc.stop()

        # see how much memory was used during processing.
        # memory used at program start.
        print(f'\nstarting memory used in GBs [current, peak] = {tracemem_start}')
        # memory used by tracemalloc.
        print(f'starting memory used by tracemalloc in GBs = {tracemem_used_start}')
        # memory used during processing.
        print(f'ending memory used in GBs [current, peak] = {tracemem_end}')
        # memory used by tracemalloc.
        print(f'ending memory used by tracemalloc in GBs = {tracemem_used_end}')
    
    return result

def getCutout_housekeeping_procedures(query_type, metadata, dataset_title, axes_ranges_original, xyzt_strides, var, timepoint_range_original):
    """
    complete all of the getCutout housekeeping procedures before data processing.
    """
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(metadata, var, dataset_title, query_type)
    # check that the user-input timepoint is a valid timepoint for the dataset.
    check_timepoint(metadata, timepoint_range_original, dataset_title, query_type)
    # check that the user-input x-, y-, and z-axis ranges are all specified correctly as [minimum, maximum] integer values.
    check_axes_ranges(metadata, axes_ranges_original, dataset_title, var)
    # check that the user-input strides are all positive integers.
    check_strides(xyzt_strides)
    
    # pre-processing steps.
    # -----
    # convert the original input timepoint to the correct time index.
    timepoint_range = get_time_index_from_timepoint(metadata, dataset_title, timepoint_range_original, tint = 'none', query_type = query_type)
    
    # set var_offsets to var for getCutout. 'velocity' is handled differently in getData for the 'sabl2048low', 'sabl2048high', 'stsabl2048low', and 'stsabl2048high' datasets.
    if dataset_title in ['sabl2048low', 'sabl2048high', 'stsabl2048low', 'stsabl2048high'] and var == 'velocity':
        # temporary placeholder value to initialize the dataset constants.
        var_offsets = var + '_uv'
    else:
        var_offsets = var
    
    return (var_offsets, timepoint_range)

def getData(cube, var, timepoint_original, temporal_method, spatial_method_original, spatial_operator, points,
            option = [-999.9, -999.9],
            return_times = False, trace_memory = False, verbose = True):
    """
    interpolate/differentiate the variable for the specified points from the various JHTDB datasets.
    """
    if verbose:
        print('\n' + '-' * 5 + '\ngetData is processing...')
        sys.stdout.flush()
    
    # calculate how much time it takes to run the code.
    start_time = time.perf_counter()
    
    # set cube attributes.
    metadata = cube.metadata
    dataset_title = cube.dataset_title
    auth_token = cube.auth_token
    
    # define the query type.
    query_type = 'getdata'
    
    # data constants.
    c = metadata['constants']
    
    # -----
    # housekeeping procedures. will handle multiple variables, e.g. 'pressure' and 'velocity'.
    var_offsets, timepoint, spatial_method = \
        getData_housekeeping_procedures(query_type, metadata, dataset_title, points, var, timepoint_original,
                                        temporal_method, spatial_method_original, spatial_operator,
                                        option, c)
    
    # check the authorization token for larger queries.
    if auth_token == c['pyJHTDB_testing_token'] and len(points) > 4096:
        turb_email = c['turbulence_email_address']
        raise Exception(f'too many points requested for the testing authorization token: {len(points)} > 4096\n\n' + \
                        f'an authorization token can be requested by email from {turb_email}\n' + \
                        f' include your name, email address, institutional affiliation and department, together with a short description of your intended use of the database')
    
    # option parameter values.
    timepoint_end, delta_t = option
    
    # default timepoint range which only queries the first timepoint for non-'position' variables and non-time series queries. in the givernylocal code
    # this is only used to verify the integrity of the results retrieved through the rest service.
    timepoint_range = np.arange(timepoint_original, timepoint_original + 1, 2)
    if var != 'position' and option != [-999.9, -999.9]:
        # timepoint range for the time series queries.
        timepoint_range = np.arange(timepoint_original, timepoint_end, delta_t)
        
        # add in the last timepoint if the final timepoint in the range is delta_t less than timepoint_end. np.arange is not good at handling
        # floating point step sizes.
        if math.isclose(timepoint_range[-1] + delta_t, timepoint_end, rel_tol = 10**-9, abs_tol = 0.0):
            timepoint_range = np.append(timepoint_range, timepoint_end)
            
    num_timepoints = len(timepoint_range)
    # if more than one timepoint was queried, then checks if ({number of points} * {number of timepoints}) <= c['max_data_points'].
    if (len(points) * num_timepoints) > c['max_data_points']:
        raise Exception(f"too many 'points' and 'times' queried together, please limit the number of (points * times) to <= {c['max_data_points']:,}")
    
    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_cardinality(metadata, var)
    # initialize cube constants. this is done so that all of the constants are known for pre-processing of the data.
    cube.init_constants(query_type, var, var_offsets, timepoint, timepoint_original,
                        spatial_method, temporal_method, option, num_values_per_datapoint, c)
    
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)
    
    # pre-fill the result array that will be filled with the data that is read in. initially the datatype is set to "f" (float)
    # so that the array is filled with the missing placeholder value (-999.9).
    result = np.array([c['missing_value_placeholder']], dtype = 'f')
    
    # convert points array to a string.
    request_points = "\n".join(["\t".join(["%.8f" % coord for coord in point]) for point in points])

    # request url.
    url = f'https://web.idies.jhu.edu/turbulence-svc/values?authToken={auth_token}&dataset={dataset_title}&function=GetVariable&var={var}' \
          f'&t={timepoint_original}&sint={spatial_method_original}&sop={spatial_operator}&tint={temporal_method}' \
          f'&timepoint_end={timepoint_end}&delta_t={delta_t}'

    try:
        # send http post request.
        response = requests.post(url, data = request_points, timeout = 1000)
        # catch server side errors, e.g. server side timeout.
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        try:
            result = response.json()
            if 'description' in result:
                # join description list with newlines.
                description = result['description']
                description = '\n'.join(description) if isinstance(description, list) else description
                raise Exception(f"HTTP Error {response.status_code}:\n{description}")
            else:
                raise Exception(f"HTTP Error {response.status_code}.")
        except ValueError:
            # response isn't JSON.
            raise Exception(f"HTTP Error {response.status_code}.")
    
    # convert the response string to a numpy array.
    result = np.array(json.loads(response.text), dtype = np.float32)
    
    # get the result header, which only contains the names for each column of the data values.
    output_header = get_interpolation_tsv_header(metadata, cube.dataset_title, cube.var, cube.timepoint_original, cube.timepoint_end, cube.delta_t, cube.sint, cube.tint)
    result_header = np.array(output_header.split('\n')[1].strip().split('\t'))[3:]
    
    # array lengths.
    points_len = len(points)
    timepoint_range_len = len(timepoint_range)
    result_header_len = len(result_header)
    
    # checks to make sure that data was read in for all points.
    if c['missing_value_placeholder'] in result or result.shape != (points_len * timepoint_range_len, result_header_len):
        raise Exception(f'result was not filled correctly')
    
    # insert the output header at the beginning of the result for each timepoint.
    result = result.reshape((timepoint_range_len, points_len, result_header_len))
    results = []
    for result_array in result:
        df = pd.DataFrame(data = result_array, columns = result_header)
        # give the index column a name for each dataframe in results.
        df.index.name = 'index'
        results.append(df)
    
    # -----
    end_time = time.perf_counter()
    
    if verbose:
        print(f'\ntotal time elapsed = {end_time - start_time:0.3f} seconds ({(end_time - start_time) / 60:0.3f} minutes)')
        sys.stdout.flush()

        print('\nquery completed successfully.\n' + '-' * 5)
        sys.stdout.flush()
    
    # closing the tracemalloc library.
    if trace_memory:
        # memory used during processing as calculated by tracemalloc.
        tracemem_end = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_end = tracemalloc.get_tracemalloc_memory() / (1024**3)
        # stopping the tracemalloc library.
        tracemalloc.stop()

        # see how much memory was used during processing.
        # memory used at program start.
        print(f'\nstarting memory used in GBs [current, peak] = {tracemem_start}')
        # memory used by tracemalloc.
        print(f'starting memory used by tracemalloc in GBs = {tracemem_used_start}')
        # memory used during processing.
        print(f'ending memory used in GBs [current, peak] = {tracemem_end}')
        # memory used by tracemalloc.
        print(f'ending memory used by tracemalloc in GBs = {tracemem_used_end}')
    
    if not return_times:
        return results
    else:
        return results, timepoint_range

def getData_housekeeping_procedures(query_type, metadata, dataset_title, points, var, timepoint_original,
                                    temporal_method, spatial_method, spatial_operator,
                                    option, c):
    """
    complete all of the getData housekeeping procedures before data processing.
    """
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(metadata, var, dataset_title, query_type)
    # check that not too many points were queried and the points are all within axes domain for the dataset.
    check_points(metadata, points, dataset_title, var, c['max_data_points'])
    # check how many chunks the queried points intersect.
    # check_points_chunks_intersection(metadata, points, dataset_title, var)
    # check that the user-input timepoint is a valid timepoint for the dataset.
    check_timepoint(metadata, timepoint_original, dataset_title, query_type)
    # check that the user-input interpolation spatial operator (spatial_operator) is a valid interpolation operator.
    check_spatial_operator(metadata, spatial_operator, dataset_title, var)
    # check that the user-input spatial interpolation (spatial_method) is a valid spatial interpolation method.
    spatial_method = check_spatial_method(metadata, spatial_method, dataset_title, var, spatial_operator)
    # check that the user-input temporal interpolation (temporal_method) is a valid temporal interpolation method.
    check_temporal_method(metadata, temporal_method, dataset_title, var)
    # check that option parameters are valid if specified (applies to getPosition and time series queries).
    if var == 'position' or option != [-999.9, -999.9]:
        check_option_parameter(metadata, option, dataset_title, timepoint_original)
        
        # check that the user-input ending timepoint for 'position' is a valid timepoint for this dataset.
        timepoint_end = option[0]
        check_timepoint(metadata, timepoint_end, dataset_title, query_type)
    
    # pre-processing steps.
    # -----
    # convert the original input timepoint to the correct time index.
    timepoint = get_time_index_from_timepoint(metadata, dataset_title, timepoint_original, temporal_method, query_type)
    
    # set var_offsets to var. 'velocity' is handled differently for the 'sabl2048low', 'sabl2048high', 'stsabl2048low', and 'stsabl2048high' datasets.
    if dataset_title in ['sabl2048low', 'sabl2048high', 'stsabl2048low', 'stsabl2048high'] and var == 'velocity':
        # temporary placeholder value to initialize the dataset constants.
        var_offsets = var + '_uv'
    else:
        var_offsets = var
    
    return (var_offsets, timepoint, spatial_method)

def getTurbineData(cube, turbine_numbers, var, original_times,
                   trace_memory = False, verbose = True):
    """
    retrieve turbine data at a set of specified times for the specified turbine and variable.
    """
    if verbose:
        print('\n' + '-' * 5 + '\ngetTurbineData is processing...')
        sys.stdout.flush()
    
    # calculate how much time it takes to run the code.
    start_time = time.perf_counter()
    
    # set cube attributes.
    metadata = cube.metadata
    dataset_title = cube.dataset_title
    auth_token = cube.auth_token
    
    # define the query type.
    query_type = 'getturbinedata'
    
    # data constants.
    c = metadata['constants']
    
    # -----
    # housekeeping procedures.
    turbine_numbers = getTurbineData_housekeeping_procedures(query_type, metadata, dataset_title, var, original_times, turbine_numbers, c)
    
    # number of queried times.
    num_original_times = len(original_times)
    # check the authorization token for larger queries.
    if auth_token == c['pyJHTDB_testing_token'] and num_original_times > 4096:
        turb_email = c['turbulence_email_address']
        raise Exception(f'too many times requested for the testing authorization token: {num_original_times} > 4096\n\n' + \
                        f'an authorization token can be requested by email from {turb_email}\n' + \
                        f' include your name, email address, institutional affiliation and department, together with a short description of your intended use of the database')
    
    # number of queried turbines.
    num_turbines = len(turbine_numbers)
    # if more than one turbine was queried, then checks if ({number of times} * {number of turbines}) <= c['max_data_points'].
    if (num_original_times * num_turbines) > c['max_data_points']:
        raise Exception(f"too many 'times' and 'turbines' queried together, please limit the number of (times * turbines) to <= {c['max_data_points']:,}")
    
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)
    
    # dictionary of the request data.
    request_data = {
        "auth_token": auth_token,
        "dataset_title": dataset_title,
        "turbine_variable": var,
        "turbines": turbine_numbers.tolist(),
        "turbine_times": np.array(original_times, dtype = np.float64).tolist()
    }

    # convert to json string.
    json_data = json.dumps(request_data)

    try:
        # send http post request.
        response = requests.post(
            "https://web.idies.jhu.edu/turbulence-svc/turbine?include_metadata=0", 
            headers = {"Content-Type": "application/json"},
            data = json_data,
            timeout = 1000
        )
        
        # catch server side errors, e.g. server side timeout.
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        try:
            result = response.json()
            if 'description' in result:
                # join description list with newlines.
                description = result['description']
                description = '\n'.join(description) if isinstance(description, list) else description
                raise Exception(f"HTTP Error {response.status_code}:\n{description}")
            else:
                raise Exception(f"HTTP Error {response.status_code}.")
        except ValueError:
            # response isn't JSON.
            raise Exception(f"HTTP Error {response.status_code}.")
    
    # convert the response string to a pandas dataframe.
    column_names = ['time', 'turbine', var]
    result = pd.DataFrame(json.loads(response.text), columns = column_names)
    result['turbine'] = result['turbine'].astype(int)
    # sort by 'turbine', and then 'time' columns.
    result = result.sort_values(by = ['turbine', 'time']).reset_index(drop = True)
    # reset the indices for each turbine.
    reset_indices = np.arange(len(result)) % num_original_times
    result.index = reset_indices
    result.index.name = 'index'
    
    # -----
    end_time = time.perf_counter()
    
    if verbose:
        print(f'\ntotal time elapsed = {end_time - start_time:0.3f} seconds ({(end_time - start_time) / 60:0.3f} minutes)')
        sys.stdout.flush()

        print('\nquery completed successfully.\n' + '-' * 5)
        sys.stdout.flush()
    
    # closing the tracemalloc library.
    if trace_memory:
        # memory used during processing as calculated by tracemalloc.
        tracemem_end = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_end = tracemalloc.get_tracemalloc_memory() / (1024**3)
        # stopping the tracemalloc library.
        tracemalloc.stop()

        # see how much memory was used during processing.
        # memory used at program start.
        print(f'\nstarting memory used in GBs [current, peak] = {tracemem_start}')
        # memory used by tracemalloc.
        print(f'starting memory used by tracemalloc in GBs = {tracemem_used_start}')
        # memory used during processing.
        print(f'ending memory used in GBs [current, peak] = {tracemem_end}')
        # memory used by tracemalloc.
        print(f'ending memory used by tracemalloc in GBs = {tracemem_used_end}')
    
    return result

def getTurbineData_housekeeping_procedures(query_type, metadata, dataset_title, var, times, turbine_numbers, c):
    """
    complete all of the getTurbineData housekeeping procedures before data processing.
    """
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(metadata, var, dataset_title, query_type)
    # check that the user-input times are valid times for the dataset.
    check_timepoint(metadata, times, dataset_title, query_type, max_num_timepoints = c['max_data_points'])
    # check that the user-input turbine numbers are valid turbines.
    turbine_numbers = check_turbine_numbers(metadata, dataset_title, turbine_numbers)
    
    return turbine_numbers

def getBladeData(cube, turbine_numbers, blade_numbers, var, original_times, blade_points,
                 trace_memory = False, verbose = True):
    """
    retrieve blade data at a set of specified times and blade actuator points for the specified turbine, blade, and variable.
    """
    if verbose:
        print('\n' + '-' * 5 + '\ngetBladeData is processing...')
        sys.stdout.flush()
    
    # calculate how much time it takes to run the code.
    start_time = time.perf_counter()
    
    # set cube attributes.
    metadata = cube.metadata
    dataset_title = cube.dataset_title
    auth_token = cube.auth_token
    
    # define the query type.
    query_type = 'getbladedata'
    
    # data constants.
    c = metadata['constants']
    
    # -----
    # housekeeping procedures.
    turbine_numbers, blade_numbers, blade_points = \
        getBladeData_housekeeping_procedures(query_type, metadata, dataset_title, var, original_times, turbine_numbers, blade_numbers, blade_points, c)
    
    # number of queried times.
    num_original_times = len(original_times)
    # check the authorization token for larger queries.
    if auth_token == c['pyJHTDB_testing_token'] and num_original_times > 4096:
        turb_email = c['turbulence_email_address']
        raise Exception(f'too many times requested for the testing authorization token: {num_original_times} > 4096\n\n' + \
                        f'an authorization token can be requested by email from {turb_email}\n' + \
                        f' include your name, email address, institutional affiliation and department, together with a short description of your intended use of the database')
    
    # number of queried turbines.
    num_turbines = len(turbine_numbers)
    num_blades = len(blade_numbers)
    # if more than one turbine and/or blade was queried, then checks if ({number of times} * {number of turbines} * {number of blades}) <= c['max_data_points'].
    if (num_original_times * num_turbines * num_blades) > c['max_data_points']:
        raise Exception(f"too many 'times', 'turbines', and 'blades' queried together, please limit the number of (times * turbines * blades) to <= {c['max_data_points']:,}")
    
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)
    
    # dictionary of the request data.
    request_data = {
        "auth_token": auth_token,
        "dataset_title": dataset_title,
        "blade_variable": var,
        "turbines": turbine_numbers.tolist(),
        "blades": blade_numbers.tolist(),
        "blade_times": np.array(original_times, dtype = np.float64).tolist(),
        "blade_actuator_points": blade_points.tolist()
    }

    # convert to json string.
    json_data = json.dumps(request_data)

    try:
        # send http post request.
        response = requests.post(
            "https://web.idies.jhu.edu/turbulence-svc/blade?include_metadata=0", 
            headers = {"Content-Type": "application/json"},
            data = json_data,
            timeout = 1000
        )
        
        # catch server side errors, e.g. server side timeout.
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        try:
            result = response.json()
            if 'description' in result:
                # join description list with newlines.
                description = result['description']
                description = '\n'.join(description) if isinstance(description, list) else description
                raise Exception(f"HTTP Error {response.status_code}:\n{description}")
            else:
                raise Exception(f"HTTP Error {response.status_code}.")
        except ValueError:
            # response isn't JSON.
            raise Exception(f"HTTP Error {response.status_code}.")
    
    # convert the response string to a pandas dataframe.
    column_names = ['time', 'turbine', 'blade'] + [f'{var}_{actuator_point}' for actuator_point in blade_points]
    result = pd.DataFrame(json.loads(response.text), columns = column_names)
    result['turbine'] = result['turbine'].astype(int)
    result['blade'] = result['blade'].astype(int)
    # sort by 'turbine', 'blade', and then 'time' columns.
    result = result.sort_values(by = ['turbine', 'blade', 'time']).reset_index(drop = True)
    # reset the indices for each turbine.
    reset_indices = np.arange(len(result)) % num_original_times
    result.index = reset_indices
    result.index.name = 'index'
    
    # -----
    end_time = time.perf_counter()
    
    if verbose:
        print(f'\ntotal time elapsed = {end_time - start_time:0.3f} seconds ({(end_time - start_time) / 60:0.3f} minutes)')
        sys.stdout.flush()

        print('\nquery completed successfully.\n' + '-' * 5)
        sys.stdout.flush()
    
    # closing the tracemalloc library.
    if trace_memory:
        # memory used during processing as calculated by tracemalloc.
        tracemem_end = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_end = tracemalloc.get_tracemalloc_memory() / (1024**3)
        # stopping the tracemalloc library.
        tracemalloc.stop()

        # see how much memory was used during processing.
        # memory used at program start.
        print(f'\nstarting memory used in GBs [current, peak] = {tracemem_start}')
        # memory used by tracemalloc.
        print(f'starting memory used by tracemalloc in GBs = {tracemem_used_start}')
        # memory used during processing.
        print(f'ending memory used in GBs [current, peak] = {tracemem_end}')
        # memory used by tracemalloc.
        print(f'ending memory used by tracemalloc in GBs = {tracemem_used_end}')
    
    return result

def getBladeData_housekeeping_procedures(query_type, metadata, dataset_title, var, times, turbine_numbers, blade_numbers, blade_points, c):
    """
    complete all of the getBladeData housekeeping procedures before data processing.
    """
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(metadata, var, dataset_title, query_type)
    # check that the user-input times are valid times for the dataset.
    check_timepoint(metadata, times, dataset_title, query_type, max_num_timepoints = c['max_data_points'])
    # check that the user-input turbine numbers are valid turbines.
    turbine_numbers = check_turbine_numbers(metadata, dataset_title, turbine_numbers)
    # check that the user-input blade numbers are valid blades.
    blade_numbers = check_blade_numbers(metadata, dataset_title, blade_numbers)
    # check that the user-input blade actuator points are valid blade points.
    blade_points = check_blade_points(metadata, dataset_title, blade_points)
    
    return turbine_numbers, blade_numbers, blade_points

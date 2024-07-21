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
import requests
import tracemalloc
import numpy as np
import pandas as pd
from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_gizmos.basic_gizmos import *
from givernylocal.turbulence_gizmos.constants import get_constants

def getData(cube, var_original, timepoint_original, temporal_method_original, spatial_method_original, spatial_operator_original, points,
            option = [-999.9, -999.9],
            trace_memory = False, verbose = True):
    """
    interpolate/differentiate the variable for the specified points from the various JHTDB datasets.
    """
    if verbose:
        print('\n' + '-' * 5 + '\ngetData is processing...')
        sys.stdout.flush()
    
    # calculate how much time it takes to run the code.
    start_time = time.perf_counter()
    
    # set cube attributes.
    dataset_title = cube.dataset_title
    auth_token = cube.auth_token
    
    # define the query type.
    query_type = 'getdata'
    
    # data constants.
    c = get_constants()
    
    # -----
    # housekeeping procedures. will handle multiple variables, e.g. 'pressure' and 'velocity'.
    points, var, var_offsets, timepoint, temporal_method, spatial_method, spatial_method_specified, spatial_operator, datatype = \
        getData_housekeeping_procedures(query_type, dataset_title, points, var_original, timepoint_original,
                                        temporal_method_original, spatial_method_original, spatial_operator_original,
                                        option, c)
    
    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)
    # initialize cube constants. this is done so that all of the constants are known for pre-processing of the data.
    cube.init_constants(query_type, var, var_original, var_offsets, timepoint, timepoint_original,
                        spatial_method, spatial_method_specified, temporal_method, option, num_values_per_datapoint, c)
    
    # option parameter values.
    timepoint_end, delta_t = option
    
    # default timepoint range which only queries the first timepoint for non-'position' variables and non-time series queries. in the givernylocal code
    # this is only used to verify the integrity of the results retrieved through the rest service.
    timepoint_range = np.arange(timepoint_original, timepoint_original + 1, 2)
    if var_original != 'position' and option != [-999.9, -999.9]:
        # timepoint range for the time series queries.
        timepoint_range = np.arange(timepoint_original, timepoint_end, delta_t)
        
        # add in the last timepoint if the final timepoint in the range is delta_t less than timepoint_end. np.arange is not good at handling
        # floating point step sizes.
        if math.isclose(timepoint_range[-1] + delta_t, timepoint_end, rel_tol = 10**-9, abs_tol = 0.0):
            timepoint_range = np.append(timepoint_range, timepoint_end)
    
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
    url = f'https://web.idies.jhu.edu/turbulence-svc-test/values?authToken={auth_token}&dataset={dataset_title}&function=GetVariable&var={var_original}' \
          f'&t={timepoint_original}&sint={spatial_method_original}&sop={spatial_operator_original}&tint={temporal_method_original}' \
          f'&timepoint_end={timepoint_end}&delta_t={delta_t}'

    # send http post request.
    response = requests.post(url, data = request_points, timeout = 1000)
    
    # convert the response string to a numpy array.
    result = np.array(json.loads(response.text), dtype = np.float32)
    
    # get the output header.
    output_header = get_interpolation_tsv_header(cube.dataset_title, cube.var_name, cube.timepoint_original, cube.timepoint_end, cube.delta_t, cube.sint, cube.tint)
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
    results = [pd.DataFrame(data = result_array, columns = result_header) for result_array in result]
    
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
    
    return results

def getData_housekeeping_procedures(query_type, dataset_title, points, var_original, timepoint_original,
                                    temporal_method, spatial_method, spatial_operator,
                                    option, c):
    """
    complete all of the housekeeping procedures before data processing.
        - format the variable name and get the variable identifier.
        - convert 1-based timepoint to 0-based.
    """
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(var_original, dataset_title, query_type)
    # check that not too many points were queried and the points are all within axes domain for the dataset.
    check_points(dataset_title, points, c['max_data_points'])
    # check that the user-input timepoint is a valid timepoint for the dataset.
    check_timepoint(timepoint_original, dataset_title, query_type)
    # check that the user-input interpolation spatial operator (spatial_operator) is a valid interpolation operator.
    check_operator(spatial_operator, var_original)
    # check that the user-input spatial interpolation (spatial_method) is a valid spatial interpolation method.
    spatial_method = check_spatial_interpolation(dataset_title, var_original, spatial_method, spatial_operator)
    # check that the user-input temporal interpolation (temporal_method) is a valid temporal interpolation method.
    check_temporal_interpolation(dataset_title, var_original, temporal_method)
    # check that option parameters are valid if specified (applies to getPosition and time series queries).
    if var_original == 'position' or option != [-999.9, -999.9]:
        check_option_parameter(option, dataset_title, timepoint_original)
        
        # check that the user-input ending timepoint for 'position' is a valid timepoint for this dataset.
        timepoint_end = option[0]
        check_timepoint(timepoint_end, dataset_title, query_type)
    
    # pre-processing steps.
    # -----
    # convert the variable name from var_original into a variable identifier.
    var = get_variable_identifier(var_original)
    
    # convert the original input timepoint to the correct time index.
    timepoint = get_time_index_from_timepoint(dataset_title, timepoint_original, temporal_method, query_type)
        
    # set var_offsets to var_original.
    var_offsets = var_original
    
    # copy of the spatial interpolation that was specified by the user. needed for the 'sabl_linear*' step-down interpolation methods for the 'sabl2048*' datasets.
    spatial_method_specified = spatial_method
    
    # get the full variable name for determining the datatype.
    datatype_var = get_output_variable_name(var_original)
    
    # remove 'field' from operator for determining the datatype.
    datatype_operator = spatial_operator if spatial_operator != 'field' else ''
    
    # define datatype from the datatype_var and datatype_operator variables.
    datatype = f'{datatype_var}{datatype_operator.title()}'
    
    return (points, var, var_offsets, timepoint, temporal_method, spatial_method, spatial_method_specified, spatial_operator, datatype)

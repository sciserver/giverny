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
import xarray as xr
from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_gizmos.basic_gizmos import *
from givernylocal.turbulence_gizmos.constants import get_constants

def getCutout(cube, var_original, timepoint_original, axes_ranges_original, strides,
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
    dataset_title = cube.dataset_title
    auth_token = cube.auth_token
    
    # define the query type.
    query_type = 'getcutout'
    
    # data constants.
    c = get_constants()
    
    # only time_step and filter_width values of 1 are currently allowed.
    time_step = 1
    filter_width = 1
    
    # field (variable) map for legacy datasets.
    field_map = {
        'velocity': 'u',
        'vectorpotential': 'a',
        'magneticfield': 'b',
        'pressure': 'p',
        'density': 'd',
        'temperature': 't'
    }
    
    # retrieve the list of datasets processed by the giverny code.
    giverny_datasets = get_giverny_datasets()
    
    # housekeeping procedures.
    # -----
    var, var_offsets, axes_ranges, timepoint = \
        getCutout_housekeeping_procedures(query_type, dataset_title, axes_ranges_original, strides, var_original, timepoint_original)
    
    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)
    # number of original datapoints along each axis specified by the user. used for checking that the user did not request
    # too much data and that result is filled correctly.
    axes_lengths_original = axes_ranges_original[:, 1] - axes_ranges_original[:, 0] + 1
    # total number of datapoints, used for checking if the user requested too much data..
    num_datapoints = np.prod(axes_lengths_original)
    # total size of data, in GBs, requested by the user's box.
    requested_data_size = (num_datapoints * c['bytes_per_datapoint'] * num_values_per_datapoint) / float(1024**3)
    # maximum number of datapoints that can be read in. currently set to 16 GBs worth of datapoints.
    max_cutout_size = c['max_cutout_size']
    max_datapoints = int((max_cutout_size * (1024**3)) / (c['bytes_per_datapoint'] * float(num_values_per_datapoint)))

    if requested_data_size > max_cutout_size:
        raise ValueError(f'max cutout size, {max_cutout_size} GB, exceeded. please specify a box with fewer than {max_datapoints + 1:,} data points.')
        
    # placeholder values for getData settings.
    spatial_method = 'none'
    spatial_method_specified = 'none'
    temporal_method = 'none'
    option = ['none', 'none']
    # initialize cube constants. this is done so that all of the constants are known for pre-processing of the data.
    cube.init_constants(query_type, var, var_original, var_offsets, timepoint, timepoint_original,
                        spatial_method, spatial_method_specified, temporal_method, option, num_values_per_datapoint, c)
    
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)
    
    # create a small placeholder array for error checking. a full pre-filled array is created in lJHTDB.getbigCutout (pyJHTDB datasets) and
    # getCutout_process_data (giverny datasets). initially the datatype is set to "f" (float) so that the array is filled with the
    # missing placeholder value (-999.9).
    result = np.array([c['missing_value_placeholder']], dtype = 'f')
    
    # process the data query, retrieve a cutout for the various datasets.
    if dataset_title in giverny_datasets:
        """
        get the results for the datasets processed by giverny.
        """
        # parse the database files, generate the result matrix.
        result = getCutout_process_data(cube, axes_ranges, var, timepoint,
                                        axes_ranges_original, strides, var_original, var_offsets, timepoint_original, c)
    else:
        """
        get the results for the legacy datasets processed by pyJHTDB.
        """
        # initialize lJHTDB gSOAP resources and add the user's authorization token.
        if auth_token == c['pyJHTDB_testing_token'] and num_datapoints > 4096:
            turb_email = c['turbulence_email_address']
            raise Exception(f'too many points requested for the testing authorization token: {num_datapoints} > 4096\n\n' + \
                            f'an authorization token can be requested by email from {turb_email}\n' + \
                            f' include your name, email address, institutional affiliation and department, together with a short description of your intended use of the database')
        
        lJHTDB = pyJHTDB.libJHTDB(auth_token = auth_token)
        lJHTDB.initialize()
        
        # get the field (variable) integer for the legacy datasets.
        field = field_map[var_original]
        
        result = lJHTDB.getbigCutout(data_set = dataset_title, fields = field, t_start = timepoint_original, t_end = timepoint_original, t_step = time_step,
                                     start = np.array([axes_ranges[0, 0], axes_ranges[1, 0], axes_ranges[2, 0]], dtype = int),
                                     end = np.array([axes_ranges[0, 1], axes_ranges[1, 1], axes_ranges[2, 1]], dtype = int),
                                     step = np.array([strides[0], strides[1], strides[2]], dtype = int),
                                     filter_width = filter_width)
    
        # free up gSOAP resources.
        lJHTDB.finalize()
        
    # determines how many copies of data need to be me made along each axis when the number of datapoints the user specified
    # exceeds the cube resolution (cube.N). note: no copies of the data values should be made, hence data_value_multiplier equals 1.
    axes_multipliers = np.ceil(axes_lengths_original / cube.N).astype(int)
    data_value_multiplier = 1
    
    # duplicates the data along the z-, y-, and x-axes of output_data if the the user asked for more datapoints than the cube resolution along any axis.
    if np.any(axes_multipliers > 1):
        result = np.tile(result, (axes_multipliers[2], axes_multipliers[1], axes_multipliers[0], data_value_multiplier))
        # truncate any extra datapoints from the duplicate data outside of the original range of the datapoints specified by the user.
        result = np.copy(result[0 : axes_lengths_original[2], 0 : axes_lengths_original[1], 0 : axes_lengths_original[0], :])
    
    # checks to make sure that data was read in for all points.
    if c['missing_value_placeholder'] in result or result.shape != (axes_lengths_original[2], axes_lengths_original[1], axes_lengths_original[0], num_values_per_datapoint):
        raise Exception(f'result was not filled correctly')
        
    # datasets that have an irregular y-grid.
    irregular_ygrid_datasets = get_irregular_mesh_ygrid_datasets()
        
    # create axis coordinate ranges, shifted to 0-based indices, to store in the xarray metadata.
    z_coords = np.around(np.arange(axes_ranges_original[2][0] - 1, axes_ranges_original[2][1], strides[2], dtype = np.float32) * cube.dz, decimals = c['decimals'])
    z_coords += cube.coor_offsets[2]
    if dataset_title in irregular_ygrid_datasets:
        # note: this assumes that the y-axis of the irregular grid datasets is non-periodic.
        y_coords = cube.dy[np.arange(axes_ranges_original[1][0] - 1, axes_ranges_original[1][1], strides[1])]
    else:
        y_coords = np.around(np.arange(axes_ranges_original[1][0] - 1, axes_ranges_original[1][1], strides[1], dtype = np.float32) * cube.dy, decimals = c['decimals'])
        y_coords += cube.coor_offsets[1]
    x_coords = np.around(np.arange(axes_ranges_original[0][0] - 1, axes_ranges_original[0][1], strides[0], dtype = np.float32) * cube.dx, decimals = c['decimals'])
    x_coords += cube.coor_offsets[0]
    
    # set the dataset name to be used in the hdf5 file.
    h5_dataset_name = cube.dataset_name
    
    if dataset_title in ['sabl2048low', 'sabl2048high'] and var_original == 'velocity':
        # zcoor_uv are the default z-axis coordinates for the 'velocity' variable of the 'sabl' datasets.
        coords_map = {'zcoor_uv':z_coords, 'zcoor_w':z_coords + (0.1953125 / 2), 'ycoor':y_coords, 'xcoor':x_coords}
        dims_list = ['zcoor_uv', 'ycoor', 'xcoor', 'values']
    else:
        coords_map = {'zcoor':z_coords, 'ycoor':y_coords, 'xcoor':x_coords}
        dims_list = ['zcoor', 'ycoor', 'xcoor', 'values']
        
    # apply the strides to output_data.
    result = xr.DataArray(data = result[::strides[2], ::strides[1], ::strides[0], :],
                          dims = dims_list)
    
    result = xr.Dataset(data_vars = {h5_dataset_name:result},
                        coords = coords_map, 
                        attrs = {'dataset':dataset_title, 't_start':timepoint_original, 't_end':timepoint_original, 't_step':time_step,
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

def getCutout_housekeeping_procedures(query_type, dataset_title, axes_ranges_original, strides, var_original, timepoint_original):
    """
    complete all of the getCutout housekeeping procedures before data processing.
        - convert 1-based axes ranges to 0-based.
        - format the variable name and get the variable identifier.
        - convert 1-based timepoint to 0-based.
    """
    # validate user-input.
    # -----
    # check that the user-input variable is a valid variable name.
    check_variable(var_original, dataset_title, query_type)
    # check that the user-input timepoint is a valid timepoint for the dataset.
    check_timepoint(timepoint_original, dataset_title, query_type)
    # check that the user-input x-, y-, and z-axis ranges are all specified correctly as [minimum, maximum] integer values.
    check_axes_ranges(dataset_title, axes_ranges_original)
    # check that the user-input strides are all positive integers.
    check_strides(strides)
    
    # pre-processing steps.
    # -----
    # converts the 1-based axes ranges above to 0-based axes ranges, and truncates the ranges if they are longer than 
    # the cube resolution (N) since the boundaries are periodic. result will be filled in with the duplicate data 
    # for the truncated data points after processing so that the data files are not read redundantly.
    axes_ranges = convert_to_0_based_ranges(dataset_title, axes_ranges_original)

    # convert the variable name from var_original into a variable identifier.
    var = get_variable_identifier(var_original)
    
    # convert the original input timepoint to the correct time index.
    timepoint = get_time_index_from_timepoint(dataset_title, timepoint_original, tint = 'none', query_type = query_type)
    
    # set var_offsets to var_original for getCutout. 'velocity' is handled differently in getData for the 'sabl2048low' and 'sabl2048high' datasets.
    var_offsets = var_original
    
    return (var, var_offsets, axes_ranges, timepoint)

def getData(cube, var_original, timepoint_original, temporal_method_original, spatial_method_original, spatial_operator_original, points,
            option = ['none', 'none'],
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
          f'&t={timepoint_original}&sint={spatial_method_original}&sop={spatial_operator_original}&tint={temporal_method_original}'

    # send http post request.
    response = requests.post(url, data = request_points, timeout = 1000)
    
    # convert the response string to a numpy array.
    result = np.array(json.loads(response.text), dtype = np.float32)
    
    # get the output header.
    output_header = get_interpolation_tsv_header(cube.dataset_title, cube.var_name, cube.timepoint_original, cube.timepoint_end, cube.delta_t, cube.sint, cube.tint)
    result_header = np.array(output_header.split('\n')[1].strip().split('\t'))[3:]
    
    # checks to make sure that data was read in for all points.
    if c['missing_value_placeholder'] in result or result.shape != (len(points), len(result_header)):
        raise Exception(f'result was not filled correctly')
    
    # insert the output header at the beginning of result.
    result = pd.DataFrame(data = result, columns = result_header)
    
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
    if var_original == 'position' or option != ['none', 'none']:
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

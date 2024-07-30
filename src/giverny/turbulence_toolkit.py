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
import tracemalloc
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm
from giverny.turbulence_dataset import *
from giverny.turbulence_gizmos.basic_gizmos import *
from giverny.turbulence_gizmos.constants import get_constants
from giverny.turbulence_gizmos.getData import getData_process_data
from giverny.turbulence_gizmos.getCutout import getCutout_process_data

# installs sympy if necessary.
try:
    import sympy
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sympy'])
    
# installs pyJHTDB if necessary.
try:
    import pyJHTDB
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyJHTDB'])
finally:
    import pyJHTDB

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
        raise ValueError(f'max cutout size, {max_cutout_size} GB, exceeded. please specify a box with fewer than (xe - xs) * (ye - ys) * (ze - zs) = {max_datapoints + 1:,} ' + \
                         f'data points, regardless of strides.')
        
    # placeholder values for getData settings.
    spatial_method = 'none'
    spatial_method_specified = 'none'
    temporal_method = 'none'
    option = [-999.9, -999.9]
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

def getData(cube, var_original, timepoint_original_notebook, temporal_method_original, spatial_method_original, spatial_operator_original, points,
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
    
    # spatial interpolation map for legacy datasets.
    spatial_map = { 
        'none': 0, 'lag4': 4, 'lag6': 6, 'lag8': 8,
        'fd4noint': 40, 'fd6noint': 60, 'fd8noint': 80,
        'fd4lag4': 44,
        'm1q4': 104, 'm1q6': 106, 'm1q8': 108, 'm1q10': 110, 'm1q12': 112, 'm1q14': 114,
        'm2q4': 204, 'm2q6': 206, 'm2q8': 208, 'm2q10': 210, 'm2q12': 212, 'm2q14': 214,
        'm3q4': 304, 'm3q6': 306, 'm3q8': 308, 'm3q10': 310, 'm3q12': 312, 'm3q14': 314,
        'm4q4': 404, 'm4q6': 406, 'm4q8': 408, 'm4q10': 410, 'm4q12': 412, 'm4q14': 414
    }

    # temporal interpolation map for legacy datasets.
    temporal_map = {
        'none': 0,
        'pchip': 1
    }
    
    # retrieve the list of datasets processed by the giverny code.
    giverny_datasets = get_giverny_datasets()
    
    # -----
    # housekeeping procedures. will handle multiple variables, e.g. 'pressure' and 'velocity'.
    points, var, var_offsets, timepoint, temporal_method, spatial_method, spatial_method_specified, spatial_operator, datatype = \
        getData_housekeeping_procedures(query_type, dataset_title, points, var_original, timepoint_original_notebook,
                                        temporal_method_original, spatial_method_original, spatial_operator_original,
                                        option, c)
    
    # the number of values to read per datapoint. for pressure data this value is 1.  for velocity
    # data this value is 3, because there is a velocity measurement along each axis.
    num_values_per_datapoint = get_num_values_per_datapoint(var)
    # initialize cube constants. this is done so that all of the constants are known for pre-processing of the data.
    cube.init_constants(query_type, var, var_original, var_offsets, timepoint, timepoint_original_notebook,
                        spatial_method, spatial_method_specified, temporal_method, option, num_values_per_datapoint, c)
    
    # get the result header, which only contains the names for each column of the data values.
    output_header = get_interpolation_tsv_header(cube.dataset_title, cube.var_name, cube.timepoint_original, cube.timepoint_end, cube.delta_t, cube.sint, cube.tint)
    result_header = np.array(output_header.split('\n')[1].strip().split('\t'))[3:]
    
    # option parameter values.
    timepoint_end, delta_t = option
    
    # default timepoint range which only queries the first timepoint for non-'position' variables and non-time series queries.
    timepoint_range = np.arange(timepoint_original_notebook, timepoint_original_notebook + 1, 2)
    if var_original != 'position' and option != [-999.9, -999.9]:
        # timepoint range for the time series queries.
        timepoint_range = np.arange(timepoint_original_notebook, timepoint_end, delta_t)
        
        # add in the last timepoint if the final timepoint in the range is delta_t less than timepoint_end. np.arange is not good at handling
        # floating point step sizes.
        if math.isclose(timepoint_range[-1] + delta_t, timepoint_end, rel_tol = 10**-9, abs_tol = 0.0):
            timepoint_range = np.append(timepoint_range, timepoint_end)
            
    num_timepoints = len(timepoint_range)
        
    # only print the progress bar if verbose output.
    if verbose:
        timepoint_range = tqdm(timepoint_range, desc = f'times completed (n = {len(timepoint_range)}) ')
    
    # -----
    # starting the tracemalloc library.
    if trace_memory:
        tracemalloc.start()
        # checking the memory usage of the program.
        tracemem_start = [mem_value / (1024**3) for mem_value in tracemalloc.get_traced_memory()]
        tracemem_used_start = tracemalloc.get_tracemalloc_memory() / (1024**3)
    
    results = []
    # process the data query, retrieve interpolation/differentiation results for the various datasets.
    if dataset_title in giverny_datasets:
        """
        get the results for the datasets processed by giverny.
        """
        for timepoint_original in timepoint_range:
            # check that the user-input timepoint is a valid timepoint for the dataset.
            check_timepoint(timepoint_original, dataset_title, query_type)
            # convert the original input timepoint to the correct time index.
            timepoint = get_time_index_from_timepoint(dataset_title, timepoint_original, temporal_method, query_type)

            # pre-fill the result array that will be filled with the data that is read in. initially the datatype is set to "f" (float)
            # so that the array is filled with the missing placeholder value (-999.9).
            result = np.array([c['missing_value_placeholder']], dtype = 'f')

            if dataset_title not in ['sabl2048low', 'sabl2048high']:
                # get the results.
                result = getData_process_data(cube, points, var, timepoint, temporal_method, spatial_method,
                                              var_original, var_offsets, timepoint_original, spatial_method_specified, option, c)
            else:
                # separate points into different interpolation methods for the 'sabl2048low' and 'sabl2048high' datasets if they are near the boundary.
                points_map, original_indices_map = get_sabl_points_map(cube, points)
                # first key in points_map.
                first_key = next(iter(points_map))
                # last key in points_map.
                last_key = next(reversed(points_map))

                # update all the timepoints that need to be read.
                timepoints = [timepoint]
                if temporal_method == 'pchip':
                    floor_timepoint = math.floor(timepoint)
                    timepoints = [floor_timepoint - 1, floor_timepoint, floor_timepoint + 1, floor_timepoint + 2]

                # results and their corresponding specified order.
                result = []
                original_points_indices = []

                for timepoint_i, timepoint_tmp in enumerate(timepoints):
                    # results for each spatial interpolation method at timepoint_tmp.
                    result_spatial_methods = []
                    for spatial_method_tmp in points_map:
                        # changes spatial_method_specified to 'fd4noint_*' from 'fd4lag4_*/m2q8_*/m1q4_*' if the step-down interpolation method is 'sabl_linear_*' since
                        # only finite differencing is applied in the linear step-down region of the gradient/hessian/laplacian. replace does *not* affect the
                        # 'm2q8/m1q4' 'field' interpolations.
                        spatial_method_specified_tmp = spatial_method_specified.replace('fd4lag4_', 'fd4noint_').replace('m2q8_', 'fd4noint_').replace('m1q4_', 'fd4noint_') \
                                                                                        if 'sabl_linear' in spatial_method_tmp else spatial_method_specified

                        # points to be interpolated with this spatial_method_tmp interpolation method.
                        points_tmp = points_map[spatial_method_tmp]

                        # only keep track of the ordering of points for the first timepoint as duplicates of the ordering are not needed.
                        if timepoint_i == 0:
                            # keep track of the ordering of the points specified by the user.
                            for original_point_index in original_indices_map[spatial_method_tmp]:
                                original_points_indices.append(original_point_index)

                        if var_original == 'velocity':
                            # handles the velocity variable for the 'sabl2048low' and 'sabl2048high' datasets. queries getData twice for the (u, v) and (w) components
                            # of velocity separately.
                            var_offsets_tmp = 'velocity_uv'
                            result_tmp = getData_process_data(cube, points_tmp, var, timepoint_tmp, temporal_method, spatial_method_tmp,
                                                              var_original, var_offsets_tmp, timepoint_original, spatial_method_specified_tmp, option, c)

                            var_offsets_tmp = 'velocity_w'
                            result_tmp_w = getData_process_data(cube, points_tmp, var, timepoint_tmp, temporal_method, spatial_method_tmp,
                                                                var_original, var_offsets_tmp, timepoint_original, spatial_method_specified_tmp, option, c)

                            # overwrite the (w) values in result_tmp with the (w) values from result_tmp_w.
                            if '_g' in spatial_method_specified:
                                # handles the velocity gradient differentiations.
                                result_tmp[:, 6:] = result_tmp_w[:, 6:]
                            elif '_h' in spatial_method_specified:
                                # handles the velocity hessian differentiations.
                                result_tmp[:, 12:] = result_tmp_w[:, 12:]
                            else:
                                # handles the velocity field interpolations and laplacian differentiations.
                                result_tmp[:, 2] = result_tmp_w[:, 2]
                        else:
                            # handles all non-velocity variables for the 'sabl2048low' and 'sabl2048high' datasets.    
                            # get the results.
                            result_tmp = getData_process_data(cube, points_tmp, var, timepoint_tmp, temporal_method, spatial_method_tmp,
                                                              var_original, var_offsets, timepoint_original, spatial_method_specified_tmp, option, c)

                        # append the result for this interpolation method.
                        result_spatial_methods.append(result_tmp)

                    # append the result for timepoint_tmp.
                    result.append(np.vstack(result_spatial_methods))

                if temporal_method == 'pchip':
                    # dt between timepoints.
                    dt = get_time_dt(dataset_title, query_type)
                    # addition to map the time index back to the real time. 
                    time_index_shift = get_time_index_shift(dataset_title, query_type)
                    # convert the timepoints (time indices) back to real time.
                    times = [dt * (timepoint_val - time_index_shift) for timepoint_val in timepoints]

                    # pchip interpolation.
                    result = pchip(timepoint_original, times, result, dt)

                # stack all of the results together.
                result = np.vstack(result)

                # re-sort result to match the original ordering of points.
                original_points_indices, result = zip(*sorted(zip(original_points_indices, result), key = lambda x: x[0]))

                # convert the result list to a numpy array.
                result = np.array(result)

                if last_key != spatial_method or num_timepoints > 1:
                    # reset cube constants if the user-specified spatial_method was not utilized for the specific point query or 'pchip' temporal interpolation was specified.
                    # e.g. if all the points queried utilized a step-down interpolation method, then the cube constants are reset after processing so that cube.sint matches
                    # the specified spatial_method. if 'pchip' temporal interpolation was specified then this will also reset cube.timepoint to match the specified timepoint.
                    cube.init_constants(query_type, var, var_original, var_offsets, timepoint, timepoint_original_notebook,
                                        spatial_method, spatial_method_specified, temporal_method, option, num_values_per_datapoint, c)

            # checks to make sure that data was read in for all points.
            if c['missing_value_placeholder'] in result or result.shape != (len(points), len(result_header)):
                raise Exception(f'result was not filled correctly')

            # insert the output header at the beginning of result.
            result = pd.DataFrame(data = result, columns = result_header)

            # append the result into results.
            results.append(result)
    else:
        """
        get the results for the legacy datasets processed by pyJHTDB.
        """
        # initialize lJHTDB gSOAP resources and add the user's authorization token.
        if auth_token == c['pyJHTDB_testing_token'] and len(points) > 4096:
            turb_email = c['turbulence_email_address']
            raise Exception(f'too many points requested for the testing authorization token: {len(points)} > 4096\n\n' + \
                            f'an authorization token can be requested by email from {turb_email}\n' + \
                            f' include your name, email address, institutional affiliation and department, together with a short description of your intended use of the database')

        lJHTDB = pyJHTDB.libJHTDB(auth_token = auth_token)
        lJHTDB.initialize()

        # recast the points array as np.float32 because np.float64 does not work for the legacy datasets.
        points_tmp = points.astype(np.float32)

        for timepoint_original in timepoint_range:
            # check that the user-input timepoint is a valid timepoint for the dataset.
            check_timepoint(timepoint_original, dataset_title, query_type)
            # convert the original input timepoint to the correct time index.
            timepoint = get_time_index_from_timepoint(dataset_title, timepoint_original, temporal_method, query_type)

            # pre-fill the result array that will be filled with the data that is read in. initially the datatype is set to "f" (float)
            # so that the array is filled with the missing placeholder value (-999.9).
            result = np.array([c['missing_value_placeholder']], dtype = 'f')

            if datatype == 'Position':
                timepoint_end, delta_t = option

                # set the number of steps to keep to 1. for now this will not be a user-modifiable parameter.
                steps_to_keep = 1

                # formatting the output since getPosition prints output, whereas lJHTDB.getData does not.
                if verbose:
                    print()

                # only returning the position array ('result') to keep consistent with other getData variables. the time array can be calculated in the notebook if needed
                # as t = np.linspace(timepoint, timepoint_end, steps_to_keep + 1).astype(np.float32).
                result, t = lJHTDB.getPosition(data_set = dataset_title,
                                               starttime = timepoint, endtime = timepoint_end, dt = delta_t,
                                               point_coords = points_tmp, steps_to_keep = steps_to_keep)

                # only return the final point positions to keep consistent with the other "get" functions.
                result = result[-1]
            else:
                # get the spatial and temporal interpolation integers for the legacy datasets.
                sint = spatial_map[spatial_method_original]
                tint = temporal_map[temporal_method_original]

                # get the results.
                result = lJHTDB.getData(timepoint, points_tmp, data_set = dataset_title, sinterp = sint, tinterp = tint, getFunction = f'get{datatype}')

            # checks to make sure that data was read in for all points.
            if c['missing_value_placeholder'] in result or result.shape != (len(points), len(result_header)):
                raise Exception(f'result was not filled correctly')

            # insert the output header at the beginning of result.
            result = pd.DataFrame(data = result, columns = result_header)

            # append the result into results.
            results.append(result)

        # free up gSOAP resources.
        lJHTDB.finalize()
    
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

def pchip(time, times, results, dt):
    """
    pchip temporal interpolation.
    """
    # separate times and results for each time index.
    time0, time1, time2, time3 = times
    result0, result1, result2, result3 = results
    
    # interpolation derivatives.
    drv1 = (((result2 - result1) / dt) + ((result1 - result0) / (time1 - time0))) / 2
    drv2 = (((result3 - result2) / (time3 - time2)) + ((result2 - result1) / dt)) / 2

    # interpolation coefficients.
    a = result1
    b = drv1
    c = (((result2 - result1) / dt) - drv1) / dt
    d = 2 / dt / dt * (((drv1 + drv2) / 2) - ((result2 - result1) / dt))
    
    # interpolate the results.
    interpolated_results = a + b * (time - time1) + c * ((time - time1) * (time - time1)) + d * ((time - time1) * (time - time1) * (time - time2))
    
    return interpolated_results

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

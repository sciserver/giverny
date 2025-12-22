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
import csv
import sys
import json
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict
from plotly.subplots import make_subplots
from givernylocal.turbulence_gizmos.variable_grids import *
from givernylocal.turbulence_gizmos.jhtdb_schema import TurbulenceDB

"""
user-input checking gizmos.
"""
def load_json_metadata(url):
    """
    load the json simulation metadata for user input verification.
    """
    response = requests.get(url)
    metadata_json = json.loads(response.text)
    
    # validate the json metadata file.
    try:
        TurbulenceDB(**metadata_json)
    except Exception as e:
        raise Exception(f"json metadata validation error: {e}")
    
    metadata = {'name': metadata_json['name'],
                'description': metadata_json['description'],
                'pickled_metadata_filepath': metadata_json['pickled_metadata_filepath'],
                'variables': metadata_json['variables'],
                'spatial_operators': metadata_json['spatial_operators'],
                'spatial_methods': metadata_json['spatial_methods'],
                'temporal_methods': metadata_json['temporal_methods'],
                'constants': metadata_json['giverny_constants'],
                'datasets': {}}
    for dataset in metadata_json['datasets']:
        code = dataset['code']
        metadata['datasets'][code] = dataset
        
    return metadata

def check_dataset_title(metadata, dataset_title):
    """
    check that dataset_title is a valid dataset title.
    """
    valid_dataset_titles = list(metadata['datasets'])
    
    if dataset_title not in valid_dataset_titles:
        chunks = [', '.join(valid_dataset_titles[q : q + 7]) for q in range(0, len(valid_dataset_titles), 7)]
        raise Exception(f"'{dataset_title}' (case-sensitive) is not a valid dataset title:\n[" + 
                        ',\n '.join(chunks) + "]")
        
    return

def check_variable(metadata, variable, dataset_title, query_type):
    """
    check that variable is a valid variable name.
    """
    # populate the valid_variables list with the variable codes from the metadata file for the given dataset and query type.
    valid_variables_map = metadata['datasets'][dataset_title]['physicalVariables']
    if query_type == 'getcutout':
        # if the variable is 'gridded' (i.e. values stored at each grid point), then it can be queried with getcutout.
        valid_variables = [variable_info['code'] for variable_info in valid_variables_map if variable_info['isPersistent']]
    elif query_type == 'getdata':
        # getdata can query all variables, including 'gridded' variables and variables calculated in silico (e.g. 'position' and 'force' for some datasets).
        valid_variables = [variable_info['code'] for variable_info in valid_variables_map]
    elif query_type == 'getturbinedata':
        # getturbinedata can query the parquet turbine_variables.
        valid_variables = metadata['datasets'][dataset_title]['parquet']['turbine_variables']
    elif query_type == 'getbladedata':
        # getbladedata can query the parquet blade_variables.
        valid_variables = metadata['datasets'][dataset_title]['parquet']['blade_variables']
    
    if variable not in valid_variables:
        chunks = [', '.join(valid_variables[q : q + 7]) for q in range(0, len(valid_variables), 7)]
        raise Exception(f"'{variable}' (case-sensitive) is not a valid variable for ('{dataset_title}', '{query_type}'):\n[" + 
                        ',\n '.join(chunks) + "]")
        
    return

def check_timepoint(metadata, timepoint, dataset_title, query_type, max_num_timepoints = 1):
    """
    check that timepoint is a valid timepoint for the dataset.
    """
    # list of datasets which are low-resolution and thus the timepoint is specified as a discrete time index.
    time_index_datasets = [dataset for dataset in metadata['datasets'] if metadata['datasets'][dataset]['simulation']['tlims']['isDiscrete']]
    
    time_metadata = metadata['datasets'][dataset_title]['simulation']['tlims']
    time_lower = np.float64(time_metadata['lower'])
    time_upper = np.float64(time_metadata['upper'])
    time_steps = np.int64(time_metadata['n'])
    
    if query_type == 'getcutout':
        try:
            list(timepoint)
        except:
            raise Exception(f"t_range must be specified as a python list or numpy array, e.g. [{timepoint}, {timepoint}]")
        
        # check that the time range is specified as minimum and maximum integer values.
        if timepoint.dtype not in [np.int32, np.int64]:
            raise Exception('all t_range values, [minimum, maximum], should be specified as integers')
    
        if len(timepoint) != 2 or timepoint[0] > timepoint[1]:
            raise Exception(f't_range, {list(timepoint)}, is not correctly specified as [minimum, maximum]')
            
        valid_timepoints = range(1, time_steps + 1)
        
        # handles checking datasets with time indices.
        if timepoint[0] not in valid_timepoints or timepoint[1] not in valid_timepoints:
            raise Exception(f"'t_range', [{timepoint[0]}, {timepoint[1]}], is not a valid time range for '{dataset_title}': all times must be in the inclusive range of " +
                            f'[{valid_timepoints[0]}, {valid_timepoints[-1]}]')
    elif query_type == 'getdata':
        if dataset_title in time_index_datasets:
            valid_timepoints = range(1, time_steps + 1)
        
            # handles checking datasets with time indices.
            if timepoint not in valid_timepoints:
                raise Exception(f"{timepoint} is not a valid time for '{dataset_title}': must be an integer and in the inclusive range of " +
                                f'[{valid_timepoints[0]}, {valid_timepoints[-1]}]')
        else:
            valid_timepoints = (time_lower, time_upper)

            # handles checking datasets with real times.
            if timepoint < valid_timepoints[0] or timepoint > valid_timepoints[1]:
                raise Exception(f"{timepoint} is not a valid time for '{dataset_title}': must be in the inclusive range of " +
                                f'[{valid_timepoints[0]}, {valid_timepoints[1]}]')
    elif query_type == 'getturbinedata':
        try:
            timepoint = list(timepoint)
        except:
            raise Exception(f"turbine_times must be specified as a python list or numpy array, e.g. [{timepoint}]")
        
        # checks if too many timepoints were queried.
        if len(timepoint) > max_num_timepoints:
            raise Exception(f'too many timepoints queried, please limit the number of timepoints to <= {max_num_timepoints:,}')
        
        valid_timepoints = (time_lower, time_upper)
        
        min_time = np.nanmin(timepoint)
        max_time = np.nanmax(timepoint)
        
        if min_time < valid_timepoints[0] or max_time > valid_timepoints[1]:
            raise Exception(f"'times' (minimum = {min_time}, maximum = {max_time}) are not valid for '{dataset_title}': all times must be in the inclusive range of " +
                            f'[{valid_timepoints[0]}, {valid_timepoints[1]}]')
    elif query_type == 'getbladedata':
        try:
            timepoint = list(timepoint)
        except:
            raise Exception(f"blade_times must be specified as a python list or numpy array, e.g. [{timepoint}]")
        
        # checks if too many timepoints were queried.
        if len(timepoint) > max_num_timepoints:
            raise Exception(f'too many timepoints queried, please limit the number of timepoints to <= {max_num_timepoints:,}')
        
        valid_timepoints = (time_lower, time_upper)
        
        min_time = np.nanmin(timepoint)
        max_time = np.nanmax(timepoint)
        
        if min_time < valid_timepoints[0] or max_time > valid_timepoints[1]:
            raise Exception(f"'times' (minimum = {min_time}, maximum = {max_time}) are not valid for '{dataset_title}': all times must be in the inclusive range of " +
                            f'[{valid_timepoints[0]}, {valid_timepoints[1]}]')
    
    return

def check_points(metadata, points, dataset_title, variable, max_num_points):
    """
    check that the points are inside the acceptable domain along each axis for the dataset. 'modulo' placeholder values
    are used for cases where points outside the domain are allowed as modulo(domain range).
    """
    # retrieves the physical dimension limits metadata for the queried variable.
    variable_lims = {
        variable_info['code']: {
            lim: variable_info.get(lim, metadata['datasets'][dataset_title]['simulation'][lim])
            for lim in ['xlims', 'ylims', 'zlims']
        }
        for variable_info in metadata['datasets'][dataset_title]['physicalVariables']
    }[variable]
    
    # get the axes domain limits from the metadata file.
    axes_domain = []
    for lims in ['xlims', 'ylims', 'zlims']:
        lims_metadata = variable_lims[lims]
        bounds = [lims_metadata['lower'], lims_metadata['upper']]
        
        if lims_metadata['isPeriodic']:
            bounds = f'modulo[{bounds[0]}, {bounds[1]}]'
        else:
            bounds = [np.float64(bounds[0]), np.float64(bounds[1])]

        axes_domain.append(bounds)
    
    # checks if too many points were queried.
    if len(points) > max_num_points:
        raise Exception(f'too many points queried, please limit the number of points to <= {max_num_points:,}')

    # minimum and maximum point values along each axis.
    points_min = np.min(points, axis = 0)
    points_max = np.max(points, axis = 0)

    # checks if all points are within the axes domain for the dataset.
    points_domain_check = np.all([points_min[axis] >= axes_domain[axis][0] and points_max[axis] <= axes_domain[axis][1]
                                  if type(axes_domain[axis]) == list else True for axis in range(len(axes_domain))])

    if not points_domain_check:
        raise Exception(f"all points are not within the allowed domain [minimum, maximum] for ('{dataset_title}', '{variable}'):\n" +
                        f"x: {axes_domain[0]}\ny: {axes_domain[1]}\nz: {axes_domain[2]}")
    
    return

def check_points_chunks_intersection(metadata, points, dataset_title, variable):
    """
    check that the queried points do not intersect more than the maximum number of chunks.
    """
    chunk_size = get_dataset_chunk_size(metadata, dataset_title, variable)
    
    # number of dimensions per point.
    n_dims = points.shape[1]
    chunk_indices = np.zeros_like(points, dtype = int)
    
    # determine which axes have irregular grid spacing.
    spacing = get_dataset_spacing(metadata, dataset_title, variable)
    irregular_axes = []
    for axis_index, axis_spacing in enumerate(spacing):
        if isinstance(axis_spacing, np.ndarray):
            irregular_axes.append(axis_index)
    
    if irregular_axes:
        # process each dimension separately.
        for dim in range(n_dims):
            if dim in irregular_axes:
                if dataset_title == 'diurnal_windfarm' and variable == 'soiltemperature':
                    grid = spacing[dim]
                    # find the index in the z-gridpoint list where each of the z-points would be inserted.
                    indices = np.searchsorted(grid, -points[:, dim], side = 'right') - 1
                    # handles the bottom and top boundary gridpoints. shifts the index for z_point == z_grid[-1] down by 1 to account for needing a
                    # index before the specified z-point (equivalent to np.floor() for the x- and y-points).
                    # no modulo needed since the z-axis is non-periodic and all queried points are restricted to the z-domain.
                    indices = np.clip(indices, 0, len(grid) - 1)
                    chunk_indices[:, dim] = indices // chunk_size[dim]
            else:
                chunk_indices[:, dim] = np.floor_divide(points[:, dim], spacing[dim]) // chunk_size[dim]
    else:
        # process all dimensions together if regular grid spacing.
        chunk_indices = np.floor_divide(points, spacing) // chunk_size
    
    # number of unique chunks that are intersected by points.
    unique_chunks = len(np.unique(chunk_indices.astype(int), axis = 0))
    
    # check if the query requires reading from too many chunks.
    if unique_chunks > metadata['constants']['max_num_chunks']:
        raise Exception('query intersects too many storage chunks. please break up the points into separate queries, or query a smaller spatial domain.')

def check_spatial_operator(metadata, operator, dataset_title, variable):
    """
    check that the spatial interpolation operator is a valid operator.
    """
    # retrieves the valid spatial operator codes.
    valid_operators = {variables_info['code']: [operators['operator'] for operators in variables_info['spatialOperatorMethods']]
                       for variables_info in metadata['datasets'][dataset_title]['physicalVariables']}[variable]
    
    if operator not in valid_operators:
        raise Exception(f"'{operator}' (case-sensitive) is not a valid interpolation operator for '{variable}':\n{valid_operators}")
        
    return

def check_spatial_method(metadata, sint, dataset_title, variable, operator):
    """
    check that the spatial interpolation method is a valid method.
    """
    # retrieves the valid spatial method codes.
    valid_sints = {variables_info['code']: {operators['operator']: operators['methods'] for operators in variables_info['spatialOperatorMethods']}
                   for variables_info in metadata['datasets'][dataset_title]['physicalVariables']}[variable][operator]
    # suffix corresponding the spatial interpolation operator. an empty string is returned for "field" since this is the default operator.
    sint_suffix = '_' + operator if operator != 'field' else ''
        
    if sint not in valid_sints:
        raise Exception(f"'{sint}' (case-sensitive) is not a valid spatial interpolation method for ('{dataset_title}', '{variable}', '{operator}'):\n{valid_sints}")
        
    return sint + sint_suffix

def check_temporal_method(metadata, tint, dataset_title, variable):
    """
    check that the temporal interpolation method is a valid method.
    """
    # retrieves the valid temporal interpolation methods.
    valid_tints = {variables_info['code']: variables_info['temporalMethods'] for variables_info in metadata['datasets'][dataset_title]['physicalVariables']}[variable]
        
    if tint not in valid_tints:
        raise Exception(f"'{tint}' (case-sensitive) is not a valid temporal interpolation method for ('{dataset_title}', '{variable}'):\n{valid_tints}")
        
    return

def check_option_parameter(metadata, option, dataset_title, timepoint_start):
    """
    check that the 'option' parameter used by getPosition was correctly specified.
    """
    timepoint_end, delta_t = option
    
    # list of datasets which are low-resolution and thus the timepoint is specified as a discrete time index.
    time_index_datasets = [dataset for dataset in metadata['datasets'] if metadata['datasets'][dataset]['simulation']['tlims']['isDiscrete']]
    
    if timepoint_end == -999.9 or delta_t == -999.9:
        focus_text = '\033[43m'
        default_text = '\033[0m'
        raise Exception(f"'time_end' and 'delta_t' (option = [time_end, delta_t]) must be specified for the 'position' variable and time series queries, e.g.:\n" \
                        f"result = getData(dataset, variable, time, temporal_method, spatial_method, spatial_operator, points, {focus_text}{'option'}{default_text})")
    elif (timepoint_start + delta_t) > timepoint_end and not math.isclose(timepoint_start + delta_t, timepoint_end, rel_tol = 10**-9, abs_tol = 0.0):
        raise Exception(f"'time' + 'delta_t' is greater than 'time_end': {timepoint_start} + {delta_t} = {timepoint_start + delta_t} > {timepoint_end}")
    elif dataset_title in time_index_datasets and delta_t != int(delta_t):
        raise Exception(f"delta_t must be an integer for discrete time datasets:\n{time_index_datasets}")
        
    return

def check_turbine_numbers(metadata, dataset_title, turbine_numbers):
    """
    check that the specified turbine number(s) are valid.
    """
    valid_turbine_numbers = metadata['datasets'][dataset_title]['parquet']['turbine_numbers']
    
    try:
        turbine_numbers = list(turbine_numbers)
    except:
        raise Exception(f"turbine_numbers must be specified as a python list or numpy array, e.g. [{turbine_numbers}]")
    
    for turbine_number in turbine_numbers:
        if turbine_number != int(turbine_number) or turbine_number not in valid_turbine_numbers:
            raise Exception(f"'{turbine_number}' is not a valid turbine number (must be integer) for '{dataset_title}':\n{valid_turbine_numbers}")
    
    return np.array(turbine_numbers, dtype = np.int64)

def check_blade_numbers(metadata, dataset_title, blade_numbers):
    """
    check that the specified blade number(s) are valid.
    """
    valid_blade_numbers = metadata['datasets'][dataset_title]['parquet']['blade_numbers']
    
    try:
        blade_numbers = list(blade_numbers)
    except:
        raise Exception(f"blade_numbers must be specified as a python list or numpy array, e.g. [{blade_numbers}]")
    
    for blade_number in blade_numbers:
        if blade_number != int(blade_number) or blade_number not in valid_blade_numbers:
            raise Exception(f"'{blade_number}' is not a valid blade number (must be integer) for '{dataset_title}':\n{valid_blade_numbers}")
    
    return np.array(blade_numbers, dtype = np.int64)

def check_blade_points(metadata, dataset_title, blade_points):
    """
    check that the specified blade point(s) are valid.
    """
    valid_blade_points = metadata['datasets'][dataset_title]['parquet']['blade_actuator_points']
    
    try:
        blade_points = list(blade_points)
    except:
        raise Exception(f"blade_actuator_points must be specified as a python list or numpy array, e.g. [{blade_points}]")
    
    for blade_point in blade_points:
        if blade_point != int(blade_point) or blade_point not in valid_blade_points:
            chunks = [', '.join(str(point) for point in valid_blade_points[q : q + 10]) for q in range(0, len(valid_blade_points), 10)]
            raise Exception(f"'{blade_point}' is not a valid blade actuator point (must be integer) for '{dataset_title}':\n[" + 
                            ',\n '.join(chunks) + "]")
    
    return np.array(blade_points, dtype = np.int64)

def check_axes_ranges(metadata, axes_ranges, dataset_title, variable):
    """
    check that the specified cutout axes are ranges are within the allowed domain.
    """
    axes_resolution = get_dataset_resolution(metadata, dataset_title, variable)
    
    # retrieves the physical dimension limits metadata for the queried variable.
    variable_lims = {
        variable_info['code']: {
            lim: variable_info.get(lim, metadata['datasets'][dataset_title]['simulation'][lim])
            for lim in ['xlims', 'ylims', 'zlims']
        }
        for variable_info in metadata['datasets'][dataset_title]['physicalVariables']
    }[variable]
    
    # get the axes domain limits from the metadata file.
    axes_periodic = []
    for lims in ['xlims', 'ylims', 'zlims']:
        axes_periodic.append(variable_lims[lims]['isPeriodic'])
    
    axes_periodic = np.array(axes_periodic)
    
    if axes_ranges.dtype not in [np.int32, np.int64]:
        raise Exception('all axis range values, [minimum, maximum], should be specified as integers')
    
    # check that the axis ranges are all specified as minimum and maximum integer values and are in the domain for the dataset.
    for axis_index, (axis_resolution, axis_periodic, axis_range) in enumerate(zip(axes_resolution, axes_periodic, axes_ranges)):
        if len(axis_range) != 2 or axis_range[0] > axis_range[1]:
            raise Exception(f'axis range, {list(axis_range)}, is not correctly specified as [minimum, maximum]')
        
        if not axis_periodic and (axis_range[0] < 1 or axis_range[1] > axis_resolution):
            axis_name = ['x', 'y', 'z'][axis_index]
            raise Exception(f'{axis_name}-axis does not have a periodic boundary condition and so must be specified in the domain, [1, {axis_resolution}]')
                
def check_strides(strides):
    """
    check that the strides are all positive integer values.
    """
    for stride in strides:
        if type(stride) not in [np.int32, np.int64] or stride < 1:
            raise Exception(f'stride, {stride}, is not an integer value >= 1')

"""
mapping gizmos.
"""
def get_dataset_filepath(metadata, dataset_title):
    """
    get the zarr filepath of the dataset.
    """
    return metadata['datasets'][dataset_title]['storage']['filepath']

def get_dataset_resolution(metadata, dataset_title, variable):
    """
    get the number of datapoints (resolution) along each axis of the dataset.
    """
    # retrieves the physical dimension limits metadata for the queried variable.
    variable_lims = {
        variable_info['code']: {
            lim: variable_info.get(lim, metadata['datasets'][dataset_title]['simulation'][lim])
            for lim in ['xlims', 'ylims', 'zlims']
        }
        for variable_info in metadata['datasets'][dataset_title]['physicalVariables']
    }[variable]
    
    axes_domain = []
    for lims in ['xlims', 'ylims', 'zlims']:
        axes_domain.append(variable_lims[lims]['n'])
    
    return np.array(axes_domain)

def get_dataset_spacing(metadata, dataset_title, variable):
    """
    get the dataset spacing between datapoints along each axis of the dataset.
    """
    # irregular grid axis function names.
    irregular_grids = {
        'irregular channel dy': get_channel_ys,
        'irregular channel5200 dy': get_channel5200_ys,
        'irregular transition bl dy': get_transition_bl_ys,
        'irregular diurnal windfarm dz': get_diurnal_windfarm_zs
    }
    
    # retrieves the physical dimension limits metadata for the queried variable.
    variable_lims = {
        variable_info['code']: {
            lim: variable_info.get(lim, metadata['datasets'][dataset_title]['simulation'][lim])
            for lim in ['xlims', 'ylims', 'zlims']
        }
        for variable_info in metadata['datasets'][dataset_title]['physicalVariables']
    }[variable]
    
    dataset_spacing = []
    for lims in ['xlims', 'ylims', 'zlims']:
        axis_spacing = variable_lims[lims]['spacing']
        # retrieve the irregular grid coordinates.
        if axis_spacing in irregular_grids:
            axis_spacing = irregular_grids[axis_spacing]()
        
        dataset_spacing.append(axis_spacing)
    
    return np.array(dataset_spacing, dtype = object)

def get_dataset_chunk_size(metadata, dataset_title, variable):
    """
    chunk size along each dimension (x, y, z).
    """
    # retrieves the physical dimension limits metadata for the queried variable.
    axes_chunks = {
        variable_info['code']: variable_info.get('storage', {}).get('chunks', metadata['datasets'][dataset_title]['storage']['chunks'])
        for variable_info in metadata['datasets'][dataset_title]['physicalVariables']
    }[variable]
    
    return np.array(axes_chunks)

def get_dataset_coordinate_offsets(metadata, dataset_title, variable_offsets, variable):
    """
    get the dataset *coor offsets. values are the axes coordinate offsets from 0.
    """
    # retrieves the grid offsets.
    offsets_map = {variables_info['code']: {offsets['code']: offsets['coordinate'] for offsets in variables_info['offsets']}
                   for variables_info in metadata['datasets'][dataset_title]['physicalVariables']}
    
    return np.array(offsets_map[variable][variable_offsets], dtype = np.float64)

def get_nonperiodic_and_regular_spacing_spatial_axes(metadata, dataset_title, variable):
    """
    get list of non-periodic and regularly spaced (not irregular grid spacing) spatial axes.
    """
    # retrieves the physical dimension limits metadata for the queried variable.
    variable_lims = {
        variable_info['code']: {
            lim: variable_info.get(lim, metadata['datasets'][dataset_title]['simulation'][lim])
            for lim in ['xlims', 'ylims', 'zlims']
        }
        for variable_info in metadata['datasets'][dataset_title]['physicalVariables']
    }[variable]
    
    # get the axes periodicity and grid spacing from the metadata file.
    nonperiodic_axes = []
    for axis_index, lims in enumerate(['xlims', 'ylims', 'zlims']):
        # verifies that the spacing is regular (specified as a single floating point or integer value).
        if (not variable_lims[lims]['isPeriodic']) and (type(variable_lims[lims]['spacing']) != str):
            nonperiodic_axes.append(axis_index)
    
    return nonperiodic_axes

def get_time_dt(metadata, dataset_title, query_type):
    """
    dt between timepoints.
    """
    time_metadata = metadata['datasets'][dataset_title]['simulation']['tlims']
    time_lower = np.float64(time_metadata['lower'])
    time_upper = np.float64(time_metadata['upper'])
    time_steps = np.int64(time_metadata['n'])
    
    return {'getcutout': 1,
            'getdata': 1.0 if time_steps == 1 else (time_upper - time_lower) / (time_steps - 1)
           }[query_type]
    
def get_time_index_shift(metadata, dataset_title, query_type):
    """
    addition to map the time to a correct time index in the filenames. e.g. the first time index in the 'sabl2048high' dataset is 0; this time index
    is disallowed for queries to handle 'pchip' time interpolation queries. so, time 0 specified by the user corresponds
    to time index 1. for 'sabl2048high' this means the time index shift is +1 for 'getdata' queries. the default of -1 for getcutout
    and getdata is used for converting low-resolution datasets time indices to 0-based time indices and also as a placeholder value for
    the pyJHTDB datasets.
    """
    return metadata['datasets'][dataset_title]['simulation']['tlims']['timeIndexShift'][query_type]

def get_time_index_from_timepoint(metadata, dataset_title, timepoint, tint, query_type):
    """
    get the corresponding time index for this dataset from the specified timepoint. handles datasets that allow 'pchip' time interpolation, which
    requires 2 timepoints worth of data on either side of the timepoint specified by the user.
        - returns timepoint if the dataset is processed by the legacy pyJHTDB code because the timepoint to time index conversion is handled in pyJHTDB.
    """
    giverny_datasets = get_giverny_datasets()
    
    # addition to map the time to a correct time index in the filename.
    time_index_shift = get_time_index_shift(metadata, dataset_title, query_type)
    
    if dataset_title in giverny_datasets:
        # dt between timepoints.
        dt = get_time_dt(metadata, dataset_title, query_type)

        # convert the timepoint to a time index.
        time_index = (timepoint / dt) + time_index_shift
        # round the time index the nearest time index grid point if 'none' time interpolation was specified.
        if tint == 'none':
            time_index = np.floor(time_index + 0.5).astype(int)
    else:
        # do not convert the timepoint to a time index for datasets processed by pyJHTDB.
        time_index = timepoint
        
        # convert the time index to a 0-based index for the low-resolution pyJHTDB time index datasets.
        if query_type == 'getdata':
            time_index += time_index_shift
    
    return time_index

def get_giverny_datasets():
    """
    get the dataset titles that are processed by the giverny code (this backend code, *not* the legacy pyJHTDB code).
    """
    # TESTING. adding new datasets as they are moved to ceph.
    # return ['isotropic1024fine', 'isotropic1024coarse', 'mhd1024', 'isotropic8192', 'isotropic32768',
    #         'sabl2048low', 'sabl2048high', 'stsabl2048low', 'stsabl2048high', 'channel', 'diurnal_windfarm', 'nbl_windfarm']

    return ['isotropic8192', 'isotropic32768', 'sabl2048low', 'sabl2048high', 'stsabl2048low', 'stsabl2048high',
            'diurnal_windfarm', 'nbl_windfarm']

def get_irregular_mesh_ygrid_datasets(metadata, variable):
    """
    get the dataset titles that are irregular mesh y-grids (nonconstant dy).
    """
    irregular_dy_datasets = []
    for dataset_title in metadata['datasets']:
        # retrieves the dy spacing for the queried variable.
        dy_spacing = {
            variable_info['code']: variable_info.get('ylims', metadata['datasets'][dataset_title]['simulation']['ylims'])
            for variable_info in metadata['datasets'][dataset_title]['physicalVariables']
        }
        
        if variable in dy_spacing and "irregular" in str(dy_spacing[variable]['spacing']):
            irregular_dy_datasets.append(dataset_title)
    
    return irregular_dy_datasets

def get_irregular_mesh_zgrid_datasets(metadata, variable):
    """
    get the dataset titles that are irregular mesh z-grids (nonconstant dz).
    """
    irregular_dz_datasets = []
    for dataset_title in metadata['datasets']:
        # retrieves the dz spacing for the queried variable.
        dz_spacing = {
            variable_info['code']: variable_info.get('zlims', metadata['datasets'][dataset_title]['simulation']['zlims'])
            for variable_info in metadata['datasets'][dataset_title]['physicalVariables']
        }
        
        if variable in dz_spacing and "irregular" in str(dz_spacing[variable]['spacing']):
            irregular_dz_datasets.append(dataset_title)
    
    return irregular_dz_datasets

def get_variable_component_names_map(metadata):
    """
    map of the component names for each variable, e.g. "ux" is the x-component of the velocity.
    """
    return {variable['code']: {q: code for q, code in enumerate(variable['component_codes'])} 
            for variable in metadata['variables']}

def get_cardinality(metadata, variable):
    """
    get the number of values per datapoint for the user-specified variable.
    """
    return {variable['code']: variable['cardinality'] for variable in metadata['variables']}[variable]

def get_output_title(metadata, dataset_title):
    """
    format the dataset title string for the contour plot titles.
    """
    return metadata['datasets'][dataset_title]['name']

def get_output_variable_name(metadata, variable):
    """
    format the variable name string for the HDF5 output file dataset name.
    """
    return {variable['code']: variable['name'] for variable in metadata['variables']}[variable]

def get_cardinality_name(metadata, variable):
    """
    get the cardinality of the output data.
    """
    cardinality = {variable['code']: variable['cardinality'] for variable in metadata['variables']}[variable]
    if cardinality == 1:
        cardinality = 'Scalar'
    else:
        cardinality = 'Vector'
    
    return cardinality

def get_parquet_folderpath(metadata, dataset_title):
    """
    get the folderpath for the parquet files.
    """
    return metadata['datasets'][dataset_title]['parquet']['folderpath']
    
def get_parquet_time_info(metadata, dataset_title):
    """
    get the time info (time_offset, time_align, time_step) for the parquet files.
    """
    time_offset = np.float64(metadata['datasets'][dataset_title]['parquet']['time_offset'])
    time_align = np.float64(metadata['datasets'][dataset_title]['parquet']['time_align'])
    time_step = np.float64(metadata['datasets'][dataset_title]['parquet']['time_step'])
    
    return (time_offset, time_align, time_step)

def get_interpolation_tsv_header(metadata, dataset_title, variable_name, timepoint, timepoint_end, delta_t, sint, tint):
    """
    get the interpolation tsv header.
    """
    # parse the interpolation method and operator from sint.
    sint_split = sint.split('_')
    
    # interpolation method (e.g. 'lag4', 'm2q8', 'fd6noint').
    method = sint.replace('_gradient', '').replace('_hessian', '').replace('_laplacian', '')
    
    # interpolation operator (field, gradient, hessian, or laplacian).
    operator = 'field'
    if '_' in sint:
        operator = sint.split('_')[-1]
        
    # retrieves the variables metadata information.
    variable_map = {variables_info['code']: variables_info
                    for variables_info in metadata['variables']}[variable_name]
    
    # number of components for the specified variable.
    num_component_codes = len(variable_map['component_codes'])
    # number of data columns. used to determine number of '\t' characters in the header row so the tsv displays nicely.
    # 3 columns corresponding to 'x_point', 'y_point', and 'z_point' apply to all output.
    num_data_columns = {
        'field': 3 + num_component_codes,
        'gradient': 3 + num_component_codes * 3,
        'hessian': 3 + num_component_codes * 6,
        'laplacian': 3 + num_component_codes
    }[operator]
    
    if variable_name == 'position' or (timepoint_end != -999.9 and delta_t != -999.9):
        point_header = f'dataset: {dataset_title}, variable: {variable_name}, time: {timepoint}, time end: {timepoint_end}, delta t: {delta_t}, temporal method: {tint}, ' + \
                       f'spatial method: {method}, spatial operator: {operator}' + '\t' * num_data_columns + '\n'
    else:
        point_header = f'dataset: {dataset_title}, variable: {variable_name}, time: {timepoint}, temporal method: {tint}, ' + \
                       f'spatial method: {method}, spatial operator: {operator}' + '\t' * num_data_columns + '\n'
    point_header += 'x_point\ty_point\tz_point'
    
    return {
        'field': point_header + '\t' + '\t'.join(variable_map['component_codes']),
        'gradient': point_header + '\t' + '\t'.join([f'd{component_code}dx\td{component_code}dy\td{component_code}dz'
                                                     for component_code in variable_map['component_codes']]),
        'hessian': point_header + '\t' + '\t'.join([f'd2{component_code}dxdx\td2{component_code}dxdy\td2{component_code}dxdz' + \
                                                    f'\td2{component_code}dydy\td2{component_code}dydz\td2{component_code}dzdz'
                                                    for component_code in variable_map['component_codes']]),
        'laplacian': point_header + '\t' + '\t'.join([f'grad2{component_code}'
                                                      for component_code in variable_map['component_codes']])
    }[operator]

"""
processing gizmos.
"""
def assemble_axis_data(axes_data):
    """
    assemble all of the axis data together into one numpy array.
    """
    return np.array(axes_data, dtype = np.ndarray)

def get_axes_ranges_num_datapoints(axes_ranges):
    """
    number of datapoints along each axis.
    """
    return axes_ranges[:, 1] - axes_ranges[:, 0] + 1

def convert_to_0_based_ranges(metadata, axes_ranges, dataset_title, variable):
    """
    convert the axes ranges to 0-based indices (giverny datasets) from 1-based user input. indices are left as 1-based for pyJHTDB datasets.
        - truncates the axes ranges if they are longer than the cube size along each axis dimension. this is done so that each point is 
          only read from a data file once. periodic axes ranges that are specified longer than the axis dimension will be tiled after reading
          from the files.
    """
    cube_resolution = get_dataset_resolution(metadata, dataset_title, variable)

    # calculate the number of datapoints along each axis range.
    axes_lengths = get_axes_ranges_num_datapoints(axes_ranges)
    # make a copy of axes_ranges for adjusting the axes ranges as needed to simplify the queries.
    converted_axes_ranges = np.array(axes_ranges)
    
    # retrieve the list of datasets processed by the giverny code.
    giverny_datasets = get_giverny_datasets()
    
    # adjust the axes ranges to 0-based indices for datasets processed by giverny. the pyJHTDB code handles this for datasets it processes.
    if dataset_title in giverny_datasets:
        # convert the 1-based axes ranges to 0-based axes ranges.
        converted_axes_ranges = converted_axes_ranges - 1

    # truncate the axes range if necessary.
    for axis_index in range(len(converted_axes_ranges)):
        if axes_lengths[axis_index] > cube_resolution[axis_index]:
            converted_axes_ranges[axis_index, 1] = converted_axes_ranges[axis_index, 0] + cube_resolution[axis_index] - 1
    
    return converted_axes_ranges
    
"""
output folder gizmos.
"""
def create_output_folder(output_path):
    """
    create the output folder directory.
    """
    os.makedirs(output_path, exist_ok = True)
        
"""
user-interfacing gizmos.
"""
def write_interpolation_tsv_file(cube, points, interpolation_data, output_filename):
    """
    write the interpolation results to a tsv file.
    """
    print('writing the interpolation .tsv file...')
    sys.stdout.flush()

    # create the output folder if it does not already exist.
    create_output_folder(cube.output_path)

    # write the tsv file.
    with open(cube.output_path.joinpath(output_filename + '.tsv'), 'w', encoding = 'utf-8', newline = '') as output_file:
        # output header.
        output_header = [header_row.split('\t') 
                         for header_row in get_interpolation_tsv_header(cube.metadata, cube.dataset_title, cube.var, cube.timepoint_original, cube.timepoint_end, cube.delta_t,
                                                                        cube.sint, cube.tint).split('\n')]
        # append the timepoint column header because the output array is now multi-dimensional with time components to handle time series queries.
        output_header[-1].insert(0, 'time')

        # create a csv writer object with tab delimiter.
        writer = csv.writer(output_file, delimiter = '\t')
        # write the header row to the tsv file.
        writer.writerows(output_header)
        
        for time_index, interpolation_time_component in enumerate(interpolation_data):
            # concatenate the points and interpolation_time_component matrices together.
            output_data = np.concatenate((points, interpolation_time_component), axis = 1)
            output_data = np.column_stack((np.full(output_data.shape[0], cube.timepoint_original + time_index * cube.delta_t), output_data))
            
            # write output_data to the tsv file.
            writer.writerows(output_data)

    print('\nfile written successfully.')
    print('-----')
    sys.stdout.flush()
    
def write_turbine_tsv_file(cube, turbine_results, output_filename):
    """
    write the turbine results to a tsv file.
    """
    print('writing the turbine .tsv file...')
    sys.stdout.flush()

    # create the output folder if it does not already exist.
    create_output_folder(cube.output_path)
    output_file = cube.output_path.joinpath(output_filename + '.tsv')
    
    # write to tsv file.
    turbine_results.to_csv(output_file, sep = '\t', index = True, encoding = 'utf-8')

    print('\nfile written successfully.')
    print('-----')
    sys.stdout.flush()
    
def write_blade_tsv_file(cube, blade_results, output_filename):
    """
    write the blade results to a tsv file.
    """
    print('writing the blade .tsv file...')
    sys.stdout.flush()

    # create the output folder if it does not already exist.
    create_output_folder(cube.output_path)
    output_file = cube.output_path.joinpath(output_filename + '.tsv')
    
    # write to tsv file.
    blade_results.to_csv(output_file, sep = '\t', index = True, encoding = 'utf-8')

    print('\nfile written successfully.')
    print('-----')
    sys.stdout.flush()
            
def write_cutout_hdf5_and_xmf_files(cube, output_data, output_filename):
    """
    write the hdf5 and xmf files for the getCutout result.
    """
    print('writing the cutout .h5 and .xmf files...')
    sys.stdout.flush()
    
    # write output_data to a hdf5 file.
    # -----
    output_data.to_netcdf(cube.output_path.joinpath(output_filename + '.h5'),
                          format = "NETCDF4", mode = "w")
    
    # write the xmf file.
    # -----
    # get the dataset name used for the hdf5 file.
    h5_var = cube.var
    h5_attribute_type = get_cardinality_name(cube.metadata, cube.var)
    h5_dataset_names = list(output_data.data_vars.keys())
    
    # the shape of the cutout. ordering of the dimensions in the xarray, output_data, is (z, y, x), so shape is reversed ([::-1]) to keep
    # consistent with the expected (x, y, z) ordering.
    shape = [*output_data.sizes.values()][:3][::-1]
    
    if cube.dataset_title in ['sabl2048low', 'sabl2048high', 'stsabl2048low', 'stsabl2048high'] and cube.var == 'velocity':
        # split up "zcoor" into "zcoor_uv" and "zcoor_w" for the "velocity" variable of the "sabl" datasets.
        output_str = f"""<?xml version=\"1.0\" ?>
<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>
<Xdmf Version=\"2.0\">
  <Domain>
    <Grid Name="{h5_var}" GridType="Collection" CollectionType="Temporal">"""
        
        for h5_dataset_name in h5_dataset_names:
            # get the output timepoint.
            xmf_timepoint = int(h5_dataset_name.split('_')[1].strip())
            
            output_str += f"""
      <Grid Name=\"Structured Grid\" GridType=\"Uniform\">
        <Time Value=\"{xmf_timepoint}\"/>
        <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"{shape[2]} {shape[1]} {shape[0]}\"/>
        <Geometry GeometryType=\"VXVYVZ\">
          <DataItem Name=\"Xcoor\" Dimensions=\"{shape[0]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
            {output_filename}.h5:/xcoor
          </DataItem>
          <DataItem Name=\"Ycoor\" Dimensions=\"{shape[1]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
            {output_filename}.h5:/ycoor
          </DataItem>
          <DataItem Name=\"Zcoor_uv\" Dimensions=\"{shape[2]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
            {output_filename}.h5:/zcoor_uv
          </DataItem>
          <DataItem Name=\"Zcoor_w\" Dimensions=\"{shape[2]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
            {output_filename}.h5:/zcoor_w
          </DataItem>
        </Geometry>
        <Attribute Name=\"{h5_var}\" AttributeType=\"{h5_attribute_type}\" Center=\"Node\">
          <DataItem Dimensions=\"{shape[2]} {shape[1]} {shape[0]} {cube.num_values_per_datapoint}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
            {output_filename}.h5:/{h5_dataset_name}
          </DataItem>
        </Attribute>
      </Grid>"""
            
        output_str += f"""
    </Grid>
  </Domain>
</Xdmf>"""
    else:
        # handle other datasets and variables.
        output_str = f"""<?xml version=\"1.0\" ?>
<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>
<Xdmf Version=\"2.0\">
  <Domain>
    <Grid Name="{h5_var}" GridType="Collection" CollectionType="Temporal">"""
        
        for h5_dataset_name in h5_dataset_names:
            # get the output timepoint.
            xmf_timepoint = int(h5_dataset_name.split('_')[1].strip())
            
            output_str += f"""
      <Grid Name=\"Structured Grid\" GridType=\"Uniform\">
        <Time Value=\"{xmf_timepoint}\"/>
        <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"{shape[2]} {shape[1]} {shape[0]}\"/>
        <Geometry GeometryType=\"VXVYVZ\">
          <DataItem Name=\"Xcoor\" Dimensions=\"{shape[0]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
            {output_filename}.h5:/xcoor
          </DataItem>
          <DataItem Name=\"Ycoor\" Dimensions=\"{shape[1]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
            {output_filename}.h5:/ycoor
          </DataItem>
          <DataItem Name=\"Zcoor\" Dimensions=\"{shape[2]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
            {output_filename}.h5:/zcoor
          </DataItem>
        </Geometry>
        <Attribute Name=\"{h5_var}\" AttributeType=\"{h5_attribute_type}\" Center=\"Node\">
          <DataItem Dimensions=\"{shape[2]} {shape[1]} {shape[0]} {cube.num_values_per_datapoint}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
            {output_filename}.h5:/{h5_dataset_name}
          </DataItem>
        </Attribute>
      </Grid>"""
            
        output_str += f"""
    </Grid>
  </Domain>
</Xdmf>"""

    with open(cube.output_path.joinpath(output_filename + '.xmf'), 'w') as output_file:
        output_file.write(output_str)
    
    print('\nfiles written successfully.')
    print('-----')
    sys.stdout.flush()

def contour_plot(cube, value_index, cutout_data, time, plot_ranges, xyzt_axes_ranges, xyzt_strides, output_filename,
                 colormap = 'inferno', equal_aspect_ratio = True):
    """
    create a contour plot from the getCutout output.
    """
    # constants.
    # -----
    figsize = 8.5
    dpi = 300
    
    # dictionaries.
    # -----
    metadata = cube.metadata
    variable = cube.var
    dataset_title = cube.dataset_title
    dataset_name = variable + '_' + str(time).zfill(4)
    
    # remove the time axis from axes_ranges and strides.
    axes_ranges = xyzt_axes_ranges[:3]
    strides = xyzt_strides[:3]
    # get the time range.
    time_range = xyzt_axes_ranges[3]
    
    # names for each value, e.g. value index 0 for velocity data corresponds to the x-component of the velocity ("ux").
    value_name_map = get_variable_component_names_map(metadata)
    
    # create the output folder if it does not already exist.
    create_output_folder(cube.output_path)
    
    # exception handling.
    # -----
    # check that the user-input x-, y-, and z-axis plot ranges are all specified correctly as [minimum, maximum] integer values.
    check_axes_ranges(metadata, plot_ranges, dataset_title, variable)
    
    # check that the user specified a valid value index.
    if value_index not in value_name_map[variable]:
        raise Exception(f"{value_index} is not a valid value_index: {list(value_name_map[variable].keys())}")
        
    # transposed minimum and maximum arrays for both plot_ranges and axes_ranges.
    plot_ranges_min = plot_ranges[:, 0]
    plot_ranges_max = plot_ranges[:, 1]
    
    # calculate the strides to downscale the plot to avoid plotting more gridpoints than pixels.
    plot_axes_lengths = plot_ranges_max - plot_ranges_min + 1
    adjusted_strides = strides * np.ceil(np.ceil(plot_axes_lengths / (figsize * dpi)).astype(np.int32) / strides).astype(np.int32)
    
    axes_min = axes_ranges[:, 0]
    axes_max = axes_ranges[:, 1]
    
    # raise exception if all of the plot datapoints are not inside the bounds of the user box volume.
    if not(np.all(axes_min <= plot_ranges_min) and np.all(plot_ranges_max <= axes_max)):
        raise Exception(f'the specified plot ranges are not all bounded by the box volume defined by:\n{axes_ranges}')
        
    # raise exception if the plot time is not one of the queried timepoints.
    if time not in range(time_range[0], time_range[1] + 1, 1):
        raise Exception(f'the specified time ({time}) is not a queried time, t_range:\n{time_range}')
        
    # determine how many of the axis minimum values are equal to their corresponding axis maximum value.
    num_axes_equal_min_max = plot_ranges_min == plot_ranges_max
    # raise exception if the data being plotted is not 2-dimensional.
    if np.count_nonzero(num_axes_equal_min_max == True) != 1:
        raise Exception(f'only one axis (x, y, or z) should be specified as a single point, e.g. z_plot_range = [3, 3], to create a contour plot')
        
    # datasets that have an irregular grid.
    irregular_ygrid_datasets = get_irregular_mesh_ygrid_datasets(metadata, variable)
    irregular_zgrid_datasets = get_irregular_mesh_zgrid_datasets(metadata, variable)
    
    # convert the requested plot ranges to the data domain.
    xcoor_values = np.around(np.arange(plot_ranges_min[0] - 1, plot_ranges_max[0], adjusted_strides[0], dtype = np.float32) * cube.dx, cube.decimals)
    xcoor_values += cube.coor_offsets[0]
    if dataset_title in irregular_ygrid_datasets:
        # note: this assumes that the y-axis of the irregular grid datasets is non-periodic.
        ycoor_values = cube.dy[np.arange(plot_ranges_min[1] - 1, plot_ranges_max[1], adjusted_strides[1])]
    else:
        ycoor_values = np.around(np.arange(plot_ranges_min[1] - 1, plot_ranges_max[1], adjusted_strides[1], dtype = np.float32) * cube.dy, cube.decimals)
        ycoor_values += cube.coor_offsets[1]
    if dataset_title in irregular_zgrid_datasets:
        # note: this assumes that the z-axis of the irregular grid datasets is non-periodic.
        zcoor_values = cube.dz[np.arange(plot_ranges_min[2] - 1, plot_ranges_max[2], adjusted_strides[2])]
    else:
        zcoor_values = np.around(np.arange(plot_ranges_min[2] - 1, plot_ranges_max[2], adjusted_strides[2], dtype = np.float32) * cube.dz, cube.decimals)
        zcoor_values += cube.coor_offsets[2]

    # generate the plot.
    print('generating contour plot...')
    print('-----')
    sys.stdout.flush()
    
    # -----
    # name of the value that is being plotted.
    value_name = value_name_map[variable][value_index]
    
    # get the formatted dataset title for use in the plot title.
    output_dataset_title = get_output_title(metadata, dataset_title)
    
    # specify the subset (or full) axes ranges to use for plotting. cutout_data is of the format [z-range, y-range, x-range, output value index].
    if dataset_title in ['sabl2048low', 'sabl2048high', 'stsabl2048low', 'stsabl2048high'] and variable == 'velocity':
        # zcoor_uv are the default z-axis coordinates for the 'velocity' variable of the 'sabl' datasets.
        plot_data = cutout_data[dataset_name].sel(xcoor = xcoor_values, 
                                                  ycoor = ycoor_values, 
                                                  zcoor_uv = zcoor_values,
                                                  values = value_index)
    else:
        plot_data = cutout_data[dataset_name].sel(xcoor = xcoor_values, 
                                                  ycoor = ycoor_values, 
                                                  zcoor = zcoor_values,
                                                  values = value_index)
    
    # raise exception if only one point is going to be plotted along more than 1 axis. a contour plot requires more 
    # than 1 point along 2 axes. this check is required in case the user specifies a stride along an axis that 
    # is >= number of datapoints along that axis.
    if plot_data.shape.count(1) > 1:
        raise Exception('the contour plot could not be created because more than 1 axis only had 1 datapoint')
    
    # map the axis name ('x', 'y', and 'z') to dx, dy, and dz respectively.
    axis_index_map = {'x': 0, 'y': 1, 'z': 2}
    # map the axis variable ('xcoor', 'ycoor', and 'zcoor') to xcoor_values, ycoor_values, and zcoor_values respectively.
    axis_coor_values_map = {'xcoor': xcoor_values, 'ycoor': ycoor_values, 'zcoor': zcoor_values}
    # map the axis variable ('xcoor', 'ycoor', and 'zcoor') to the plot ranges along the respective axis.
    axis_plot_ranges_map = {'xcoor': plot_ranges[0], 'ycoor': plot_ranges[1], 'zcoor': plot_ranges[2]}
    
    # create the figure.
    fig = plt.figure(figsize = (figsize, figsize), dpi = dpi)
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    cf = plot_data.plot(ax = ax, cmap = colormap, center = False)
    
    # get the x-axis and y-axis variable names (e.g. 'x' and 'y') before the axis labels are appended to.
    x_axis_variable = ax.get_xlabel()
    y_axis_variable = ax.get_ylabel()
    
    # get the min and max coordinate values along the plotted axes.
    plot_x_size = plot_data[x_axis_variable].max().values - plot_data[x_axis_variable].min().values
    plot_y_size = plot_data[y_axis_variable].max().values - plot_data[y_axis_variable].min().values
    
    # remove original colorbar.
    plt.delaxes(cf.colorbar.ax)
    # replot the colorbar with the correct orientation depending on which axis is larger.
    colorbar_orientation = 'vertical' if plot_y_size >= plot_x_size else 'horizontal'
    plt.colorbar(cf, shrink = 0.67, orientation = colorbar_orientation)
    if equal_aspect_ratio:
        plt.gca().set_aspect('equal')
    # colorbar labels.
    cbar = cf.colorbar
    cbar.set_label(f'{cube.var} field', fontsize = 14, labelpad = 15.0)
    # rotate the horizontal colorbar labels.
    if colorbar_orientation == 'horizontal':
        for label in cbar.ax.get_xticklabels():
            label.set_rotation(90)
    
    # plot labels.
    # convert the data domain plot label to the integer index in the [1, 8192] domain.
    original_axis_title = ax.get_title().replace('coor', '').replace('_uv', '').split('=')
    plane_axis = original_axis_title[0].strip()
    # plot_ranges_min = plot_ranges_max for plane_axis.
    plane_point = plot_ranges_min[axis_index_map[plane_axis]]
    axis_title = plane_axis + ' = ' + str(plane_point) + ', t = ' + str(time)
    title_str = f'{output_dataset_title}\n{variable} ({value_name}) contour plot ({axis_title})'
    # remove '_uv' from the axis variable names for display in the plot.
    x_axis_variable = x_axis_variable.replace('_uv', '')
    y_axis_variable = y_axis_variable.replace('_uv', '')
    x_label = x_axis_variable.strip().replace('coor', '')
    y_label = y_axis_variable.strip().replace('coor', '')
    x_axis_stride = cutout_data.attrs[f'{x_label}_step']
    y_axis_stride = cutout_data.attrs[f'{y_label}_step']
    plt.title(title_str, fontsize = 16, pad = 20)
    plt.xlabel(f'{x_label} (stride = {x_axis_stride})', fontsize = 14)
    plt.ylabel(f'{y_label} (stride = {y_axis_stride})', fontsize = 14)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    # adjust the axis ticks to the center of each datapoint.
    x_ticks = [axis_coor_values_map[x_axis_variable][0], axis_coor_values_map[x_axis_variable][-1]]
    y_ticks = [axis_coor_values_map[y_axis_variable][0], axis_coor_values_map[y_axis_variable][-1]]
    plt.xticks(x_ticks, [axis_plot_ranges_map[x_axis_variable][0], axis_plot_ranges_map[x_axis_variable][-1]])
    plt.yticks(y_ticks, [axis_plot_ranges_map[y_axis_variable][0], axis_plot_ranges_map[y_axis_variable][-1]])
    # save the figure.
    plt.tight_layout()
    plt.savefig(cube.output_path.joinpath(output_filename + '.png'))
    
    # show the figure in the notebook, and shrinks the dpi to make it easily visible.
    fig.set_dpi(67)
    plt.tight_layout()
    plt.show()
    
    # close the figure.
    plt.close()
    
    print('-----')
    print('contour plot created successfully.')
    sys.stdout.flush()

def cutout_values(cube, x, y, z, output_data, time, xyzt_axes_ranges, xyzt_strides):
    """
    retrieve data values for all of the specified points.
    """
    # dictionaries.
    # -----
    metadata = cube.metadata
    variable = cube.var
    dataset_title = cube.dataset_title
    dataset_name = variable + '_' + str(time).zfill(4)
    
    # remove the time axis from axes_ranges and strides.
    axes_ranges = xyzt_axes_ranges[:3]
    strides = xyzt_strides[:3]
    time_range = xyzt_axes_ranges[3]
    
    # minimum and maximum endpoints along each axis for the points the user requested.
    endpoints_min = np.array([np.min(x), np.min(y), np.min(z)], dtype = np.int32)
    endpoints_max = np.array([np.max(x), np.max(y), np.max(z)], dtype = np.int32)
    
    # exception_handling.
    # -----
    # raise exception if all of the user requested datapoints are not inside the bounds of the user box volume.
    if not(np.all(axes_ranges[:, 0] <= endpoints_min) and np.all(endpoints_max <= axes_ranges[:, 1])):
        raise Exception(f'the specified point(s) are not all bounded by the box volume defined by:\n{axes_ranges}')
        
    # raise exception if the time is not one of the queried timepoints.
    if time not in range(time_range[0], time_range[1] + 1, 1):
        raise Exception(f'the specified time ({time}) is not a queried time, t_range:\n{time_range}')
    
    # datasets that have an irregular y-grid.
    irregular_ygrid_datasets = get_irregular_mesh_ygrid_datasets(metadata, variable)
    irregular_zgrid_datasets = get_irregular_mesh_zgrid_datasets(metadata, variable)
    
    # convert the requested plot ranges to 0-based indices and then to their corresponding values in the data domain.
    xcoor_values = np.around(np.arange(endpoints_min[0] - 1, endpoints_max[0], strides[0], dtype = np.float32) * cube.dx, cube.decimals)
    xcoor_values += cube.coor_offsets[0]
    if dataset_title in irregular_ygrid_datasets:
        # note: this assumes that the y-axis of the irregular grid datasets is non-periodic.
        ycoor_values = cube.dy[np.arange(endpoints_min[1] - 1, endpoints_max[1], strides[1])]
    else:
        ycoor_values = np.around(np.arange(endpoints_min[1] - 1, endpoints_max[1], strides[1], dtype = np.float32) * cube.dy, cube.decimals)
        ycoor_values += cube.coor_offsets[1]
    if dataset_title in irregular_zgrid_datasets:
        # note: this assumes that the z-axis of the irregular grid datasets is non-periodic.
        zcoor_values = cube.dz[np.arange(endpoints_min[2] - 1, endpoints_max[2], strides[2])]
    else:
        zcoor_values = np.around(np.arange(endpoints_min[2] - 1, endpoints_max[2], strides[2], dtype = np.float32) * cube.dz, cube.decimals)
        zcoor_values += cube.coor_offsets[2]

    # value(s) corresponding to the specified (x, y, z) datapoint(s).
    if dataset_title in ['sabl2048low', 'sabl2048high', 'stsabl2048low', 'stsabl2048high'] and variable == 'velocity':
        # zcoor_uv are the default z-axis coordinates for the 'velocity' variable of the 'sabl' datasets.
        output_values = output_data[dataset_name].sel(xcoor = xcoor_values,
                                                      ycoor = ycoor_values,
                                                      zcoor_uv = zcoor_values)
        
        # add the zcoor_w coordinate to the returned xarray dataarray.
        output_values = output_values.assign_coords({'zcoor_w': output_values.zcoor_uv + (cube.dz / 2)})
        
        return output_values
    else:
        return output_data[dataset_name].sel(xcoor = xcoor_values,
                                             ycoor = ycoor_values,
                                             zcoor = zcoor_values)
